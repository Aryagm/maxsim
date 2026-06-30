#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

constexpr int kBlockThreads = 128;
constexpr int kTopkThreads = 256;

void check_cuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
  }
}

__global__ void maxsim_batched_kernel(
    const float* query,
    const std::uint8_t* packed,
    const std::int64_t* offsets,
    float* output,
    int batch,
    int query_tokens,
    int dim,
    int num_docs,
    float scale) {
  const int doc_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float reductions[kBlockThreads];

  if (doc_idx >= num_docs || batch_idx >= batch) {
    return;
  }

  const int byte_dim = dim / 8;
  const std::int64_t start = offsets[doc_idx];
  const std::int64_t end = offsets[doc_idx + 1];
  float doc_score = 0.0F;

  for (int q = 0; q < query_tokens; ++q) {
    float local_best = -3.402823466e+38F;
    const float* query_row = query + (static_cast<std::int64_t>(batch_idx) * query_tokens + q) * dim;
    for (std::int64_t token = start + tid; token < end; token += blockDim.x) {
      float dot = 0.0F;
      for (int byte_idx = 0; byte_idx < byte_dim; ++byte_idx) {
        const std::uint8_t byte = packed[token * byte_dim + byte_idx];
        for (int bit = 0; bit < 8; ++bit) {
          const int dim_idx = byte_idx * 8 + bit;
          const float sign = ((byte >> bit) & 1U) ? 1.0F : -1.0F;
          dot += sign * query_row[dim_idx];
        }
      }
      local_best = dot > local_best ? dot : local_best;
    }

    reductions[tid] = local_best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float other = reductions[tid + stride];
        reductions[tid] = other > reductions[tid] ? other : reductions[tid];
      }
      __syncthreads();
    }
    if (tid == 0 && start != end) {
      doc_score += reductions[0];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[static_cast<std::int64_t>(batch_idx) * num_docs + doc_idx] = doc_score * scale;
  }
}

__device__ __forceinline__ float dot_dim128_unrolled(const float* query_row, const std::uint8_t* packed_row) {
  float dot = 0.0F;
#pragma unroll
  for (int byte_idx = 0; byte_idx < 16; ++byte_idx) {
    const std::uint8_t byte = packed_row[byte_idx];
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
      const int dim_idx = byte_idx * 8 + bit;
      const float sign = ((byte >> bit) & 1U) ? 1.0F : -1.0F;
      dot += sign * query_row[dim_idx];
    }
  }
  return dot;
}

__global__ void maxsim_batched_dim128_unrolled_kernel(
    const float* query,
    const std::uint8_t* packed,
    const std::int64_t* offsets,
    float* output,
    int batch,
    int query_tokens,
    int num_docs,
    float scale) {
  const int doc_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float reductions[kBlockThreads];

  if (doc_idx >= num_docs || batch_idx >= batch) {
    return;
  }

  const std::int64_t start = offsets[doc_idx];
  const std::int64_t end = offsets[doc_idx + 1];
  float doc_score = 0.0F;

  for (int q = 0; q < query_tokens; ++q) {
    float local_best = -3.402823466e+38F;
    const float* query_row = query + (static_cast<std::int64_t>(batch_idx) * query_tokens + q) * 128;
    for (std::int64_t token = start + tid; token < end; token += blockDim.x) {
      const std::uint8_t* packed_row = packed + token * 16;
      const float dot = dot_dim128_unrolled(query_row, packed_row);
      local_best = dot > local_best ? dot : local_best;
    }

    reductions[tid] = local_best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float other = reductions[tid + stride];
        reductions[tid] = other > reductions[tid] ? other : reductions[tid];
      }
      __syncthreads();
    }
    if (tid == 0 && start != end) {
      doc_score += reductions[0];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[static_cast<std::int64_t>(batch_idx) * num_docs + doc_idx] = doc_score * scale;
  }
}

__global__ void topk_kernel(
    const float* scores,
    float* top_scores,
    std::int64_t* top_indices,
    int batch,
    int num_docs,
    int k) {
  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float best_scores[kTopkThreads];
  __shared__ int best_indices[kTopkThreads];

  if (batch_idx >= batch) {
    return;
  }

  const float* row = scores + static_cast<std::int64_t>(batch_idx) * num_docs;
  float* out_scores = top_scores + static_cast<std::int64_t>(batch_idx) * k;
  std::int64_t* out_indices = top_indices + static_cast<std::int64_t>(batch_idx) * k;

  for (int rank = 0; rank < k; ++rank) {
    float local_score = -3.402823466e+38F;
    int local_doc = -1;
    for (int doc = tid; doc < num_docs; doc += blockDim.x) {
      bool already_selected = false;
      for (int previous = 0; previous < rank; ++previous) {
        if (out_indices[previous] == doc) {
          already_selected = true;
          break;
        }
      }
      if (already_selected) {
        continue;
      }

      const float score = row[doc];
      if (local_doc < 0 || score > local_score || (score == local_score && doc < local_doc)) {
        local_score = score;
        local_doc = doc;
      }
    }

    best_scores[tid] = local_score;
    best_indices[tid] = local_doc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float other_score = best_scores[tid + stride];
        const int other_doc = best_indices[tid + stride];
        const int current_doc = best_indices[tid];
        if (other_doc >= 0 && (current_doc < 0 || other_score > best_scores[tid] || (other_score == best_scores[tid] && other_doc < current_doc))) {
          best_scores[tid] = other_score;
          best_indices[tid] = other_doc;
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      out_scores[rank] = best_scores[0];
      out_indices[rank] = static_cast<std::int64_t>(best_indices[0]);
    }
    __syncthreads();
  }
}

void validate_inputs(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& query,
    const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& packed,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& offsets,
    int dim) {
  if (query.ndim() != 2) {
    throw std::invalid_argument("query must have shape [query_tokens, dim]");
  }
  if (packed.ndim() != 2) {
    throw std::invalid_argument("packed must have shape [num_doc_tokens, dim / 8]");
  }
  if (offsets.ndim() != 1) {
    throw std::invalid_argument("offsets must be a 1-D array");
  }
  if (dim <= 0 || dim % 8 != 0) {
    throw std::invalid_argument("dim must be positive and divisible by 8");
  }
  if (query.shape(1) != dim || packed.shape(1) != dim / 8) {
    throw std::invalid_argument("query/packed shapes do not match dim");
  }
}

void validate_batch_inputs(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& query,
    const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& packed,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& offsets,
    int dim) {
  if (query.ndim() != 3) {
    throw std::invalid_argument("query must have shape [batch, query_tokens, dim]");
  }
  if (packed.ndim() != 2) {
    throw std::invalid_argument("packed must have shape [num_doc_tokens, dim / 8]");
  }
  if (offsets.ndim() != 1) {
    throw std::invalid_argument("offsets must be a 1-D array");
  }
  if (dim <= 0 || dim % 8 != 0) {
    throw std::invalid_argument("dim must be positive and divisible by 8");
  }
  if (query.shape(2) != dim || packed.shape(1) != dim / 8) {
    throw std::invalid_argument("query/packed shapes do not match dim");
  }
}

class CudaPackedDocs {
 public:
  CudaPackedDocs(
      py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> packed,
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
      int dim)
      : dim_(dim),
        num_docs_(static_cast<int>(offsets.shape(0) - 1)),
        packed_size_(static_cast<std::size_t>(packed.size())),
        offsets_size_(static_cast<std::size_t>(offsets.size())) {
    if (packed.ndim() != 2) {
      throw std::invalid_argument("packed must have shape [num_doc_tokens, dim / 8]");
    }
    if (offsets.ndim() != 1) {
      throw std::invalid_argument("offsets must be a 1-D array");
    }
    if (dim <= 0 || dim % 8 != 0) {
      throw std::invalid_argument("dim must be positive and divisible by 8");
    }
    if (packed.shape(1) != dim / 8) {
      throw std::invalid_argument("packed shape does not match dim");
    }
    if (offsets.shape(0) < 2) {
      throw std::invalid_argument("offsets must contain at least [0, num_doc_tokens]");
    }

    const std::size_t packed_bytes = packed_size_ * sizeof(std::uint8_t);
    const std::size_t offsets_bytes = offsets_size_ * sizeof(std::int64_t);
    check_cuda(cudaMalloc(&d_packed_, packed_bytes), "cudaMalloc packed docs");
    check_cuda(cudaMalloc(&d_offsets_, offsets_bytes), "cudaMalloc offsets");
    try {
      check_cuda(cudaMemcpy(d_packed_, packed.data(), packed_bytes, cudaMemcpyHostToDevice), "copy packed docs to device");
      check_cuda(cudaMemcpy(d_offsets_, offsets.data(), offsets_bytes, cudaMemcpyHostToDevice), "copy offsets to device");
    } catch (...) {
      cudaFree(d_packed_);
      cudaFree(d_offsets_);
      d_packed_ = nullptr;
      d_offsets_ = nullptr;
      throw;
    }
  }

  CudaPackedDocs(const CudaPackedDocs&) = delete;
  CudaPackedDocs& operator=(const CudaPackedDocs&) = delete;

  ~CudaPackedDocs() {
    cudaFree(d_packed_);
    cudaFree(d_offsets_);
    cudaFree(d_query_);
    cudaFree(d_scores_);
    cudaFree(d_top_scores_);
    cudaFree(d_top_indices_);
  }

  py::array_t<float> maxsim_batch(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      float scale) {
    if (query.ndim() != 3) {
      throw std::invalid_argument("query must have shape [batch, query_tokens, dim]");
    }
    if (query.shape(2) != dim_) {
      throw std::invalid_argument("query shape does not match dim");
    }

    const int batch = static_cast<int>(query.shape(0));
    const int query_tokens = static_cast<int>(query.shape(1));
    const std::size_t query_bytes = static_cast<std::size_t>(query.size()) * sizeof(float);
    const std::size_t output_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(num_docs_) * sizeof(float);

    ensure_device_capacity(&d_query_, &query_capacity_, query_bytes, "cudaMalloc resident query cache");
    ensure_device_capacity(&d_scores_, &scores_capacity_, output_bytes, "cudaMalloc resident score cache");

    check_cuda(cudaMemcpy(d_query_, query.data(), query_bytes, cudaMemcpyHostToDevice), "copy query to device");

    launch_maxsim(d_query_, d_scores_, batch, query_tokens, scale, "resident batched maxsim");
    check_cuda(cudaDeviceSynchronize(), "synchronize resident batched maxsim kernel");

    py::array_t<float> output({batch, num_docs_});
    check_cuda(cudaMemcpy(output.mutable_data(), d_scores_, output_bytes, cudaMemcpyDeviceToHost), "copy output to host");
    return output;
  }

  py::tuple topk_batch(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      int k,
      float scale) {
    if (k < 1 || k > num_docs_) {
      throw std::invalid_argument("k must be between 1 and num_docs");
    }
    if (query.ndim() != 3) {
      throw std::invalid_argument("query must have shape [batch, query_tokens, dim]");
    }
    if (query.shape(2) != dim_) {
      throw std::invalid_argument("query shape does not match dim");
    }

    const int batch = static_cast<int>(query.shape(0));
    const int query_tokens = static_cast<int>(query.shape(1));
    const std::size_t query_bytes = static_cast<std::size_t>(query.size()) * sizeof(float);
    const std::size_t scores_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(num_docs_) * sizeof(float);
    const std::size_t top_scores_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) * sizeof(float);
    const std::size_t top_indices_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) * sizeof(std::int64_t);

    ensure_device_capacity(&d_query_, &query_capacity_, query_bytes, "cudaMalloc resident query cache");
    ensure_device_capacity(&d_scores_, &scores_capacity_, scores_bytes, "cudaMalloc resident full score cache");
    ensure_device_capacity(&d_top_scores_, &top_scores_capacity_, top_scores_bytes, "cudaMalloc resident top score cache");
    ensure_device_capacity(&d_top_indices_, &top_indices_capacity_, top_indices_bytes, "cudaMalloc resident top index cache");

    check_cuda(cudaMemcpy(d_query_, query.data(), query_bytes, cudaMemcpyHostToDevice), "copy query to device");

    launch_maxsim(d_query_, d_scores_, batch, query_tokens, scale, "topk maxsim");
    topk_kernel<<<batch, kTopkThreads>>>(d_scores_, d_top_scores_, d_top_indices_, batch, num_docs_, k);
    check_cuda(cudaGetLastError(), "launch topk selection kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize topk kernels");

    py::array_t<float> output_scores({batch, k});
    py::array_t<std::int64_t> output_indices({batch, k});
    check_cuda(cudaMemcpy(output_scores.mutable_data(), d_top_scores_, top_scores_bytes, cudaMemcpyDeviceToHost), "copy top scores to host");
    check_cuda(cudaMemcpy(output_indices.mutable_data(), d_top_indices_, top_indices_bytes, cudaMemcpyDeviceToHost), "copy top indices to host");
    return py::make_tuple(output_scores, output_indices);
  }

  int dim() const { return dim_; }
  int num_docs() const { return num_docs_; }
  std::size_t packed_size() const { return packed_size_; }
  const char* maxsim_kernel_variant() const {
    return use_dim128_unrolled() ? "dim128_unrolled" : "generic";
  }

 private:
  template <typename T>
  void ensure_device_capacity(T** ptr, std::size_t* capacity, std::size_t bytes, const char* action) {
    if (bytes <= *capacity) {
      return;
    }
    T* next = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&next), bytes), action);
    cudaFree(*ptr);
    *ptr = next;
    *capacity = bytes;
  }

  void launch_maxsim(float* d_query, float* d_output, int batch, int query_tokens, float scale, const char* label) {
    dim3 grid(num_docs_, batch);
    if (use_dim128_unrolled()) {
      maxsim_batched_dim128_unrolled_kernel<<<grid, kBlockThreads>>>(d_query, d_packed_, d_offsets_, d_output, batch, query_tokens, num_docs_, scale);
      check_cuda(cudaGetLastError(), (std::string("launch ") + label + " dim128 unrolled kernel").c_str());
      return;
    }

    maxsim_batched_kernel<<<grid, kBlockThreads>>>(d_query, d_packed_, d_offsets_, d_output, batch, query_tokens, dim_, num_docs_, scale);
    check_cuda(cudaGetLastError(), (std::string("launch ") + label + " generic kernel").c_str());
  }

  bool use_dim128_unrolled() const {
    return dim_ == 128 && num_docs_ <= 128;
  }

  std::uint8_t* d_packed_ = nullptr;
  std::int64_t* d_offsets_ = nullptr;
  float* d_query_ = nullptr;
  float* d_scores_ = nullptr;
  float* d_top_scores_ = nullptr;
  std::int64_t* d_top_indices_ = nullptr;
  int dim_ = 0;
  int num_docs_ = 0;
  std::size_t packed_size_ = 0;
  std::size_t offsets_size_ = 0;
  std::size_t query_capacity_ = 0;
  std::size_t scores_capacity_ = 0;
  std::size_t top_scores_capacity_ = 0;
  std::size_t top_indices_capacity_ = 0;
};

py::array_t<float> maxsim_cuda(
    py::array_t<float, py::array::c_style | py::array::forcecast> query,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> packed,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
    int dim,
    float scale) {
  validate_inputs(query, packed, offsets, dim);

  const int batch = 1;
  const int query_tokens = static_cast<int>(query.shape(0));
  const int num_docs = static_cast<int>(offsets.shape(0) - 1);
  const std::size_t query_bytes = static_cast<std::size_t>(query.size()) * sizeof(float);
  const std::size_t packed_bytes = static_cast<std::size_t>(packed.size()) * sizeof(std::uint8_t);
  const std::size_t offsets_bytes = static_cast<std::size_t>(offsets.size()) * sizeof(std::int64_t);
  const std::size_t output_bytes = static_cast<std::size_t>(num_docs) * sizeof(float);

  float* d_query = nullptr;
  std::uint8_t* d_packed = nullptr;
  std::int64_t* d_offsets = nullptr;
  float* d_output = nullptr;

  check_cuda(cudaMalloc(&d_query, query_bytes), "cudaMalloc query");
  check_cuda(cudaMalloc(&d_packed, packed_bytes), "cudaMalloc packed");
  check_cuda(cudaMalloc(&d_offsets, offsets_bytes), "cudaMalloc offsets");
  check_cuda(cudaMalloc(&d_output, output_bytes), "cudaMalloc output");

  try {
    check_cuda(cudaMemcpy(d_query, query.data(), query_bytes, cudaMemcpyHostToDevice), "copy query to device");
    check_cuda(cudaMemcpy(d_packed, packed.data(), packed_bytes, cudaMemcpyHostToDevice), "copy packed docs to device");
    check_cuda(cudaMemcpy(d_offsets, offsets.data(), offsets_bytes, cudaMemcpyHostToDevice), "copy offsets to device");

    dim3 grid(num_docs, batch);
    maxsim_batched_kernel<<<grid, kBlockThreads>>>(d_query, d_packed, d_offsets, d_output, batch, query_tokens, dim, num_docs, scale);
    check_cuda(cudaGetLastError(), "launch maxsim kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize maxsim kernel");

    py::array_t<float> output({num_docs});
    check_cuda(cudaMemcpy(output.mutable_data(), d_output, output_bytes, cudaMemcpyDeviceToHost), "copy output to host");

    cudaFree(d_query);
    cudaFree(d_packed);
    cudaFree(d_offsets);
    cudaFree(d_output);
    return output;
  } catch (...) {
    cudaFree(d_query);
    cudaFree(d_packed);
    cudaFree(d_offsets);
    cudaFree(d_output);
    throw;
  }
}

py::array_t<float> maxsim_cuda_batch(
    py::array_t<float, py::array::c_style | py::array::forcecast> query,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> packed,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
    int dim,
    float scale) {
  validate_batch_inputs(query, packed, offsets, dim);

  const int batch = static_cast<int>(query.shape(0));
  const int query_tokens = static_cast<int>(query.shape(1));
  const int num_docs = static_cast<int>(offsets.shape(0) - 1);
  const std::size_t query_bytes = static_cast<std::size_t>(query.size()) * sizeof(float);
  const std::size_t packed_bytes = static_cast<std::size_t>(packed.size()) * sizeof(std::uint8_t);
  const std::size_t offsets_bytes = static_cast<std::size_t>(offsets.size()) * sizeof(std::int64_t);
  const std::size_t output_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(num_docs) * sizeof(float);

  float* d_query = nullptr;
  std::uint8_t* d_packed = nullptr;
  std::int64_t* d_offsets = nullptr;
  float* d_output = nullptr;

  check_cuda(cudaMalloc(&d_query, query_bytes), "cudaMalloc query");
  check_cuda(cudaMalloc(&d_packed, packed_bytes), "cudaMalloc packed");
  check_cuda(cudaMalloc(&d_offsets, offsets_bytes), "cudaMalloc offsets");
  check_cuda(cudaMalloc(&d_output, output_bytes), "cudaMalloc output");

  try {
    check_cuda(cudaMemcpy(d_query, query.data(), query_bytes, cudaMemcpyHostToDevice), "copy query to device");
    check_cuda(cudaMemcpy(d_packed, packed.data(), packed_bytes, cudaMemcpyHostToDevice), "copy packed docs to device");
    check_cuda(cudaMemcpy(d_offsets, offsets.data(), offsets_bytes, cudaMemcpyHostToDevice), "copy offsets to device");

    dim3 grid(num_docs, batch);
    maxsim_batched_kernel<<<grid, kBlockThreads>>>(d_query, d_packed, d_offsets, d_output, batch, query_tokens, dim, num_docs, scale);
    check_cuda(cudaGetLastError(), "launch batched maxsim kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize batched maxsim kernel");

    py::array_t<float> output({batch, num_docs});
    check_cuda(cudaMemcpy(output.mutable_data(), d_output, output_bytes, cudaMemcpyDeviceToHost), "copy output to host");

    cudaFree(d_query);
    cudaFree(d_packed);
    cudaFree(d_offsets);
    cudaFree(d_output);
    return output;
  } catch (...) {
    cudaFree(d_query);
    cudaFree(d_packed);
    cudaFree(d_offsets);
    cudaFree(d_output);
    throw;
  }
}

}  // namespace

PYBIND11_MODULE(_bitmax_cuda, m) {
  m.doc() = "Optional CUDA kernels for bitmax";
  py::class_<CudaPackedDocs>(m, "CudaPackedDocs")
      .def(py::init<
           py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>,
           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>,
           int>())
      .def("maxsim_batch", &CudaPackedDocs::maxsim_batch, py::arg("query"), py::arg("scale") = 1.0F)
      .def("topk_batch", &CudaPackedDocs::topk_batch, py::arg("query"), py::arg("k"), py::arg("scale") = 1.0F)
      .def_property_readonly("dim", &CudaPackedDocs::dim)
      .def_property_readonly("num_docs", &CudaPackedDocs::num_docs)
      .def_property_readonly("packed_size", &CudaPackedDocs::packed_size)
      .def_property_readonly("maxsim_kernel_variant", &CudaPackedDocs::maxsim_kernel_variant);
  m.def("maxsim_cuda", &maxsim_cuda, py::arg("query"), py::arg("packed"), py::arg("offsets"), py::arg("dim"), py::arg("scale") = 1.0F);
  m.def("maxsim_cuda_batch", &maxsim_cuda_batch, py::arg("query"), py::arg("packed"), py::arg("offsets"), py::arg("dim"), py::arg("scale") = 1.0F);
}
