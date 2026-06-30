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
  }

  py::array_t<float> maxsim_batch(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      float scale) const {
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

    float* d_query = nullptr;
    float* d_output = nullptr;
    check_cuda(cudaMalloc(&d_query, query_bytes), "cudaMalloc query");
    check_cuda(cudaMalloc(&d_output, output_bytes), "cudaMalloc output");
    try {
      check_cuda(cudaMemcpy(d_query, query.data(), query_bytes, cudaMemcpyHostToDevice), "copy query to device");

      dim3 grid(num_docs_, batch);
      maxsim_batched_kernel<<<grid, kBlockThreads>>>(d_query, d_packed_, d_offsets_, d_output, batch, query_tokens, dim_, num_docs_, scale);
      check_cuda(cudaGetLastError(), "launch resident batched maxsim kernel");
      check_cuda(cudaDeviceSynchronize(), "synchronize resident batched maxsim kernel");

      py::array_t<float> output({batch, num_docs_});
      check_cuda(cudaMemcpy(output.mutable_data(), d_output, output_bytes, cudaMemcpyDeviceToHost), "copy output to host");

      cudaFree(d_query);
      cudaFree(d_output);
      return output;
    } catch (...) {
      cudaFree(d_query);
      cudaFree(d_output);
      throw;
    }
  }

  int dim() const { return dim_; }
  int num_docs() const { return num_docs_; }
  std::size_t packed_size() const { return packed_size_; }

 private:
  std::uint8_t* d_packed_ = nullptr;
  std::int64_t* d_offsets_ = nullptr;
  int dim_ = 0;
  int num_docs_ = 0;
  std::size_t packed_size_ = 0;
  std::size_t offsets_size_ = 0;
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
      .def_property_readonly("dim", &CudaPackedDocs::dim)
      .def_property_readonly("num_docs", &CudaPackedDocs::num_docs)
      .def_property_readonly("packed_size", &CudaPackedDocs::packed_size);
  m.def("maxsim_cuda", &maxsim_cuda, py::arg("query"), py::arg("packed"), py::arg("offsets"), py::arg("dim"), py::arg("scale") = 1.0F);
  m.def("maxsim_cuda_batch", &maxsim_cuda_batch, py::arg("query"), py::arg("packed"), py::arg("offsets"), py::arg("dim"), py::arg("scale") = 1.0F);
}
