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
    float scale,
    const float* scale_vector) {
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
    const float doc_scale = scale_vector == nullptr ? scale : scale_vector[doc_idx];
    output[static_cast<std::int64_t>(batch_idx) * num_docs + doc_idx] = doc_score * doc_scale;
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

__global__ void build_query_lut_dim128_kernel(
    const float* query,
    float* query_lut,
    int batch,
    int query_tokens) {
  const int total = batch * query_tokens * 16 * 256;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
    const int byte_value = idx & 255;
    const int byte_idx = (idx >> 8) & 15;
    const int q = (idx >> 12) % query_tokens;
    const int batch_idx = idx / (query_tokens * 16 * 256);
    const float* query_row = query + (static_cast<std::int64_t>(batch_idx) * query_tokens + q) * 128;

    float dot = 0.0F;
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
      const int dim_idx = byte_idx * 8 + bit;
      const float sign = ((byte_value >> bit) & 1U) ? 1.0F : -1.0F;
      dot += sign * query_row[dim_idx];
    }
    query_lut[idx] = dot;
  }
}

__device__ __forceinline__ float dot_dim128_lut(const float* query_lut_row, const std::uint8_t* packed_row) {
  float dot = 0.0F;
#pragma unroll
  for (int byte_idx = 0; byte_idx < 16; ++byte_idx) {
    dot += query_lut_row[byte_idx * 256 + packed_row[byte_idx]];
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
    float scale,
    const float* scale_vector) {
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
    const float doc_scale = scale_vector == nullptr ? scale : scale_vector[doc_idx];
    output[static_cast<std::int64_t>(batch_idx) * num_docs + doc_idx] = doc_score * doc_scale;
  }
}

__global__ void maxsim_batched_dim128_lut_kernel(
    const float* query_lut,
    const std::uint8_t* packed,
    const std::int64_t* offsets,
    float* output,
    int batch,
    int query_tokens,
    int num_docs,
    float scale,
    const float* scale_vector) {
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
    const float* query_lut_row = query_lut + (static_cast<std::int64_t>(batch_idx) * query_tokens + q) * 16 * 256;
    for (std::int64_t token = start + tid; token < end; token += blockDim.x) {
      const std::uint8_t* packed_row = packed + token * 16;
      const float dot = dot_dim128_lut(query_lut_row, packed_row);
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
    const float doc_scale = scale_vector == nullptr ? scale : scale_vector[doc_idx];
    output[static_cast<std::int64_t>(batch_idx) * num_docs + doc_idx] = doc_score * doc_scale;
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

__device__ __forceinline__ bool score_is_better(float candidate_score, int candidate_doc, float current_score, std::int64_t current_doc) {
  return candidate_score > current_score || (candidate_score == current_score && (current_doc < 0 || candidate_doc < current_doc));
}

__device__ void insert_topk_candidate(float* scores, std::int64_t* indices, int k, float candidate_score, int candidate_doc) {
  for (int pos = 0; pos < k; ++pos) {
    if (!score_is_better(candidate_score, candidate_doc, scores[pos], indices[pos])) {
      continue;
    }
    for (int shift = k - 1; shift > pos; --shift) {
      scores[shift] = scores[shift - 1];
      indices[shift] = indices[shift - 1];
    }
    scores[pos] = candidate_score;
    indices[pos] = static_cast<std::int64_t>(candidate_doc);
    return;
  }
}

__global__ void init_streaming_topk_kernel(
    float* top_scores,
    std::int64_t* top_indices,
    int* locks,
    int batch,
    int k) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * k;
  for (int item = idx; item < total; item += blockDim.x * gridDim.x) {
    top_scores[item] = -3.402823466e+38F;
    top_indices[item] = -1;
  }
  for (int batch_idx = idx; batch_idx < batch; batch_idx += blockDim.x * gridDim.x) {
    locks[batch_idx] = 0;
  }
}

__global__ void streaming_topk_batched_kernel(
    const float* query,
    const std::uint8_t* packed,
    const std::int64_t* offsets,
    float* top_scores,
    std::int64_t* top_indices,
    int* locks,
    int batch,
    int query_tokens,
    int dim,
    int num_docs,
    int k,
    float scale,
    const float* scale_vector) {
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
    const float doc_scale = scale_vector == nullptr ? scale : scale_vector[doc_idx];
    const float scaled_score = doc_score * doc_scale;
    int* lock = locks + batch_idx;
    while (atomicCAS(lock, 0, 1) != 0) {
    }
    insert_topk_candidate(
        top_scores + static_cast<std::int64_t>(batch_idx) * k,
        top_indices + static_cast<std::int64_t>(batch_idx) * k,
        k,
        scaled_score,
        doc_idx);
    __threadfence();
    atomicExch(lock, 0);
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
    cudaFree(d_scale_);
    cudaFree(d_query_);
    cudaFree(d_query_lut_);
    cudaFree(d_scores_);
    cudaFree(d_top_scores_);
    cudaFree(d_top_indices_);
    cudaFree(d_top_locks_);
  }

  void set_scale_vector(py::array_t<float, py::array::c_style | py::array::forcecast> scale) {
    if (scale.ndim() != 1 || scale.shape(0) != num_docs_) {
      throw std::invalid_argument("scale vector must have shape [num_docs]");
    }

    const std::size_t scale_bytes = static_cast<std::size_t>(num_docs_) * sizeof(float);
    ensure_device_capacity(&d_scale_, &scale_capacity_, scale_bytes, "cudaMalloc resident scale vector");
    check_cuda(cudaMemcpy(d_scale_, scale.data(), scale_bytes, cudaMemcpyHostToDevice), "copy scale vector to device");
    has_scale_vector_ = true;
  }

  void clear_scale_vector() {
    has_scale_vector_ = false;
  }

  py::array_t<float> maxsim_batch(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      float scale,
      bool use_scale_vector = false) {
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

    launch_maxsim(d_query_, d_scores_, batch, query_tokens, scale, scale_vector_ptr(use_scale_vector), "resident batched maxsim");
    check_cuda(cudaDeviceSynchronize(), "synchronize resident batched maxsim kernel");

    py::array_t<float> output({batch, num_docs_});
    check_cuda(cudaMemcpy(output.mutable_data(), d_scores_, output_bytes, cudaMemcpyDeviceToHost), "copy output to host");
    return output;
  }

  py::tuple topk_batch(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      int k,
      float scale,
      bool use_scale_vector = false) {
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

    launch_maxsim(d_query_, d_scores_, batch, query_tokens, scale, scale_vector_ptr(use_scale_vector), "topk maxsim");
    topk_kernel<<<batch, kTopkThreads>>>(d_scores_, d_top_scores_, d_top_indices_, batch, num_docs_, k);
    check_cuda(cudaGetLastError(), "launch topk selection kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize topk kernels");

    py::array_t<float> output_scores({batch, k});
    py::array_t<std::int64_t> output_indices({batch, k});
    check_cuda(cudaMemcpy(output_scores.mutable_data(), d_top_scores_, top_scores_bytes, cudaMemcpyDeviceToHost), "copy top scores to host");
    check_cuda(cudaMemcpy(output_indices.mutable_data(), d_top_indices_, top_indices_bytes, cudaMemcpyDeviceToHost), "copy top indices to host");
    return py::make_tuple(output_scores, output_indices);
  }

  py::tuple topk_lut_batch(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      int k,
      float scale,
      bool use_scale_vector = false) {
    if (dim_ != 128) {
      throw std::invalid_argument("topk_lut_batch currently requires dim=128");
    }
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
    const std::size_t lut_values = static_cast<std::size_t>(batch) * static_cast<std::size_t>(query_tokens) * 16U * 256U;
    const std::size_t lut_bytes = lut_values * sizeof(float);
    const std::size_t scores_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(num_docs_) * sizeof(float);
    const std::size_t top_scores_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) * sizeof(float);
    const std::size_t top_indices_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) * sizeof(std::int64_t);

    ensure_device_capacity(&d_query_, &query_capacity_, query_bytes, "cudaMalloc resident query cache");
    ensure_device_capacity(&d_query_lut_, &query_lut_capacity_, lut_bytes, "cudaMalloc resident query LUT cache");
    ensure_device_capacity(&d_scores_, &scores_capacity_, scores_bytes, "cudaMalloc resident LUT score cache");
    ensure_device_capacity(&d_top_scores_, &top_scores_capacity_, top_scores_bytes, "cudaMalloc resident LUT top score cache");
    ensure_device_capacity(&d_top_indices_, &top_indices_capacity_, top_indices_bytes, "cudaMalloc resident LUT top index cache");

    check_cuda(cudaMemcpy(d_query_, query.data(), query_bytes, cudaMemcpyHostToDevice), "copy query to device");
    build_query_lut_dim128(d_query_, d_query_lut_, batch, query_tokens);
    launch_maxsim_lut_dim128(d_query_lut_, d_scores_, batch, query_tokens, scale, scale_vector_ptr(use_scale_vector), "topk LUT maxsim");
    topk_kernel<<<batch, kTopkThreads>>>(d_scores_, d_top_scores_, d_top_indices_, batch, num_docs_, k);
    check_cuda(cudaGetLastError(), "launch LUT topk selection kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize LUT topk kernels");

    py::array_t<float> output_scores({batch, k});
    py::array_t<std::int64_t> output_indices({batch, k});
    check_cuda(cudaMemcpy(output_scores.mutable_data(), d_top_scores_, top_scores_bytes, cudaMemcpyDeviceToHost), "copy LUT top scores to host");
    check_cuda(cudaMemcpy(output_indices.mutable_data(), d_top_indices_, top_indices_bytes, cudaMemcpyDeviceToHost), "copy LUT top indices to host");
    return py::make_tuple(output_scores, output_indices);
  }

  py::tuple streaming_topk_batch(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      int k,
      float scale,
      bool use_scale_vector = false) {
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
    const std::size_t top_scores_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) * sizeof(float);
    const std::size_t top_indices_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) * sizeof(std::int64_t);
    const std::size_t top_locks_bytes = static_cast<std::size_t>(batch) * sizeof(int);

    ensure_device_capacity(&d_query_, &query_capacity_, query_bytes, "cudaMalloc resident query cache");
    ensure_device_capacity(&d_top_scores_, &top_scores_capacity_, top_scores_bytes, "cudaMalloc resident streaming top score cache");
    ensure_device_capacity(&d_top_indices_, &top_indices_capacity_, top_indices_bytes, "cudaMalloc resident streaming top index cache");
    ensure_device_capacity(&d_top_locks_, &top_locks_capacity_, top_locks_bytes, "cudaMalloc resident streaming top lock cache");

    check_cuda(cudaMemcpy(d_query_, query.data(), query_bytes, cudaMemcpyHostToDevice), "copy query to device");

    const int init_blocks = (batch * k + kBlockThreads - 1) / kBlockThreads;
    const int safe_init_blocks = init_blocks > 0 ? init_blocks : 1;
    init_streaming_topk_kernel<<<safe_init_blocks, kBlockThreads>>>(d_top_scores_, d_top_indices_, d_top_locks_, batch, k);
    check_cuda(cudaGetLastError(), "launch streaming topk init kernel");

    dim3 grid(num_docs_, batch);
    streaming_topk_batched_kernel<<<grid, kBlockThreads>>>(
        d_query_,
        d_packed_,
        d_offsets_,
        d_top_scores_,
        d_top_indices_,
        d_top_locks_,
        batch,
        query_tokens,
        dim_,
        num_docs_,
        k,
        scale,
        scale_vector_ptr(use_scale_vector));
    check_cuda(cudaGetLastError(), "launch streaming topk maxsim kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize streaming topk kernels");

    py::array_t<float> output_scores({batch, k});
    py::array_t<std::int64_t> output_indices({batch, k});
    check_cuda(cudaMemcpy(output_scores.mutable_data(), d_top_scores_, top_scores_bytes, cudaMemcpyDeviceToHost), "copy streaming top scores to host");
    check_cuda(cudaMemcpy(output_indices.mutable_data(), d_top_indices_, top_indices_bytes, cudaMemcpyDeviceToHost), "copy streaming top indices to host");
    return py::make_tuple(output_scores, output_indices);
  }

  int dim() const { return dim_; }
  int num_docs() const { return num_docs_; }
  std::size_t packed_size() const { return packed_size_; }
  bool has_scale_vector() const { return has_scale_vector_; }
  std::size_t scale_vector_size() const { return has_scale_vector_ ? static_cast<std::size_t>(num_docs_) : 0; }
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

  const float* scale_vector_ptr(bool use_scale_vector) const {
    if (!use_scale_vector) {
      return nullptr;
    }
    if (!has_scale_vector_) {
      throw std::invalid_argument("scale vector was requested but is not loaded on this CUDA handle");
    }
    return d_scale_;
  }

  void launch_maxsim(float* d_query, float* d_output, int batch, int query_tokens, float scale, const float* d_scale_vector, const char* label) {
    dim3 grid(num_docs_, batch);
    if (use_dim128_unrolled()) {
      maxsim_batched_dim128_unrolled_kernel<<<grid, kBlockThreads>>>(d_query, d_packed_, d_offsets_, d_output, batch, query_tokens, num_docs_, scale, d_scale_vector);
      check_cuda(cudaGetLastError(), (std::string("launch ") + label + " dim128 unrolled kernel").c_str());
      return;
    }

    maxsim_batched_kernel<<<grid, kBlockThreads>>>(d_query, d_packed_, d_offsets_, d_output, batch, query_tokens, dim_, num_docs_, scale, d_scale_vector);
    check_cuda(cudaGetLastError(), (std::string("launch ") + label + " generic kernel").c_str());
  }

  void build_query_lut_dim128(float* d_query, float* d_query_lut, int batch, int query_tokens) {
    const int total = batch * query_tokens * 16 * 256;
    const int blocks = (total + 255) / 256;
    build_query_lut_dim128_kernel<<<blocks, 256>>>(d_query, d_query_lut, batch, query_tokens);
    check_cuda(cudaGetLastError(), "launch dim128 query LUT build kernel");
  }

  void launch_maxsim_lut_dim128(float* d_query_lut, float* d_output, int batch, int query_tokens, float scale, const float* d_scale_vector, const char* label) {
    dim3 grid(num_docs_, batch);
    maxsim_batched_dim128_lut_kernel<<<grid, kBlockThreads>>>(d_query_lut, d_packed_, d_offsets_, d_output, batch, query_tokens, num_docs_, scale, d_scale_vector);
    check_cuda(cudaGetLastError(), (std::string("launch ") + label + " dim128 LUT kernel").c_str());
  }

  bool use_dim128_unrolled() const {
    return dim_ == 128 && num_docs_ <= 128;
  }

  std::uint8_t* d_packed_ = nullptr;
  std::int64_t* d_offsets_ = nullptr;
  float* d_scale_ = nullptr;
  float* d_query_ = nullptr;
  float* d_query_lut_ = nullptr;
  float* d_scores_ = nullptr;
  float* d_top_scores_ = nullptr;
  std::int64_t* d_top_indices_ = nullptr;
  int* d_top_locks_ = nullptr;
  int dim_ = 0;
  int num_docs_ = 0;
  std::size_t packed_size_ = 0;
  std::size_t offsets_size_ = 0;
  std::size_t scale_capacity_ = 0;
  std::size_t query_capacity_ = 0;
  std::size_t query_lut_capacity_ = 0;
  std::size_t scores_capacity_ = 0;
  std::size_t top_scores_capacity_ = 0;
  std::size_t top_indices_capacity_ = 0;
  std::size_t top_locks_capacity_ = 0;
  bool has_scale_vector_ = false;
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
    maxsim_batched_kernel<<<grid, kBlockThreads>>>(d_query, d_packed, d_offsets, d_output, batch, query_tokens, dim, num_docs, scale, nullptr);
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
    maxsim_batched_kernel<<<grid, kBlockThreads>>>(d_query, d_packed, d_offsets, d_output, batch, query_tokens, dim, num_docs, scale, nullptr);
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
      .def("set_scale_vector", &CudaPackedDocs::set_scale_vector, py::arg("scale"))
      .def("clear_scale_vector", &CudaPackedDocs::clear_scale_vector)
      .def("maxsim_batch", &CudaPackedDocs::maxsim_batch, py::arg("query"), py::arg("scale") = 1.0F, py::arg("use_scale_vector") = false)
      .def("topk_batch", &CudaPackedDocs::topk_batch, py::arg("query"), py::arg("k"), py::arg("scale") = 1.0F, py::arg("use_scale_vector") = false)
      .def("topk_lut_batch", &CudaPackedDocs::topk_lut_batch, py::arg("query"), py::arg("k"), py::arg("scale") = 1.0F, py::arg("use_scale_vector") = false)
      .def("streaming_topk_batch", &CudaPackedDocs::streaming_topk_batch, py::arg("query"), py::arg("k"), py::arg("scale") = 1.0F, py::arg("use_scale_vector") = false)
      .def_property_readonly("dim", &CudaPackedDocs::dim)
      .def_property_readonly("num_docs", &CudaPackedDocs::num_docs)
      .def_property_readonly("packed_size", &CudaPackedDocs::packed_size)
      .def_property_readonly("has_scale_vector", &CudaPackedDocs::has_scale_vector)
      .def_property_readonly("scale_vector_size", &CudaPackedDocs::scale_vector_size)
      .def_property_readonly("maxsim_kernel_variant", &CudaPackedDocs::maxsim_kernel_variant);
  m.def("maxsim_cuda", &maxsim_cuda, py::arg("query"), py::arg("packed"), py::arg("offsets"), py::arg("dim"), py::arg("scale") = 1.0F);
  m.def("maxsim_cuda_batch", &maxsim_cuda_batch, py::arg("query"), py::arg("packed"), py::arg("offsets"), py::arg("dim"), py::arg("scale") = 1.0F);
}
