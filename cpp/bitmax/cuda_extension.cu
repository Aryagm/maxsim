#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

constexpr int kBlockThreads = 128;
constexpr int kTopkThreads = 256;

// Runtime-tunable routing gate for the dim128 unrolled scoring kernel.
// Corpora with at least this many average tokens per document use the
// unrolled kernel even above 128 documents; shorter-doc corpora keep the
// generic kernel (the documented rerank_512/rerank_4096 regression shapes).
// Threshold from benchmark-results/dim128-gate-sweep.json on RTX 4090:
// unrolled wins ~2x at 750 tokens/doc for 256/1000/5000 docs and loses
// (0.71-0.86x) at 32-128 tokens/doc.
int g_dim128_unrolled_min_avg_tokens = 256;

// The q-tiled one-pass kernel is NOT routed by default: measured on RTX 4090
// (benchmark-results/qtile-sweep-*.json) it is ~9x slower than the unrolled
// kernel at every corpus size (12MB-300MB packed), because each block's
// document rows are already L1-resident across query-token iterations while
// q-tiling forces per-FMA shared-memory reads instead of register-held query
// values. Kept for evidence and future architectures; enable via
// set_dim128_qtile_min_packed_bytes.
std::size_t g_dim128_qtile_min_packed_bytes = ~static_cast<std::size_t>(0);

constexpr int kQTile = 8;

__global__ void maxsim_batched_dim128_qtile_kernel(
    const float* query,
    const std::uint8_t* packed,
    const std::int64_t* offsets,
    float* output,
    int batch,
    int query_tokens,
    int num_docs,
    float scale,
    const float* scale_vector,
    const float* token_scales) {
  const int doc_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float reductions[kBlockThreads];
  __shared__ float query_tile[kQTile][128];

  if (doc_idx >= num_docs || batch_idx >= batch) {
    return;
  }

  const std::int64_t start = offsets[doc_idx];
  const std::int64_t end = offsets[doc_idx + 1];
  float doc_score = 0.0F;

  for (int tile_start = 0; tile_start < query_tokens; tile_start += kQTile) {
    for (int idx = tid; idx < kQTile * 128; idx += blockDim.x) {
      const int q = tile_start + idx / 128;
      query_tile[idx / 128][idx % 128] =
          q < query_tokens
              ? query[(static_cast<std::int64_t>(batch_idx) * query_tokens + q) * 128 + (idx % 128)]
              : 0.0F;
    }
    __syncthreads();

    float best[kQTile];
#pragma unroll
    for (int slot = 0; slot < kQTile; ++slot) {
      best[slot] = -3.402823466e+38F;
    }

    for (std::int64_t token = start + tid; token < end; token += blockDim.x) {
      const uint4 row = *reinterpret_cast<const uint4*>(packed + token * 16);
      const std::uint32_t words[4] = {row.x, row.y, row.z, row.w};
      const float token_scale = token_scales != nullptr ? token_scales[token] : 1.0F;
#pragma unroll
      for (int slot = 0; slot < kQTile; ++slot) {
        float dot = 0.0F;
#pragma unroll
        for (int word = 0; word < 4; ++word) {
          const std::uint32_t bits = words[word];
#pragma unroll
          for (int bit = 0; bit < 32; ++bit) {
            const float value = query_tile[slot][32 * word + bit];
            dot += ((bits >> bit) & 1U) ? value : -value;
          }
        }
        dot *= token_scale;
        best[slot] = dot > best[slot] ? dot : best[slot];
      }
    }

#pragma unroll
    for (int slot = 0; slot < kQTile; ++slot) {
      __syncthreads();
      reductions[tid] = best[slot];
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
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float doc_scale = scale_vector == nullptr ? scale : scale_vector[doc_idx];
    output[static_cast<std::int64_t>(batch_idx) * num_docs + doc_idx] = doc_score * doc_scale;
  }
}

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
    const float* scale_vector,
    const float* token_scales) {
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
      if (token_scales != nullptr) {
        dot *= token_scales[token];
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

__global__ void weight_query_dims_kernel(
    float* query,
    const float* weights,
    std::int64_t total,
    int dim) {
  for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    query[idx] *= weights[idx % dim];
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
    const float* scale_vector,
    const float* token_scales) {
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
      float dot = dot_dim128_unrolled(query_row, packed_row);
      if (token_scales != nullptr) {
        dot *= token_scales[token];
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

__global__ void maxsim_batched_dim128_lut_kernel(
    const float* query_lut,
    const std::uint8_t* packed,
    const std::int64_t* offsets,
    float* output,
    int batch,
    int query_tokens,
    int num_docs,
    float scale,
    const float* scale_vector,
    const float* token_scales) {
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
      float dot = dot_dim128_lut(query_lut_row, packed_row);
      if (token_scales != nullptr) {
        dot *= token_scales[token];
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

__device__ __forceinline__ int signed_int4(std::uint8_t nibble) {
  const int value = static_cast<int>(nibble & 0x0FU);
  return value >= 8 ? value - 16 : value;
}

__device__ __forceinline__ float dot_int4(const float* query_row, const std::uint8_t* packed_row, int dim) {
  float dot = 0.0F;
  for (int byte_idx = 0; byte_idx < dim / 2; ++byte_idx) {
    const std::uint8_t byte = packed_row[byte_idx];
    const int dim_idx = byte_idx * 2;
    dot += static_cast<float>(signed_int4(byte)) * query_row[dim_idx];
    dot += static_cast<float>(signed_int4(byte >> 4)) * query_row[dim_idx + 1];
  }
  return dot;
}

__global__ void maxsim_int4_batched_kernel(
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

  const int byte_dim = dim / 2;
  const std::int64_t start = offsets[doc_idx];
  const std::int64_t end = offsets[doc_idx + 1];
  float doc_score = 0.0F;

  for (int q = 0; q < query_tokens; ++q) {
    float local_best = -3.402823466e+38F;
    const float* query_row = query + (static_cast<std::int64_t>(batch_idx) * query_tokens + q) * dim;
    for (std::int64_t token = start + tid; token < end; token += blockDim.x) {
      const std::uint8_t* packed_row = packed + token * byte_dim;
      const float dot = dot_int4(query_row, packed_row, dim);
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

// Quantize fp32 query rows to int8 with one symmetric scale per (batch, q)
// row in natural dim order (word w holds dims 4w..4w+3), also emitting the
// integer sum of each row for the binary 2*dot01 - qsum identity.
__global__ void quantize_query_int8_seq_dim128_kernel(
    const float* query,
    std::int32_t* query_int8_words,
    float* query_scales,
    std::int32_t* query_sums,
    int batch,
    int query_tokens) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_rows = batch * query_tokens;
  if (row >= total_rows) {
    return;
  }

  const float* query_row = query + static_cast<std::int64_t>(row) * 128;
  float max_abs = 0.0F;
  for (int dim_idx = 0; dim_idx < 128; ++dim_idx) {
    const float value = fabsf(query_row[dim_idx]);
    max_abs = value > max_abs ? value : max_abs;
  }
  const float scale = max_abs == 0.0F ? 1.0F : max_abs / 127.0F;
  const float inv_scale = 1.0F / scale;
  query_scales[row] = scale;

  std::int32_t* out = query_int8_words + static_cast<std::int64_t>(row) * 32;
  int total = 0;
  for (int word = 0; word < 32; ++word) {
    std::int32_t packed_word = 0;
    for (int lane = 0; lane < 4; ++lane) {
      const int q_value = max(-127, min(127, __float2int_rn(query_row[4 * word + lane] * inv_scale)));
      total += q_value;
      packed_word |= (q_value & 0xFF) << (8 * lane);
    }
    out[word] = packed_word;
  }
  query_sums[row] = total;
}

// Binary docs scored with an int8 query via dp4a: each packed byte expands
// through a shared 0/1-spread LUT, dot over set bits accumulates with dp4a,
// and the +/-1 dot is recovered as 2*dot01 - qsum.
__global__ void maxsim_binary_int8q_dim128_kernel(
    const std::int32_t* query_int8_words,
    const float* query_scales,
    const std::int32_t* query_sums,
    const std::uint8_t* packed,
    const std::int64_t* offsets,
    float* output,
    int batch,
    int query_tokens,
    int num_docs,
    float scale,
    const float* scale_vector,
    const float* token_scales) {
  const int doc_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float reductions[kBlockThreads];
  __shared__ std::uint32_t spread_lut[256][2];

  for (int value = tid; value < 256; value += blockDim.x) {
    const std::uint32_t low = (static_cast<std::uint32_t>(value & 0x0F) * 0x00204081U) & 0x01010101U;
    const std::uint32_t high = (static_cast<std::uint32_t>((value >> 4) & 0x0F) * 0x00204081U) & 0x01010101U;
    spread_lut[value][0] = low;
    spread_lut[value][1] = high;
  }
  __syncthreads();

  if (doc_idx >= num_docs || batch_idx >= batch) {
    return;
  }

  const std::int64_t start = offsets[doc_idx];
  const std::int64_t end = offsets[doc_idx + 1];
  float doc_score = 0.0F;

  for (int q = 0; q < query_tokens; ++q) {
    const std::int64_t row = static_cast<std::int64_t>(batch_idx) * query_tokens + q;
    const std::int32_t* q_words_global = query_int8_words + row * 32;
    std::int32_t q_words[32];
#pragma unroll
    for (int word = 0; word < 32; ++word) {
      q_words[word] = q_words_global[word];
    }
    const std::int32_t q_sum = query_sums[row];

    float local_best = -3.402823466e+38F;
    for (std::int64_t token = start + tid; token < end; token += blockDim.x) {
      const std::uint8_t* packed_row = packed + token * 16;
      int dot01 = 0;
#pragma unroll
      for (int byte_idx = 0; byte_idx < 16; ++byte_idx) {
        const std::uint8_t byte = packed_row[byte_idx];
        dot01 = __dp4a(static_cast<std::int32_t>(spread_lut[byte][0]), q_words[2 * byte_idx], dot01);
        dot01 = __dp4a(static_cast<std::int32_t>(spread_lut[byte][1]), q_words[2 * byte_idx + 1], dot01);
      }
      float dot = static_cast<float>(2 * dot01 - q_sum);
      if (token_scales != nullptr) {
        dot *= token_scales[token];
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
      doc_score += reductions[0] * query_scales[row];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float doc_scale = scale_vector == nullptr ? scale : scale_vector[doc_idx];
    output[static_cast<std::int64_t>(batch_idx) * num_docs + doc_idx] = doc_score * doc_scale;
  }
}

// Quantize fp32 query rows to int8 with one symmetric scale per (batch, q)
// row, permuted to match the int4 nibble interleave: output word w in [0,16)
// holds dims {8w, 8w+2, 8w+4, 8w+6} (low nibbles) and word 16+w holds dims
// {8w+1, 8w+3, 8w+5, 8w+7} (high nibbles).
__global__ void quantize_query_int8_dim128_kernel(
    const float* query,
    std::int32_t* query_int8_words,
    float* query_scales,
    int batch,
    int query_tokens) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_rows = batch * query_tokens;
  if (row >= total_rows) {
    return;
  }

  const float* query_row = query + static_cast<std::int64_t>(row) * 128;
  float max_abs = 0.0F;
  for (int dim_idx = 0; dim_idx < 128; ++dim_idx) {
    const float value = fabsf(query_row[dim_idx]);
    max_abs = value > max_abs ? value : max_abs;
  }
  const float scale = max_abs == 0.0F ? 1.0F : max_abs / 127.0F;
  const float inv_scale = 1.0F / scale;
  query_scales[row] = scale;

  std::int32_t* out = query_int8_words + static_cast<std::int64_t>(row) * 32;
  for (int word = 0; word < 16; ++word) {
    std::int32_t even_word = 0;
    std::int32_t odd_word = 0;
    for (int lane = 0; lane < 4; ++lane) {
      const int even_dim = 8 * word + 2 * lane;
      const int odd_dim = even_dim + 1;
      const int even_q = max(-127, min(127, __float2int_rn(query_row[even_dim] * inv_scale)));
      const int odd_q = max(-127, min(127, __float2int_rn(query_row[odd_dim] * inv_scale)));
      even_word |= (even_q & 0xFF) << (8 * lane);
      odd_word |= (odd_q & 0xFF) << (8 * lane);
    }
    out[word] = even_word;
    out[16 + word] = odd_word;
  }
}

__device__ __forceinline__ int dot_int4_int8q_dim128(const std::int32_t* q_words, const std::uint32_t* doc_words) {
  int acc = 0;
#pragma unroll
  for (int word = 0; word < 16; ++word) {
    const std::uint32_t packed_word = doc_words[word];
    const std::int32_t lo = __vsub4((packed_word & 0x0F0F0F0FU) ^ 0x08080808U, 0x08080808U);
    const std::int32_t hi = __vsub4(((packed_word >> 4) & 0x0F0F0F0FU) ^ 0x08080808U, 0x08080808U);
    acc = __dp4a(lo, q_words[word], acc);
    acc = __dp4a(hi, q_words[16 + word], acc);
  }
  return acc;
}

__global__ void maxsim_int4_int8q_dim128_kernel(
    const std::int32_t* query_int8_words,
    const float* query_scales,
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
  __shared__ int reductions_int[kBlockThreads];

  if (doc_idx >= num_docs || batch_idx >= batch) {
    return;
  }

  const std::int64_t start = offsets[doc_idx];
  const std::int64_t end = offsets[doc_idx + 1];
  float doc_score = 0.0F;

  for (int q = 0; q < query_tokens; ++q) {
    const std::int64_t row = static_cast<std::int64_t>(batch_idx) * query_tokens + q;
    const std::int32_t* q_words_global = query_int8_words + row * 32;
    std::int32_t q_words[32];
#pragma unroll
    for (int word = 0; word < 32; ++word) {
      q_words[word] = q_words_global[word];
    }

    int local_best = -2147483647 - 1;
    for (std::int64_t token = start + tid; token < end; token += blockDim.x) {
      const std::uint32_t* doc_words = reinterpret_cast<const std::uint32_t*>(packed + token * 64);
      const int dot = dot_int4_int8q_dim128(q_words, doc_words);
      local_best = dot > local_best ? dot : local_best;
    }

    reductions_int[tid] = local_best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const int other = reductions_int[tid + stride];
        reductions_int[tid] = other > reductions_int[tid] ? other : reductions_int[tid];
      }
      __syncthreads();
    }
    if (tid == 0 && start != end) {
      doc_score += static_cast<float>(reductions_int[0]) * query_scales[row];
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
        num_tokens_(static_cast<std::int64_t>(packed.shape(0))),
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
    cudaFree(d_token_scales_);
    cudaFree(d_query_);
    cudaFree(d_query_words_);
    cudaFree(d_query_scales_);
    cudaFree(d_query_sums_);
    cudaFree(d_centroid_weights_);
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

  void set_token_scale_vector(py::array_t<float, py::array::c_style | py::array::forcecast> token_scales) {
    if (token_scales.ndim() != 1 || token_scales.shape(0) != num_tokens_) {
      throw std::invalid_argument("token scale vector must have shape [num_doc_tokens]");
    }

    const std::size_t token_scale_bytes = static_cast<std::size_t>(num_tokens_) * sizeof(float);
    ensure_device_capacity(&d_token_scales_, &token_scales_capacity_, token_scale_bytes, "cudaMalloc resident token scale vector");
    check_cuda(cudaMemcpy(d_token_scales_, token_scales.data(), token_scale_bytes, cudaMemcpyHostToDevice), "copy token scale vector to device");
    has_token_scale_vector_ = true;
  }

  void clear_token_scale_vector() {
    has_token_scale_vector_ = false;
  }

  py::array_t<float> maxsim_batch(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      float scale,
      bool use_scale_vector = false,
      bool use_token_scale = false) {
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

    launch_maxsim(d_query_, d_scores_, batch, query_tokens, scale, scale_vector_ptr(use_scale_vector), token_scale_vector_ptr(use_token_scale), "resident batched maxsim");
    check_cuda(cudaDeviceSynchronize(), "synchronize resident batched maxsim kernel");

    py::array_t<float> output({batch, num_docs_});
    check_cuda(cudaMemcpy(output.mutable_data(), d_scores_, output_bytes, cudaMemcpyDeviceToHost), "copy output to host");
    return output;
  }

  py::tuple topk_batch(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      int k,
      float scale,
      bool use_scale_vector = false,
      bool use_token_scale = false) {
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

    launch_maxsim(d_query_, d_scores_, batch, query_tokens, scale, scale_vector_ptr(use_scale_vector), token_scale_vector_ptr(use_token_scale), "topk maxsim");
    topk_kernel<<<batch, kTopkThreads>>>(d_scores_, d_top_scores_, d_top_indices_, batch, num_docs_, k);
    check_cuda(cudaGetLastError(), "launch topk selection kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize topk kernels");

    py::array_t<float> output_scores({batch, k});
    py::array_t<std::int64_t> output_indices({batch, k});
    check_cuda(cudaMemcpy(output_scores.mutable_data(), d_top_scores_, top_scores_bytes, cudaMemcpyDeviceToHost), "copy top scores to host");
    check_cuda(cudaMemcpy(output_indices.mutable_data(), d_top_indices_, top_indices_bytes, cudaMemcpyDeviceToHost), "copy top indices to host");
    return py::make_tuple(output_scores, output_indices);
  }

  py::tuple topk_centroid_batch(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      py::array_t<float, py::array::c_style | py::array::forcecast> weights,
      int k,
      float scale,
      bool use_scale_vector = false,
      bool use_token_scale = false) {
    if (k < 1 || k > num_docs_) {
      throw std::invalid_argument("k must be between 1 and num_docs");
    }
    if (query.ndim() != 3) {
      throw std::invalid_argument("query must have shape [batch, query_tokens, dim]");
    }
    if (query.shape(2) != dim_) {
      throw std::invalid_argument("query shape does not match dim");
    }
    if (weights.ndim() != 1 || weights.shape(0) != dim_) {
      throw std::invalid_argument("weights must have shape [dim]");
    }

    const int batch = static_cast<int>(query.shape(0));
    const int query_tokens = static_cast<int>(query.shape(1));
    const std::size_t query_values = static_cast<std::size_t>(query.size());
    const std::size_t query_bytes = query_values * sizeof(float);
    const std::size_t weights_bytes = static_cast<std::size_t>(dim_) * sizeof(float);
    const std::size_t scores_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(num_docs_) * sizeof(float);
    const std::size_t top_scores_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) * sizeof(float);
    const std::size_t top_indices_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) * sizeof(std::int64_t);

    ensure_device_capacity(&d_query_, &query_capacity_, query_bytes, "cudaMalloc resident centroid query cache");
    ensure_device_capacity(&d_centroid_weights_, &centroid_weights_capacity_, weights_bytes, "cudaMalloc resident centroid weights");
    ensure_device_capacity(&d_scores_, &scores_capacity_, scores_bytes, "cudaMalloc resident centroid score cache");
    ensure_device_capacity(&d_top_scores_, &top_scores_capacity_, top_scores_bytes, "cudaMalloc resident centroid top score cache");
    ensure_device_capacity(&d_top_indices_, &top_indices_capacity_, top_indices_bytes, "cudaMalloc resident centroid top index cache");

    check_cuda(cudaMemcpy(d_query_, query.data(), query_bytes, cudaMemcpyHostToDevice), "copy centroid query to device");
    check_cuda(cudaMemcpy(d_centroid_weights_, weights.data(), weights_bytes, cudaMemcpyHostToDevice), "copy centroid weights to device");

    const int weight_blocks = static_cast<int>((query_values + 255U) / 256U);
    weight_query_dims_kernel<<<weight_blocks, 256>>>(d_query_, d_centroid_weights_, static_cast<std::int64_t>(query_values), dim_);
    check_cuda(cudaGetLastError(), "launch centroid query weighting kernel");

    launch_maxsim(d_query_, d_scores_, batch, query_tokens, scale, scale_vector_ptr(use_scale_vector), token_scale_vector_ptr(use_token_scale), "centroid topk maxsim");
    topk_kernel<<<batch, kTopkThreads>>>(d_scores_, d_top_scores_, d_top_indices_, batch, num_docs_, k);
    check_cuda(cudaGetLastError(), "launch centroid topk selection kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize centroid topk kernels");

    py::array_t<float> output_scores({batch, k});
    py::array_t<std::int64_t> output_indices({batch, k});
    check_cuda(cudaMemcpy(output_scores.mutable_data(), d_top_scores_, top_scores_bytes, cudaMemcpyDeviceToHost), "copy centroid top scores to host");
    check_cuda(cudaMemcpy(output_indices.mutable_data(), d_top_indices_, top_indices_bytes, cudaMemcpyDeviceToHost), "copy centroid top indices to host");
    return py::make_tuple(output_scores, output_indices);
  }

  py::tuple topk_lut_batch(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      int k,
      float scale,
      bool use_scale_vector = false,
      bool use_token_scale = false) {
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
    launch_maxsim_lut_dim128(d_query_lut_, d_scores_, batch, query_tokens, scale, scale_vector_ptr(use_scale_vector), token_scale_vector_ptr(use_token_scale), "topk LUT maxsim");
    topk_kernel<<<batch, kTopkThreads>>>(d_scores_, d_top_scores_, d_top_indices_, batch, num_docs_, k);
    check_cuda(cudaGetLastError(), "launch LUT topk selection kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize LUT topk kernels");

    py::array_t<float> output_scores({batch, k});
    py::array_t<std::int64_t> output_indices({batch, k});
    check_cuda(cudaMemcpy(output_scores.mutable_data(), d_top_scores_, top_scores_bytes, cudaMemcpyDeviceToHost), "copy LUT top scores to host");
    check_cuda(cudaMemcpy(output_indices.mutable_data(), d_top_indices_, top_indices_bytes, cudaMemcpyDeviceToHost), "copy LUT top indices to host");
    return py::make_tuple(output_scores, output_indices);
  }

  py::array_t<float> maxsim_batch_int8q(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      float scale,
      bool use_scale_vector = false,
      bool use_token_scale = false) {
    const auto shape = validate_int8q_query(query);
    const int batch = shape.first;
    const int query_tokens = shape.second;
    const std::size_t output_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(num_docs_) * sizeof(float);

    ensure_device_capacity(&d_scores_, &scores_capacity_, output_bytes, "cudaMalloc binary int8q score cache");
    launch_int8q_maxsim(query, batch, query_tokens, scale, scale_vector_ptr(use_scale_vector), token_scale_vector_ptr(use_token_scale), "binary int8q maxsim");
    check_cuda(cudaDeviceSynchronize(), "synchronize binary int8q maxsim");

    py::array_t<float> output({batch, num_docs_});
    check_cuda(cudaMemcpy(output.mutable_data(), d_scores_, output_bytes, cudaMemcpyDeviceToHost), "copy binary int8q output to host");
    return output;
  }

  py::tuple topk_batch_int8q(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      int k,
      float scale,
      bool use_scale_vector = false,
      bool use_token_scale = false) {
    if (k < 1 || k > num_docs_) {
      throw std::invalid_argument("k must be between 1 and num_docs");
    }
    const auto shape = validate_int8q_query(query);
    const int batch = shape.first;
    const int query_tokens = shape.second;
    const std::size_t scores_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(num_docs_) * sizeof(float);
    const std::size_t top_scores_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) * sizeof(float);
    const std::size_t top_indices_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) * sizeof(std::int64_t);

    ensure_device_capacity(&d_scores_, &scores_capacity_, scores_bytes, "cudaMalloc binary int8q full score cache");
    ensure_device_capacity(&d_top_scores_, &top_scores_capacity_, top_scores_bytes, "cudaMalloc binary int8q top score cache");
    ensure_device_capacity(&d_top_indices_, &top_indices_capacity_, top_indices_bytes, "cudaMalloc binary int8q top index cache");

    launch_int8q_maxsim(query, batch, query_tokens, scale, scale_vector_ptr(use_scale_vector), token_scale_vector_ptr(use_token_scale), "binary int8q topk maxsim");
    topk_kernel<<<batch, kTopkThreads>>>(d_scores_, d_top_scores_, d_top_indices_, batch, num_docs_, k);
    check_cuda(cudaGetLastError(), "launch binary int8q topk selection kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize binary int8q topk kernels");

    py::array_t<float> output_scores({batch, k});
    py::array_t<std::int64_t> output_indices({batch, k});
    check_cuda(cudaMemcpy(output_scores.mutable_data(), d_top_scores_, top_scores_bytes, cudaMemcpyDeviceToHost), "copy binary int8q top scores to host");
    check_cuda(cudaMemcpy(output_indices.mutable_data(), d_top_indices_, top_indices_bytes, cudaMemcpyDeviceToHost), "copy binary int8q top indices to host");
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
  bool has_token_scale_vector() const { return has_token_scale_vector_; }
  std::size_t token_scale_vector_size() const { return has_token_scale_vector_ ? static_cast<std::size_t>(num_tokens_) : 0; }
  const char* maxsim_kernel_variant() const {
    if (dim_ == 128 && packed_size_ >= g_dim128_qtile_min_packed_bytes) {
      return "dim128_qtile";
    }
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

  const float* token_scale_vector_ptr(bool use_token_scale) const {
    if (!use_token_scale) {
      return nullptr;
    }
    if (!has_token_scale_vector_) {
      throw std::invalid_argument("token scale vector was requested but is not loaded on this CUDA handle");
    }
    return d_token_scales_;
  }

  void launch_maxsim(float* d_query, float* d_output, int batch, int query_tokens, float scale, const float* d_scale_vector, const float* d_token_scales, const char* label) {
    dim3 grid(num_docs_, batch);
    if (dim_ == 128 && packed_size_ >= g_dim128_qtile_min_packed_bytes) {
      maxsim_batched_dim128_qtile_kernel<<<grid, kBlockThreads>>>(d_query, d_packed_, d_offsets_, d_output, batch, query_tokens, num_docs_, scale, d_scale_vector, d_token_scales);
      check_cuda(cudaGetLastError(), (std::string("launch ") + label + " dim128 qtile kernel").c_str());
      return;
    }
    if (use_dim128_unrolled()) {
      maxsim_batched_dim128_unrolled_kernel<<<grid, kBlockThreads>>>(d_query, d_packed_, d_offsets_, d_output, batch, query_tokens, num_docs_, scale, d_scale_vector, d_token_scales);
      check_cuda(cudaGetLastError(), (std::string("launch ") + label + " dim128 unrolled kernel").c_str());
      return;
    }

    maxsim_batched_kernel<<<grid, kBlockThreads>>>(d_query, d_packed_, d_offsets_, d_output, batch, query_tokens, dim_, num_docs_, scale, d_scale_vector, d_token_scales);
    check_cuda(cudaGetLastError(), (std::string("launch ") + label + " generic kernel").c_str());
  }

  void build_query_lut_dim128(float* d_query, float* d_query_lut, int batch, int query_tokens) {
    const int total = batch * query_tokens * 16 * 256;
    const int blocks = (total + 255) / 256;
    build_query_lut_dim128_kernel<<<blocks, 256>>>(d_query, d_query_lut, batch, query_tokens);
    check_cuda(cudaGetLastError(), "launch dim128 query LUT build kernel");
  }

  void launch_maxsim_lut_dim128(float* d_query_lut, float* d_output, int batch, int query_tokens, float scale, const float* d_scale_vector, const float* d_token_scales, const char* label) {
    dim3 grid(num_docs_, batch);
    maxsim_batched_dim128_lut_kernel<<<grid, kBlockThreads>>>(d_query_lut, d_packed_, d_offsets_, d_output, batch, query_tokens, num_docs_, scale, d_scale_vector, d_token_scales);
    check_cuda(cudaGetLastError(), (std::string("launch ") + label + " dim128 LUT kernel").c_str());
  }

  std::pair<int, int> validate_int8q_query(const py::array_t<float, py::array::c_style | py::array::forcecast>& query) const {
    if (dim_ != 128) {
      throw std::invalid_argument("int8 query scoring currently requires dim=128");
    }
    if (query.ndim() != 3) {
      throw std::invalid_argument("query must have shape [batch, query_tokens, dim]");
    }
    if (query.shape(2) != dim_) {
      throw std::invalid_argument("query shape does not match dim");
    }
    return {static_cast<int>(query.shape(0)), static_cast<int>(query.shape(1))};
  }

  void launch_int8q_maxsim(
      const py::array_t<float, py::array::c_style | py::array::forcecast>& query,
      int batch,
      int query_tokens,
      float scale,
      const float* d_scale_vector,
      const float* d_token_scales,
      const char* label) {
    const std::size_t query_bytes = static_cast<std::size_t>(query.size()) * sizeof(float);
    const std::size_t rows = static_cast<std::size_t>(batch) * static_cast<std::size_t>(query_tokens);
    const std::size_t words_bytes = rows * 32U * sizeof(std::int32_t);
    const std::size_t scales_bytes = rows * sizeof(float);
    const std::size_t sums_bytes = rows * sizeof(std::int32_t);

    ensure_device_capacity(&d_query_, &query_capacity_, query_bytes, "cudaMalloc binary int8q query cache");
    ensure_device_capacity(&d_query_words_, &query_words_capacity_, words_bytes, "cudaMalloc binary int8q word cache");
    ensure_device_capacity(&d_query_scales_, &query_scales_capacity_, scales_bytes, "cudaMalloc binary int8q scale cache");
    ensure_device_capacity(&d_query_sums_, &query_sums_capacity_, sums_bytes, "cudaMalloc binary int8q sum cache");
    check_cuda(cudaMemcpy(d_query_, query.data(), query_bytes, cudaMemcpyHostToDevice), "copy binary int8q query to device");

    const int quantize_blocks = static_cast<int>((rows + kBlockThreads - 1) / kBlockThreads);
    quantize_query_int8_seq_dim128_kernel<<<quantize_blocks > 0 ? quantize_blocks : 1, kBlockThreads>>>(
        d_query_, d_query_words_, d_query_scales_, d_query_sums_, batch, query_tokens);
    check_cuda(cudaGetLastError(), "launch binary int8 query quantize kernel");

    dim3 grid(num_docs_, batch);
    maxsim_binary_int8q_dim128_kernel<<<grid, kBlockThreads>>>(
        d_query_words_, d_query_scales_, d_query_sums_, d_packed_, d_offsets_, d_scores_,
        batch, query_tokens, num_docs_, scale, d_scale_vector, d_token_scales);
    check_cuda(cudaGetLastError(), (std::string("launch ") + label + " kernel").c_str());
  }

  bool use_dim128_unrolled() const {
    if (dim_ != 128) {
      return false;
    }
    if (num_docs_ <= 128) {
      return true;
    }
    const std::int64_t avg_tokens = num_tokens_ / std::max<std::int64_t>(num_docs_, 1);
    return avg_tokens >= g_dim128_unrolled_min_avg_tokens;
  }

  std::uint8_t* d_packed_ = nullptr;
  std::int64_t* d_offsets_ = nullptr;
  float* d_scale_ = nullptr;
  float* d_token_scales_ = nullptr;
  float* d_query_ = nullptr;
  std::int32_t* d_query_words_ = nullptr;
  float* d_query_scales_ = nullptr;
  std::int32_t* d_query_sums_ = nullptr;
  float* d_centroid_weights_ = nullptr;
  float* d_query_lut_ = nullptr;
  float* d_scores_ = nullptr;
  float* d_top_scores_ = nullptr;
  std::int64_t* d_top_indices_ = nullptr;
  int* d_top_locks_ = nullptr;
  int dim_ = 0;
  int num_docs_ = 0;
  std::int64_t num_tokens_ = 0;
  std::size_t packed_size_ = 0;
  std::size_t offsets_size_ = 0;
  std::size_t scale_capacity_ = 0;
  std::size_t token_scales_capacity_ = 0;
  std::size_t query_capacity_ = 0;
  std::size_t query_words_capacity_ = 0;
  std::size_t query_scales_capacity_ = 0;
  std::size_t query_sums_capacity_ = 0;
  std::size_t centroid_weights_capacity_ = 0;
  std::size_t query_lut_capacity_ = 0;
  std::size_t scores_capacity_ = 0;
  std::size_t top_scores_capacity_ = 0;
  std::size_t top_indices_capacity_ = 0;
  std::size_t top_locks_capacity_ = 0;
  bool has_scale_vector_ = false;
  bool has_token_scale_vector_ = false;
};

class CudaInt4PackedDocs {
 public:
  CudaInt4PackedDocs(
      py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> packed,
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
      int dim,
      float scale)
      : dim_(dim),
        num_docs_(static_cast<int>(offsets.shape(0) - 1)),
        scale_(scale),
        packed_size_(static_cast<std::size_t>(packed.size())),
        offsets_size_(static_cast<std::size_t>(offsets.size())) {
    if (packed.ndim() != 2) {
      throw std::invalid_argument("packed must have shape [num_doc_tokens, dim / 2]");
    }
    if (offsets.ndim() != 1) {
      throw std::invalid_argument("offsets must be a 1-D array");
    }
    if (dim <= 0 || dim % 2 != 0) {
      throw std::invalid_argument("dim must be positive and divisible by 2");
    }
    if (packed.shape(1) != dim / 2) {
      throw std::invalid_argument("packed shape does not match dim");
    }
    if (offsets.shape(0) < 2) {
      throw std::invalid_argument("offsets must contain at least [0, num_doc_tokens]");
    }
    if (scale <= 0.0F) {
      throw std::invalid_argument("scale must be > 0");
    }

    const std::size_t packed_bytes = packed_size_ * sizeof(std::uint8_t);
    const std::size_t offsets_bytes = offsets_size_ * sizeof(std::int64_t);
    check_cuda(cudaMalloc(&d_packed_, packed_bytes), "cudaMalloc int4 packed docs");
    check_cuda(cudaMalloc(&d_offsets_, offsets_bytes), "cudaMalloc int4 offsets");
    try {
      check_cuda(cudaMemcpy(d_packed_, packed.data(), packed_bytes, cudaMemcpyHostToDevice), "copy int4 packed docs to device");
      check_cuda(cudaMemcpy(d_offsets_, offsets.data(), offsets_bytes, cudaMemcpyHostToDevice), "copy int4 offsets to device");
    } catch (...) {
      cudaFree(d_packed_);
      cudaFree(d_offsets_);
      d_packed_ = nullptr;
      d_offsets_ = nullptr;
      throw;
    }
  }

  CudaInt4PackedDocs(const CudaInt4PackedDocs&) = delete;
  CudaInt4PackedDocs& operator=(const CudaInt4PackedDocs&) = delete;

  ~CudaInt4PackedDocs() {
    cudaFree(d_packed_);
    cudaFree(d_offsets_);
    cudaFree(d_query_);
    cudaFree(d_query_words_);
    cudaFree(d_query_scales_);
    cudaFree(d_scores_);
    cudaFree(d_top_scores_);
    cudaFree(d_top_indices_);
  }

  py::array_t<float> maxsim_batch(py::array_t<float, py::array::c_style | py::array::forcecast> query) {
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

    ensure_device_capacity(&d_query_, &query_capacity_, query_bytes, "cudaMalloc int4 query cache");
    ensure_device_capacity(&d_scores_, &scores_capacity_, output_bytes, "cudaMalloc int4 score cache");
    check_cuda(cudaMemcpy(d_query_, query.data(), query_bytes, cudaMemcpyHostToDevice), "copy int4 query to device");

    launch_maxsim(d_query_, d_scores_, batch, query_tokens, "int4 resident maxsim");
    check_cuda(cudaDeviceSynchronize(), "synchronize int4 resident maxsim");

    py::array_t<float> output({batch, num_docs_});
    check_cuda(cudaMemcpy(output.mutable_data(), d_scores_, output_bytes, cudaMemcpyDeviceToHost), "copy int4 output to host");
    return output;
  }

  py::tuple topk_batch(py::array_t<float, py::array::c_style | py::array::forcecast> query, int k) {
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

    ensure_device_capacity(&d_query_, &query_capacity_, query_bytes, "cudaMalloc int4 topk query cache");
    ensure_device_capacity(&d_scores_, &scores_capacity_, scores_bytes, "cudaMalloc int4 full score cache");
    ensure_device_capacity(&d_top_scores_, &top_scores_capacity_, top_scores_bytes, "cudaMalloc int4 top score cache");
    ensure_device_capacity(&d_top_indices_, &top_indices_capacity_, top_indices_bytes, "cudaMalloc int4 top index cache");
    check_cuda(cudaMemcpy(d_query_, query.data(), query_bytes, cudaMemcpyHostToDevice), "copy int4 topk query to device");

    launch_maxsim(d_query_, d_scores_, batch, query_tokens, "int4 topk maxsim");
    topk_kernel<<<batch, kTopkThreads>>>(d_scores_, d_top_scores_, d_top_indices_, batch, num_docs_, k);
    check_cuda(cudaGetLastError(), "launch int4 topk selection kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize int4 topk kernels");

    py::array_t<float> output_scores({batch, k});
    py::array_t<std::int64_t> output_indices({batch, k});
    check_cuda(cudaMemcpy(output_scores.mutable_data(), d_top_scores_, top_scores_bytes, cudaMemcpyDeviceToHost), "copy int4 top scores to host");
    check_cuda(cudaMemcpy(output_indices.mutable_data(), d_top_indices_, top_indices_bytes, cudaMemcpyDeviceToHost), "copy int4 top indices to host");
    return py::make_tuple(output_scores, output_indices);
  }

  py::array_t<float> maxsim_batch_int8q(py::array_t<float, py::array::c_style | py::array::forcecast> query) {
    const auto shape = validate_int8q_query(query);
    const int batch = shape.first;
    const int query_tokens = shape.second;
    const std::size_t output_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(num_docs_) * sizeof(float);

    ensure_device_capacity(&d_scores_, &scores_capacity_, output_bytes, "cudaMalloc int8q score cache");
    launch_int8q_maxsim(query, batch, query_tokens, "int4 int8q maxsim");
    check_cuda(cudaDeviceSynchronize(), "synchronize int4 int8q maxsim");

    py::array_t<float> output({batch, num_docs_});
    check_cuda(cudaMemcpy(output.mutable_data(), d_scores_, output_bytes, cudaMemcpyDeviceToHost), "copy int8q output to host");
    return output;
  }

  py::tuple topk_batch_int8q(py::array_t<float, py::array::c_style | py::array::forcecast> query, int k) {
    if (k < 1 || k > num_docs_) {
      throw std::invalid_argument("k must be between 1 and num_docs");
    }
    const auto shape = validate_int8q_query(query);
    const int batch = shape.first;
    const int query_tokens = shape.second;
    const std::size_t scores_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(num_docs_) * sizeof(float);
    const std::size_t top_scores_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) * sizeof(float);
    const std::size_t top_indices_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(k) * sizeof(std::int64_t);

    ensure_device_capacity(&d_scores_, &scores_capacity_, scores_bytes, "cudaMalloc int8q full score cache");
    ensure_device_capacity(&d_top_scores_, &top_scores_capacity_, top_scores_bytes, "cudaMalloc int8q top score cache");
    ensure_device_capacity(&d_top_indices_, &top_indices_capacity_, top_indices_bytes, "cudaMalloc int8q top index cache");

    launch_int8q_maxsim(query, batch, query_tokens, "int4 int8q topk maxsim");
    topk_kernel<<<batch, kTopkThreads>>>(d_scores_, d_top_scores_, d_top_indices_, batch, num_docs_, k);
    check_cuda(cudaGetLastError(), "launch int8q topk selection kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize int8q topk kernels");

    py::array_t<float> output_scores({batch, k});
    py::array_t<std::int64_t> output_indices({batch, k});
    check_cuda(cudaMemcpy(output_scores.mutable_data(), d_top_scores_, top_scores_bytes, cudaMemcpyDeviceToHost), "copy int8q top scores to host");
    check_cuda(cudaMemcpy(output_indices.mutable_data(), d_top_indices_, top_indices_bytes, cudaMemcpyDeviceToHost), "copy int8q top indices to host");
    return py::make_tuple(output_scores, output_indices);
  }

  int dim() const { return dim_; }
  int num_docs() const { return num_docs_; }
  float scale() const { return scale_; }
  std::size_t packed_size() const { return packed_size_; }

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

  void launch_maxsim(float* d_query, float* d_output, int batch, int query_tokens, const char* label) {
    dim3 grid(num_docs_, batch);
    maxsim_int4_batched_kernel<<<grid, kBlockThreads>>>(d_query, d_packed_, d_offsets_, d_output, batch, query_tokens, dim_, num_docs_, scale_);
    check_cuda(cudaGetLastError(), (std::string("launch ") + label + " kernel").c_str());
  }

  std::pair<int, int> validate_int8q_query(const py::array_t<float, py::array::c_style | py::array::forcecast>& query) const {
    if (dim_ != 128) {
      throw std::invalid_argument("int8 query scoring currently requires dim=128");
    }
    if (query.ndim() != 3) {
      throw std::invalid_argument("query must have shape [batch, query_tokens, dim]");
    }
    if (query.shape(2) != dim_) {
      throw std::invalid_argument("query shape does not match dim");
    }
    return {static_cast<int>(query.shape(0)), static_cast<int>(query.shape(1))};
  }

  void launch_int8q_maxsim(const py::array_t<float, py::array::c_style | py::array::forcecast>& query, int batch, int query_tokens, const char* label) {
    const std::size_t query_bytes = static_cast<std::size_t>(query.size()) * sizeof(float);
    const std::size_t rows = static_cast<std::size_t>(batch) * static_cast<std::size_t>(query_tokens);
    const std::size_t words_bytes = rows * 32U * sizeof(std::int32_t);
    const std::size_t scales_bytes = rows * sizeof(float);

    ensure_device_capacity(&d_query_, &query_capacity_, query_bytes, "cudaMalloc int8q query cache");
    ensure_device_capacity(&d_query_words_, &query_words_capacity_, words_bytes, "cudaMalloc int8q word cache");
    ensure_device_capacity(&d_query_scales_, &query_scales_capacity_, scales_bytes, "cudaMalloc int8q scale cache");
    check_cuda(cudaMemcpy(d_query_, query.data(), query_bytes, cudaMemcpyHostToDevice), "copy int8q query to device");

    const int quantize_blocks = static_cast<int>((rows + kBlockThreads - 1) / kBlockThreads);
    quantize_query_int8_dim128_kernel<<<quantize_blocks > 0 ? quantize_blocks : 1, kBlockThreads>>>(
        d_query_, d_query_words_, d_query_scales_, batch, query_tokens);
    check_cuda(cudaGetLastError(), "launch int8 query quantize kernel");

    dim3 grid(num_docs_, batch);
    maxsim_int4_int8q_dim128_kernel<<<grid, kBlockThreads>>>(
        d_query_words_, d_query_scales_, d_packed_, d_offsets_, d_scores_, batch, query_tokens, num_docs_, scale_);
    check_cuda(cudaGetLastError(), (std::string("launch ") + label + " kernel").c_str());
  }

  std::uint8_t* d_packed_ = nullptr;
  std::int64_t* d_offsets_ = nullptr;
  float* d_query_ = nullptr;
  std::int32_t* d_query_words_ = nullptr;
  float* d_query_scales_ = nullptr;
  float* d_scores_ = nullptr;
  float* d_top_scores_ = nullptr;
  std::int64_t* d_top_indices_ = nullptr;
  int dim_ = 0;
  int num_docs_ = 0;
  float scale_ = 1.0F;
  std::size_t packed_size_ = 0;
  std::size_t offsets_size_ = 0;
  std::size_t query_capacity_ = 0;
  std::size_t query_words_capacity_ = 0;
  std::size_t query_scales_capacity_ = 0;
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
    maxsim_batched_kernel<<<grid, kBlockThreads>>>(d_query, d_packed, d_offsets, d_output, batch, query_tokens, dim, num_docs, scale, nullptr, nullptr);
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
    maxsim_batched_kernel<<<grid, kBlockThreads>>>(d_query, d_packed, d_offsets, d_output, batch, query_tokens, dim, num_docs, scale, nullptr, nullptr);
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
      .def("set_token_scale_vector", &CudaPackedDocs::set_token_scale_vector, py::arg("token_scales"))
      .def("clear_token_scale_vector", &CudaPackedDocs::clear_token_scale_vector)
      .def("maxsim_batch", &CudaPackedDocs::maxsim_batch, py::arg("query"), py::arg("scale") = 1.0F, py::arg("use_scale_vector") = false, py::arg("use_token_scale") = false)
      .def("topk_batch", &CudaPackedDocs::topk_batch, py::arg("query"), py::arg("k"), py::arg("scale") = 1.0F, py::arg("use_scale_vector") = false, py::arg("use_token_scale") = false)
      .def("topk_centroid_batch", &CudaPackedDocs::topk_centroid_batch, py::arg("query"), py::arg("weights"), py::arg("k"), py::arg("scale") = 1.0F, py::arg("use_scale_vector") = false, py::arg("use_token_scale") = false)
      .def("topk_lut_batch", &CudaPackedDocs::topk_lut_batch, py::arg("query"), py::arg("k"), py::arg("scale") = 1.0F, py::arg("use_scale_vector") = false, py::arg("use_token_scale") = false)
      .def("maxsim_batch_int8q", &CudaPackedDocs::maxsim_batch_int8q, py::arg("query"), py::arg("scale") = 1.0F, py::arg("use_scale_vector") = false, py::arg("use_token_scale") = false)
      .def("topk_batch_int8q", &CudaPackedDocs::topk_batch_int8q, py::arg("query"), py::arg("k"), py::arg("scale") = 1.0F, py::arg("use_scale_vector") = false, py::arg("use_token_scale") = false)
      .def("streaming_topk_batch", &CudaPackedDocs::streaming_topk_batch, py::arg("query"), py::arg("k"), py::arg("scale") = 1.0F, py::arg("use_scale_vector") = false)
      .def_property_readonly("dim", &CudaPackedDocs::dim)
      .def_property_readonly("num_docs", &CudaPackedDocs::num_docs)
      .def_property_readonly("packed_size", &CudaPackedDocs::packed_size)
      .def_property_readonly("has_scale_vector", &CudaPackedDocs::has_scale_vector)
      .def_property_readonly("scale_vector_size", &CudaPackedDocs::scale_vector_size)
      .def_property_readonly("has_token_scale_vector", &CudaPackedDocs::has_token_scale_vector)
      .def_property_readonly("token_scale_vector_size", &CudaPackedDocs::token_scale_vector_size)
      .def_property_readonly("maxsim_kernel_variant", &CudaPackedDocs::maxsim_kernel_variant);
  py::class_<CudaInt4PackedDocs>(m, "CudaInt4PackedDocs")
      .def(py::init<
	           py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>,
	           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>,
	           int,
	           float>())
      .def("maxsim_batch", &CudaInt4PackedDocs::maxsim_batch, py::arg("query"))
      .def("topk_batch", &CudaInt4PackedDocs::topk_batch, py::arg("query"), py::arg("k"))
      .def("maxsim_batch_int8q", &CudaInt4PackedDocs::maxsim_batch_int8q, py::arg("query"))
      .def("topk_batch_int8q", &CudaInt4PackedDocs::topk_batch_int8q, py::arg("query"), py::arg("k"))
      .def_property_readonly("dim", &CudaInt4PackedDocs::dim)
      .def_property_readonly("num_docs", &CudaInt4PackedDocs::num_docs)
      .def_property_readonly("scale", &CudaInt4PackedDocs::scale)
      .def_property_readonly("packed_size", &CudaInt4PackedDocs::packed_size);
  m.def("maxsim_cuda", &maxsim_cuda, py::arg("query"), py::arg("packed"), py::arg("offsets"), py::arg("dim"), py::arg("scale") = 1.0F);
  m.def("maxsim_cuda_batch", &maxsim_cuda_batch, py::arg("query"), py::arg("packed"), py::arg("offsets"), py::arg("dim"), py::arg("scale") = 1.0F);
  m.def("set_dim128_unrolled_min_avg_tokens", [](int value) { g_dim128_unrolled_min_avg_tokens = value; }, py::arg("value"));
  m.def("get_dim128_unrolled_min_avg_tokens", []() { return g_dim128_unrolled_min_avg_tokens; });
  m.def("set_dim128_qtile_min_packed_bytes", [](std::size_t value) { g_dim128_qtile_min_packed_bytes = value; }, py::arg("value"));
  m.def("get_dim128_qtile_min_packed_bytes", []() { return g_dim128_qtile_min_packed_bytes; });
}
