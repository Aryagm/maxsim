#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using Lut = std::array<float, 256>;

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
  if (query.shape(1) != dim) {
    throw std::invalid_argument("query dim does not match dim");
  }
  if (packed.shape(1) != dim / 8) {
    throw std::invalid_argument("packed byte width does not match dim");
  }
  if (offsets.shape(0) < 2) {
    throw std::invalid_argument("offsets must contain at least two entries");
  }

  const auto* off = offsets.data();
  if (off[0] != 0) {
    throw std::invalid_argument("offsets must start at 0");
  }
  if (off[offsets.shape(0) - 1] != packed.shape(0)) {
    throw std::invalid_argument("offsets must end at num_doc_tokens");
  }
  for (py::ssize_t i = 1; i < offsets.shape(0); ++i) {
    if (off[i] < off[i - 1]) {
      throw std::invalid_argument("offsets must be monotonically nondecreasing");
    }
  }
}

std::vector<Lut> build_query_luts(const float* query, py::ssize_t query_tokens, int dim) {
  const int byte_dim = dim / 8;
  std::vector<Lut> luts(static_cast<std::size_t>(query_tokens * byte_dim));

  for (py::ssize_t q = 0; q < query_tokens; ++q) {
    for (int byte_idx = 0; byte_idx < byte_dim; ++byte_idx) {
      Lut& lut = luts[static_cast<std::size_t>(q * byte_dim + byte_idx)];
      for (int byte_value = 0; byte_value < 256; ++byte_value) {
        float sum = 0.0F;
        for (int bit = 0; bit < 8; ++bit) {
          const int dim_idx = byte_idx * 8 + bit;
          const bool positive = ((byte_value >> bit) & 1) != 0;
          const float sign = positive ? 1.0F : -1.0F;
          sum += sign * query[q * dim + dim_idx];
        }
        lut[static_cast<std::size_t>(byte_value)] = sum;
      }
    }
  }

  return luts;
}

py::array_t<float> maxsim_lut(
    py::array_t<float, py::array::c_style | py::array::forcecast> query,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> packed,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
    int dim,
    float scale) {
  validate_inputs(query, packed, offsets, dim);

  const py::ssize_t query_tokens = query.shape(0);
  const int byte_dim = dim / 8;
  const py::ssize_t num_docs = offsets.shape(0) - 1;
  const auto* q_ptr = query.data();
  const auto* packed_ptr = packed.data();
  const auto* off = offsets.data();
  auto luts = build_query_luts(q_ptr, query_tokens, dim);

  py::array_t<float> output({num_docs});
  auto* out = output.mutable_data();

  for (py::ssize_t doc_idx = 0; doc_idx < num_docs; ++doc_idx) {
    float doc_score = 0.0F;
    const std::int64_t start = off[doc_idx];
    const std::int64_t end = off[doc_idx + 1];

    for (py::ssize_t q = 0; q < query_tokens; ++q) {
      float best = -std::numeric_limits<float>::infinity();
      for (std::int64_t token = start; token < end; ++token) {
        const auto* token_bytes = packed_ptr + token * byte_dim;
        float dot = 0.0F;
        for (int byte_idx = 0; byte_idx < byte_dim; ++byte_idx) {
          const Lut& lut = luts[static_cast<std::size_t>(q * byte_dim + byte_idx)];
          dot += lut[token_bytes[byte_idx]];
        }
        best = std::max(best, dot);
      }
      if (start != end) {
        doc_score += best;
      }
    }

    out[doc_idx] = doc_score * scale;
  }

  return output;
}

py::dict cpu_features() {
  py::dict features;
#if defined(BITMAX_ARM64)
  features["arch"] = "arm64";
  features["neon_compilable"] = true;
#else
  features["arch"] = "generic";
  features["neon_compilable"] = false;
#endif
#if defined(__AVX2__)
  features["avx2_compilable"] = true;
#else
  features["avx2_compilable"] = false;
#endif
  features["kernel"] = "scalar_byte_lut";
  return features;
}

}  // namespace

PYBIND11_MODULE(_bitmax_cpp, m) {
  m.doc() = "Native CPU kernels for bitmax";
  m.def("maxsim_lut", &maxsim_lut, py::arg("query"), py::arg("packed"), py::arg("offsets"), py::arg("dim"), py::arg("scale") = 1.0F);
  m.def("cpu_features", &cpu_features);
}

