import numpy as np

import bitmax
import bitmax._api as api
from bitmax import _bitmax_cpp


def test_cpp_lut_maxsim_matches_python_reference_for_ragged_docs():
    docs = np.array(
        [
            [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16],
            [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16],
            [1, 2, 3, 4, -5, -6, -7, -8, 9, 10, 11, 12, -13, -14, -15, -16],
            [-1, -2, -3, -4, 5, 6, 7, 8, -9, -10, -11, -12, 13, 14, 15, 16],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 2, 4], dtype=np.int64)
    query = np.array(
        [
            [3, -1, 2, -4, 5, -6, 7, -8, 1, -3, 5, -7, 2, -4, 6, -8],
            [-2, 4, -6, 8, -1, 3, -5, 7, -8, 6, -4, 2, 7, -5, 3, -1],
        ],
        dtype=np.float32,
    )
    packed = bitmax.pack_signs(docs, offsets)

    cpp_scores = _bitmax_cpp.maxsim_lut(query, packed.data, packed.doc_offsets, packed.dim, 1.0)
    py_scores = bitmax.maxsim(query, packed)

    np.testing.assert_allclose(cpp_scores, py_scores, rtol=0, atol=1e-5)


def test_public_maxsim_uses_cpp_kernel_for_cpu_queries(monkeypatch):
    docs = np.ones((1, 8), dtype=np.float32)
    query = np.ones((1, 8), dtype=np.float32)
    packed = bitmax.pack_signs(docs)
    calls = []

    def fake_maxsim_lut(query_arg, data_arg, offsets_arg, dim_arg, scale_arg):
        calls.append((query_arg.shape, data_arg.shape, offsets_arg.tolist(), dim_arg, scale_arg))
        return np.array([123.0], dtype=np.float32)

    monkeypatch.setattr(api._bitmax_cpp, "maxsim_lut", fake_maxsim_lut)

    scores = bitmax.maxsim(query, packed)

    assert scores.tolist() == [123.0]
    assert calls == [((1, 8), (1, 1), [0, 1], 8, 1.0)]
