import numpy as np
import pytest
from importlib import import_module

import bitmax


@pytest.mark.cuda
def test_cuda_maxsim_matches_cpu_reference_for_small_fixture():
    pytest.importorskip("bitmax._bitmax_cuda")
    docs = np.array(
        [
            [1, -2, 3, -4, 5, -6, 7, -8],
            [-1, 2, -3, 4, -5, 6, -7, 8],
        ],
        dtype=np.float32,
    )
    query = np.array([[3, -1, 2, -4, 5, -6, 7, -8]], dtype=np.float32)
    packed = bitmax.pack_signs(docs)

    cpu_scores = bitmax.maxsim(query, packed, device="cpu")
    cuda_scores = bitmax.maxsim(query, packed, device="cuda")

    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=0, atol=1e-5)


def test_cuda_device_request_fails_clearly_without_cuda_build():
    try:
        import_module("bitmax._bitmax_cuda")
    except ImportError:
        docs = np.ones((1, 8), dtype=np.float32)
        query = np.ones((1, 8), dtype=np.float32)
        packed = bitmax.pack_signs(docs)

        with pytest.raises(NotImplementedError, match="CUDA maxsim is not available"):
            bitmax.maxsim(query, packed, device="cuda")
