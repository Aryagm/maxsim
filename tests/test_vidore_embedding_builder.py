from types import SimpleNamespace

import numpy as np

from benchmarks.build_vidore_embeddings import _deduplicate_doc_rows, _flatten_ragged, _ignore_missing_torchvision_nms_fake_registration


def test_deduplicate_doc_rows_maps_multiple_queries_to_one_document():
    rows = [
        {"questionId": "q0", "query": "first question", "docId": 7, "image_filename": "same", "page": "1"},
        {"questionId": "q1", "query": "second question", "docId": 7, "image_filename": "same", "page": "1"},
        {"questionId": "q2", "query": "third question", "docId": 8, "image_filename": "other", "page": "2"},
    ]

    doc_keys, doc_row_indices, query_doc_indices, qrels = _deduplicate_doc_rows(rows)

    assert doc_keys == ("7:same:1", "8:other:2")
    assert doc_row_indices == (0, 2)
    assert query_doc_indices == (0, 0, 1)
    assert qrels.tolist() == [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]


def test_flatten_ragged_embeddings_records_offsets():
    arrays = [
        np.ones((2, 8), dtype=np.float32),
        np.zeros((3, 8), dtype=np.float32),
    ]

    flat, offsets = _flatten_ragged(arrays)

    assert flat.shape == (5, 8)
    assert offsets.tolist() == [0, 2, 5]
    assert flat.dtype == np.float32


def test_torchvision_nms_fake_registration_patch_ignores_only_missing_nms():
    class FakeLibrary:
        def register_fake(self, op_name, *args, **kwargs):
            def decorator(fn):
                raise RuntimeError(f"operator {op_name} does not exist")

            return decorator

    fake_torch = SimpleNamespace(library=FakeLibrary())

    _ignore_missing_torchvision_nms_fake_registration(fake_torch)

    def fn():
        return None

    assert fake_torch.library.register_fake("torchvision::nms")(fn) is fn
