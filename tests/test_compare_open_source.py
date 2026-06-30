import json

import numpy as np

from benchmarks.compare_open_source import main


def _write_fixture(path):
    doc_embeddings = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1],
        ],
        dtype=np.float32,
    )
    query_embeddings = np.array(
        [
            [[1, 1, 1, 1, 1, 1, 1, 1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1]],
        ],
        dtype=np.float32,
    )
    np.savez(
        path,
        doc_embeddings=doc_embeddings,
        doc_offsets=np.array([0, 1, 2, 3], dtype=np.int64),
        query_embeddings=query_embeddings,
        qrels=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        doc_ids=np.array(["positive", "negative", "mixed"]),
        dataset_name=np.array("oss-fixture"),
    )


def test_compare_open_source_emits_dense_and_bitmax_rows(tmp_path, capsys):
    input_path = tmp_path / "fixture.npz"
    output_path = tmp_path / "oss.json"
    _write_fixture(input_path)

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--implementations",
            "dense_fp16,bitmax_binary,bitmax_int4",
        ]
    )

    captured = capsys.readouterr()
    assert "dense_fp16_baseline" in captured.out
    assert "bitmax_binary" in captured.out
    data = json.loads(output_path.read_text())
    rows = {row["implementation"]: row for row in data["results"]}
    assert data["benchmark"] == "open_source_comparison"
    assert rows["dense_fp16_baseline"]["ndcg_at_10"] == 1.0
    assert rows["bitmax_binary"]["doc_memory_compression_vs_fp32"] == 32.0
    assert rows["bitmax_int4"]["implementation_kind"] == "bitmax_sdk"
