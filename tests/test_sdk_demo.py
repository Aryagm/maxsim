import json

import numpy as np

from examples.local_multivector_search import main


def _write_demo_fixture(path):
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
        query_ids=np.array(["q0", "q1"]),
        dataset_name=np.array("demo-fixture"),
    )


def test_local_multivector_demo_emits_comparison_json(tmp_path, capsys):
    input_path = tmp_path / "fixture.npz"
    output_path = tmp_path / "demo.json"
    _write_demo_fixture(input_path)

    main(
        [
            "--input",
            str(input_path),
            "--device",
            "cpu",
            "--modes",
            "binary,binary_q40,int4",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert "dense_fp16_baseline" in captured.out
    assert "bitmax_binary" in captured.out
    data = json.loads(output_path.read_text())
    rows = {row["implementation"]: row for row in data["results"]}
    assert data["benchmark"] == "sdk_local_multivector_search"
    assert rows["dense_fp16_baseline"]["ndcg_at_10"] == 1.0
    assert rows["bitmax_binary"]["doc_memory_compression_vs_fp32"] == 32.0
    assert rows["bitmax_binary_q40"]["doc_storage_bytes"] > rows["bitmax_binary"]["doc_storage_bytes"]
    assert rows["bitmax_int4"]["doc_memory_compression_vs_fp32"] == 6.0
