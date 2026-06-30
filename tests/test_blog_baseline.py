import json

import numpy as np


def test_blog_binary_benchmark_emits_storage_speed_and_correctness(tmp_path):
    from benchmarks.blog_baseline import run_benchmark

    output = tmp_path / "blog-smoke.json"
    result = run_benchmark(stage="smoke", output_path=output, repeat=1)

    assert output.exists()
    assert json.loads(output.read_text()) == result
    assert result["benchmark"] == "blog_baseline"
    assert result["stage"] == "smoke"

    rows = {row["implementation"]: row for row in result["results"]}
    assert set(rows) == {"fp32_query_fp32_docs", "int8_query_int8_docs", "int8_query_binary_docs", "binary_query_binary_docs"}

    assert rows["fp32_query_fp32_docs"]["doc_storage_bytes_per_doc"] == 4 * 16 * 4
    assert rows["int8_query_int8_docs"]["doc_storage_bytes_per_doc"] == 4 * 16
    assert rows["int8_query_binary_docs"]["doc_storage_bytes_per_doc"] == 4 * 16 // 8
    assert rows["binary_query_binary_docs"]["doc_storage_bytes_per_doc"] == 4 * 16 // 8

    assert rows["fp32_query_fp32_docs"]["max_abs_delta_vs_reference"] == 0.0
    assert rows["int8_query_binary_docs"]["formula"] == "sum(max(q_int8 @ sign(doc).T))"
    assert np.isfinite(rows["int8_query_binary_docs"]["latency_ms"])
    assert np.isfinite(rows["int8_query_binary_docs"]["speedup_vs_fp32"])


def test_blog_stage_matches_blog_shape_metadata(tmp_path):
    from benchmarks.blog_baseline import run_benchmark

    result = run_benchmark(stage="blog-shape", output_path=tmp_path / "blog-shape.json", repeat=1)

    assert result["specs"] == [
        {
            "name": "blog_33q_1000d_786t_128dim",
            "query_tokens": 33,
            "docs": 1000,
            "doc_tokens": 786,
            "dim": 128,
        }
    ]
    rows = {row["implementation"]: row for row in result["results"]}
    assert rows["fp32_query_fp32_docs"]["doc_storage_bytes_per_doc"] == 786 * 128 * 4
    assert rows["int8_query_int8_docs"]["doc_storage_bytes_per_doc"] == 786 * 128
    assert rows["int8_query_binary_docs"]["doc_storage_bytes_per_doc"] == 786 * 128 // 8
