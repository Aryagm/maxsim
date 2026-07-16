#!/usr/bin/env bash
# Full second-encoder matrix for one RTX 4090. Cases are checkpointed so the
# script can be rerun after an SSH disconnect without rebuilding finished data.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/bitmax}"
RUN="${RUN:-/workspace/paper-models-20260715}"
TEXT_PY="${TEXT_PY:-/workspace/text-env/bin/python}"
VISION_PY="${VISION_PY:-/workspace/vision-env/bin/python}"
MAXSIM_PY="${MAXSIM_PY:-/opt/conda/bin/python}"
SOURCE_COMMIT="${SOURCE_COMMIT:-unknown}"

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
mkdir -p "$RUN"/{caches,results,logs,done}
exec > >(tee -a "$RUN/logs/models.log") 2>&1

FAILURES=0
STATUS_FILE="$RUN/case-status.tsv"
touch "$STATUS_FILE"

run_case() {
  local name="$1"
  shift
  local marker="$RUN/done/$name"
  if [[ -f "$marker" ]]; then
    printf '%s\t%s\tskipped_complete\n' "$(date -u +%FT%TZ)" "$name" | tee -a "$STATUS_FILE"
    return 0
  fi

  printf '%s\t%s\tstarted\n' "$(date -u +%FT%TZ)" "$name" | tee -a "$STATUS_FILE"
  if (set -Eeuo pipefail; "$@") 2>&1 | tee -a "$RUN/logs/$name.log"; then
    touch "$marker"
    printf '%s\t%s\tcomplete\n' "$(date -u +%FT%TZ)" "$name" | tee -a "$STATUS_FILE"
  else
    FAILURES=$((FAILURES + 1))
    printf '%s\t%s\tfailed\n' "$(date -u +%FT%TZ)" "$name" | tee -a "$STATUS_FILE"
  fi
}

validate_text_cache() {
  local cache="$1" dataset="$2" expected_docs="$3" expected_queries="$4"
  "$TEXT_PY" - "$cache" "$dataset" "$expected_docs" "$expected_queries" <<'PY'
import sys
import numpy as np

path, dataset = sys.argv[1], sys.argv[2]
expected_docs, expected_queries = int(sys.argv[3]), int(sys.argv[4])
with np.load(path, allow_pickle=False) as data:
    assert str(data["model_requested"].item()) == "jinaai/jina-colbert-v2"
    assert str(data["model_resolved"].item()) == "jinaai/jina-colbert-v2"
    assert not bool(data["model_fallback_used"].item())
    assert str(data["dataset_id"].item()) == f"BeIR/{dataset}"
    assert len(data["doc_offsets"]) - 1 == expected_docs
    assert len(data["query_offsets"]) - 1 == expected_queries
    qrels = np.asarray(data["qrels"])
    assert qrels.shape == (expected_queries, expected_docs)
    assert np.all(np.any(qrels > 0, axis=1))
PY
}

build_text() {
  local dataset="$1" expected_docs="$2" expected_queries="$3"
  local cache="$RUN/caches/beir-${dataset}-jina-colbert-v2-full.npz"
  local partial="$RUN/caches/.beir-${dataset}-jina-colbert-v2-full.partial.npz"
  if [[ ! -s "$cache" ]]; then
    rm -f "$partial"
    "$TEXT_PY" -m benchmarks.build_beir_colbert_embeddings \
      --dataset "$dataset" \
      --model jinaai/jina-colbert-v2 \
      --no-model-fallback \
      --device cuda \
      --batch-size 1 \
      --output "$partial"
    validate_text_cache "$partial" "$dataset" "$expected_docs" "$expected_queries"
    mv "$partial" "$cache"
  fi
  validate_text_cache "$cache" "$dataset" "$expected_docs" "$expected_queries"
  sha256sum "$cache" > "$cache.sha256"
  "$MAXSIM_PY" -m benchmarks.run_quality_matrix "$cache" \
    --output "$RUN/results/quality-beir-${dataset}-jina-colbert-v2-full.json" \
    --formats dense_fp16,binary,binary_token_scale_u4,pool3_binary,int4_per_tensor,int4_per_token,int8_per_token \
    --device cuda --checkpoint-every 8 --max-chunk-tokens 262144
}

validate_vision_cache() {
  local cache="$1" model="$2" dataset="$3" expected_docs="$4" expected_queries="$5"
  "$VISION_PY" - "$cache" "$model" "$dataset" "$expected_docs" "$expected_queries" <<'PY'
import sys
import numpy as np

path, model, dataset = sys.argv[1], sys.argv[2], sys.argv[3]
expected_docs, expected_queries = int(sys.argv[4]), int(sys.argv[5])
with np.load(path, allow_pickle=False) as data:
    assert str(data["model_name"].item()) == model
    assert str(data["source_dataset"].item()) == dataset
    assert str(data["source_split"].item()) == "test"
    assert len(data["doc_offsets"]) - 1 == expected_docs
    assert len(data["query_offsets"]) - 1 == expected_queries
    qrels = np.asarray(data["qrels"])
    assert qrels.shape == (expected_queries, expected_docs)
    assert np.count_nonzero(qrels) == expected_queries
PY
}

build_colpali() {
  local slug="$1" dataset="$2" expected_docs="$3" expected_queries="$4"
  local model="vidore/colpali-v1.3-hf"
  local cache="$RUN/caches/vidore-${slug}-colpali-v1.3-full.npz"
  local partial="$RUN/caches/.vidore-${slug}-colpali-v1.3-full.partial.npz"
  if [[ ! -s "$cache" ]]; then
    rm -f "$partial"
    "$VISION_PY" -m benchmarks.build_vidore_embeddings \
      --dataset "$dataset" --split test --limit 500 \
      --model "$model" --batch-size 1 --output "$partial"
    validate_vision_cache "$partial" "$model" "$dataset" "$expected_docs" "$expected_queries"
    mv "$partial" "$cache"
  fi
  validate_vision_cache "$cache" "$model" "$dataset" "$expected_docs" "$expected_queries"
  sha256sum "$cache" > "$cache.sha256"
  "$MAXSIM_PY" -m benchmarks.run_quality_matrix "$cache" \
    --output "$RUN/results/quality-colpali-v1.3-${slug}-full.json" \
    --formats dense_fp16,binary,binary_token_scale_u4,pool3_binary,int4_per_tensor,int4_per_token,int8_per_token \
    --device cuda --checkpoint-every 8 --max-chunk-tokens 262144
}

build_colqwen_tabfquad() {
  local dataset="vidore/tabfquad_test_subsampled"
  local model="vidore/colqwen2-v1.0-hf"
  local cache="$RUN/caches/vidore-tabfquad-colqwen2-full.npz"
  local partial="$RUN/caches/.vidore-tabfquad-colqwen2-full.partial.npz"
  if [[ ! -s "$cache" ]]; then
    rm -f "$partial"
    "$VISION_PY" -m benchmarks.build_vidore_embeddings \
      --dataset "$dataset" --split test --limit 500 \
      --model "$model" --batch-size 1 --output "$partial"
    validate_vision_cache "$partial" "$model" "$dataset" 70 280
    mv "$partial" "$cache"
  fi
  validate_vision_cache "$cache" "$model" "$dataset" 70 280
  sha256sum "$cache" > "$cache.sha256"
  "$MAXSIM_PY" -m benchmarks.run_quality_matrix "$cache" \
    --output "$RUN/results/quality-vidore-tabfquad-colqwen2-new-int4.json" \
    --formats int4_per_token,int4_per_tensor,int8_per_token \
    --device cuda --checkpoint-every 8 --max-chunk-tokens 262144
}

"$MAXSIM_PY" - "$RUN/environment.json" "$SOURCE_COMMIT" "$TEXT_PY" "$VISION_PY" <<'PY'
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

output, source_commit, text_python, vision_python = sys.argv[1:]
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_commit": source_commit,
    "platform": platform.platform(),
    "nvidia_smi": subprocess.check_output(["nvidia-smi", "-L"], text=True).strip(),
    "qrels_policy": "binary_positive_for_historical_gte_comparability",
    "interpreters": {},
}
probe = "import json,torch,transformers; print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}))"
for name, executable in (("text", text_python), ("vision", vision_python)):
    payload["interpreters"][name] = json.loads(subprocess.check_output([executable, "-c", probe], text=True))
with open(output, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

run_case cuda_tests "$MAXSIM_PY" -m pytest -m cuda -q
run_case jina_scifact build_text scifact 5183 300
run_case colpali_tabfquad build_colpali tabfquad vidore/tabfquad_test_subsampled 70 280
run_case jina_nfcorpus build_text nfcorpus 3633 323
run_case colpali_docvqa build_colpali docvqa vidore/docvqa_test_subsampled 500 500
run_case colpali_infovqa build_colpali infovqa vidore/infovqa_test_subsampled 500 500
run_case colpali_arxivqa build_colpali arxivqa vidore/arxivqa_test_subsampled 500 500
run_case colqwen2_tabfquad build_colqwen_tabfquad
run_case jina_fiqa build_text fiqa 57638 648

find "$HF_HOME/hub" -path '*/refs/main' -type f -print -exec sed -n '1p' {} \; \
  > "$RUN/hf-revisions.txt" 2>/dev/null || true
find "$RUN/results" -type f -name '*.json' -print0 | sort -z | xargs -0 sha256sum \
  > "$RUN/result-checksums.sha256"

printf '%s\tmodels_suite\t%s\n' "$(date -u +%FT%TZ)" "$([[ $FAILURES -eq 0 ]] && echo complete || echo failed)" \
  | tee -a "$STATUS_FILE"
exit "$FAILURES"
