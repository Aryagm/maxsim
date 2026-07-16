#!/usr/bin/env bash
# Production, quality, scaling, reducer, cascade, and closest-prior runs for
# one RTX 4090. Every case is resumable and writes an independent log/marker.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/bitmax}"
RUN="${RUN:-/workspace/paper-systems-20260715}"
PY="${PY:-/opt/conda/bin/python}"
PLAID_PY="${PLAID_PY:-/workspace/plaid-env/bin/python}"
SOURCE_COMMIT="${SOURCE_COMMIT:-unknown}"
INPUT_MANIFEST="${INPUT_MANIFEST:-/workspace/paper-inputs.sha256}"
CACHE_WAIT_SECONDS="${CACHE_WAIT_SECONDS:-21600}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
mkdir -p "$RUN"/{results,logs,done,scale}
exec > >(tee -a "$RUN/logs/systems.log") 2>&1

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

quality_matrix() {
  local cache="$1" output="$2" formats="$3"
  test -s "$cache"
  "$PY" -m benchmarks.run_quality_matrix "$cache" \
    --output "$output" --formats "$formats" --device cuda \
    --checkpoint-every 8 --max-chunk-tokens 262144
}

production_run() {
  local cache="$1" output="$2" query_limit="$3"
  shift 3
  test -s "$cache"
  "$PY" -m benchmarks.run_production_retrieval \
    --input "$cache" --output "$output" --device cuda \
    --modes dense,binary,binary_token_scale_u4,pooled_binary,int4,int4_per_token,int4_residual \
    --pooled-binary-pool-factor 3 \
    --query-limit "$query_limit" --seed 20260715 \
    --warmup 3 --latency-repeats 3 --throughput-batch-size 8 \
    --int4-query fp32 --dense-chunk-tokens 262144 "$@"
}

decide_fiqa_cascade() {
  local input="$RUN/results/production-fiqa57k.json"
  local output="$RUN/results/cascade-decision-fiqa57k.json"
  "$PY" - "$input" "$output" "$RUN/cascade-best-m.txt" <<'PY'
import json
import sys
from datetime import datetime, timezone

source, output, best_path = sys.argv[1:]
payload = json.load(open(source))
rows = {row["case_id"]: row for row in payload["cases"]}
coarse = rows["int4_per_token"]
full = rows["int4_residual"]
coarse_q = float(coarse["quality"]["ndcg_at_10_mean"])
full_q = float(full["quality"]["ndcg_at_10_mean"])
coarse_ms = float(coarse["latency"]["p50_ms"])
full_ms = float(full["latency"]["p50_ms"])
gap = full_q - coarse_q
decisions = []
for case_id, row in sorted(rows.items()):
    if not case_id.startswith("int4_residual_cascade_m"):
        continue
    quality = float(row["quality"]["ndcg_at_10_mean"])
    latency = float(row["latency"]["p50_ms"])
    recovery = 1.0 if gap <= 1e-12 and quality >= full_q - 1e-12 else ((quality - coarse_q) / gap if gap > 0 else 0.0)
    overhead = latency / coarse_ms
    speedup = full_ms / latency
    qualifies = recovery >= 0.90 and (overhead <= 1.25 or speedup >= 1.50)
    decisions.append({
        "case_id": case_id,
        "candidates": int(row["rescore_candidates"]),
        "ndcg_at_10": quality,
        "quality_gap_recovery": recovery,
        "p50_ms": latency,
        "overhead_vs_prefix": overhead,
        "speedup_vs_full_residual": speedup,
        "qualifies": qualifies,
    })
qualified = [row for row in decisions if row["qualifies"]]
best = min(qualified, key=lambda row: (row["p50_ms"], row["candidates"])) if qualified else None
decision = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "policy": {
        "minimum_quality_gap_recovery": 0.90,
        "maximum_overhead_vs_prefix": 1.25,
        "minimum_speedup_vs_full_residual": 1.50,
        "latency_condition": "overhead_vs_prefix <= maximum OR speedup_vs_full_residual >= minimum",
    },
    "prefix": {"ndcg_at_10": coarse_q, "p50_ms": coarse_ms},
    "full_residual": {"ndcg_at_10": full_q, "p50_ms": full_ms},
    "candidates": decisions,
    "retain_cascade_in_main_paper": best is not None,
    "selected": best,
}
with open(output, "w") as handle:
    json.dump(decision, handle, indent=2, sort_keys=True)
    handle.write("\n")
if best is not None:
    with open(best_path, "w") as handle:
        handle.write(f"{best['candidates']}\n")
else:
    from pathlib import Path
    Path(best_path).unlink(missing_ok=True)
PY
}

run_visual_cascade_if_selected() {
  if [[ ! -s "$RUN/cascade-best-m.txt" ]]; then
    echo "FiQA cascade gate did not qualify; skipping visual replication."
    return 0
  fi
  local candidates
  candidates="$(tr -d '[:space:]' < "$RUN/cascade-best-m.txt")"
  "$PY" -m benchmarks.run_production_retrieval \
    --input caches-full/vidore-mixed-public-unique-colqwen2-limit10000.npz \
    --output "$RUN/results/production-visual10k-cascade-m${candidates}.json" \
    --device cuda --modes int4_per_token,int4_residual \
    --query-limit 2000 --seed 20260715 --warmup 3 --latency-repeats 3 \
    --throughput-batch-size 8 --cascade-candidates "$candidates" \
    --int4-query fp32 --dense-chunk-tokens 262144
}

create_scale_plan() {
  local source="$1" output_dir="$2" sizes="$3"
  "$PY" -m benchmarks.run_nested_scale plan \
    --input "$source" --output-dir "$output_dir" \
    --sizes "$sizes" --seeds 17,29,41 \
    --query-limit 256 --query-seed 20260715 \
    --implementations dense_fp16_vectorized,bitmax_binary,bitmax_binary_token_scale_u4,bitmax_pooled_binary3,bitmax_int4,bitmax_int4_per_token,bitmax_int4_residual \
    --device cuda --repeat 3 --k 10 --metric-ks 1,5,10
}

run_scale_plan() {
  local plan="$1"
  "$PY" -m benchmarks.run_nested_scale run --plan "$plan" --continue-on-error
  "$PY" - "$plan" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).with_name("nested-scale-checkpoint.json")
payload = json.load(open(checkpoint))
assert payload["status"] == "ok", payload
assert payload["failure_count"] == 0, payload
PY
}

run_plaid() {
  local cache="$1" output="$2" query_limit="$3"
  test -x "$PLAID_PY"
  "$PLAID_PY" -m benchmarks.run_plaid_baseline \
    --input "$cache" --output "$output" \
    --backend fast-plaid --device cuda --k 10 --metric-ks 1,5,10 \
    --repeat 3 --warmup 1 --limit-queries "$query_limit" --query-seed 20260715 \
    --nbits 4 --n-full-scores 4096 --n-ivf-probe 4 --seed 42 --no-use-triton
}

wait_for_verified_inputs() {
  test -s "$INPUT_MANIFEST"
  local deadline=$((SECONDS + CACHE_WAIT_SECONDS))
  local missing

  while true; do
    missing="$($PY - "$INPUT_MANIFEST" <<'PY'
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
for line in manifest.read_text().splitlines():
    if not line.strip():
        continue
    path = line.split(maxsplit=1)[1].lstrip(" *")
    if not Path(path).is_file():
        print(path)
PY
)"
    if [[ -z "$missing" ]]; then
      break
    fi
    if (( SECONDS >= deadline )); then
      printf 'Timed out waiting for required caches:\n%s\n' "$missing" >&2
      return 1
    fi
    printf 'Waiting for required caches:\n%s\n' "$missing"
    sleep 30
  done

  sha256sum --check --strict "$INPUT_MANIFEST"
}

"$PY" - "$RUN/environment.json" "$SOURCE_COMMIT" "$PLAID_PY" <<'PY'
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

output, source_commit, plaid_python = sys.argv[1:]
probe = "import json,torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}))"
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_commit": source_commit,
    "platform": platform.platform(),
    "nvidia_smi": subprocess.check_output(["nvidia-smi", "-L"], text=True).strip(),
    "runtime": json.loads(subprocess.check_output([sys.executable, "-c", probe], text=True)),
}
if subprocess.call([plaid_python, "-c", "import torch,fast_plaid,maxsim"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
    plaid_probe = "import importlib.metadata,json,torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'fast_plaid':importlib.metadata.version('fast-plaid')}))"
    payload["plaid_runtime"] = json.loads(subprocess.check_output([plaid_python, "-c", plaid_probe], text=True))
else:
    payload["plaid_runtime"] = None
with open(output, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

FULL_FORMATS=dense_fp16,binary,binary_token_scale_u4,pool3_binary,int4_per_tensor,int4_per_token,int8_per_token
NEW_INT4_FORMATS=int4_per_token,int4_per_tensor,int8_per_token

run_case cuda_tests "$PY" -m pytest -m cuda -q
run_case reducers_n512 "$PY" -m benchmarks.run_cuda_reducers \
  --output "$RUN/results/reducers-n512.json" --repeat 200 --warmup 20 \
  --batch 1 --query-tokens 32 --docs 512 --min-doc-tokens 64 --max-doc-tokens 256 \
  --candidates 128 --dim 128 --seed 20260715
run_case reducers_n4096 "$PY" -m benchmarks.run_cuda_reducers \
  --output "$RUN/results/reducers-n4096.json" --repeat 100 --warmup 20 \
  --batch 1 --query-tokens 32 --docs 4096 --min-doc-tokens 64 --max-doc-tokens 256 \
  --candidates 512 --dim 128 --seed 20260715
run_case verify_input_caches wait_for_verified_inputs

run_case quality_gte_scifact quality_matrix \
  caches-full/beir/beir-scifact-gte-moderncolbert.npz \
  "$RUN/results/quality-gte-scifact-full.json" "$FULL_FORMATS"
run_case quality_gte_nfcorpus quality_matrix \
  caches-full/beir/beir-nfcorpus-gte-moderncolbert.npz \
  "$RUN/results/quality-gte-nfcorpus-full.json" "$FULL_FORMATS"
run_case quality_gte_fiqa quality_matrix \
  caches-full/beir/beir-fiqa-gte-moderncolbert.npz \
  "$RUN/results/quality-gte-fiqa-full.json" "$FULL_FORMATS"

for cache in \
  caches-full/vidore-arxivqa-test-colqwen2-limit500.npz \
  caches-full/vidore-docvqa-test-colqwen2-limit500.npz \
  caches-full/vidore-infovqa-test-colqwen2-limit500.npz \
  caches-full/vidore-tatdqa-test-colqwen2-limit1663.npz \
  caches-full/vidore-syntheticdocqa-ai-colqwen2-limit1000.npz \
  caches-full/vidore-syntheticdocqa-energy-colqwen2-limit1000.npz \
  caches-full/vidore-syntheticdocqa-government-colqwen2-limit1000.npz \
  caches-full/vidore-syntheticdocqa-healthcare-colqwen2-limit1000.npz \
  caches-full/vidore-syntheticdocqa-shift-colqwen2-limit1000.npz; do
  slug="$(basename "$cache" .npz)"
  run_case "quality_${slug}" quality_matrix "$cache" \
    "$RUN/results/quality-${slug}-new-int4.json" "$NEW_INT4_FORMATS"
done

run_case production_fiqa57k production_run \
  caches-full/beir/beir-fiqa-gte-moderncolbert.npz \
  "$RUN/results/production-fiqa57k.json" 0 --cascade-candidates 100,200,400
run_case cascade_decision_fiqa57k decide_fiqa_cascade
run_case production_visual10k production_run \
  caches-full/vidore-mixed-public-unique-colqwen2-limit10000.npz \
  "$RUN/results/production-visual10k-2kq.json" 2000
run_case cascade_visual_replication run_visual_cascade_if_selected

run_case scale_visual_plan create_scale_plan \
  caches-full/vidore-mixed-public-unique-colqwen2-limit10000.npz \
  "$RUN/scale/visual" 1000,5000,10000
run_case scale_visual_run run_scale_plan "$RUN/scale/visual/nested-scale-plan.json"
run_case scale_fiqa_plan create_scale_plan \
  caches-full/beir/beir-fiqa-gte-moderncolbert.npz \
  "$RUN/scale/fiqa" 1000,5000,10000,25000,57638
run_case scale_fiqa_run run_scale_plan "$RUN/scale/fiqa/nested-scale-plan.json"

run_case plaid_fiqa57k run_plaid \
  caches-full/beir/beir-fiqa-gte-moderncolbert.npz \
  "$RUN/results/plaid-fiqa57k.json" 0
run_case plaid_visual10k run_plaid \
  caches-full/vidore-mixed-public-unique-colqwen2-limit10000.npz \
  "$RUN/results/plaid-visual10k-2kq.json" 2000

find "$RUN/results" "$RUN/scale" -type f -name '*.json' -print0 | sort -z | xargs -0 sha256sum \
  > "$RUN/result-checksums.sha256"
find caches-full -type f -name '*.npz' -print0 | sort -z | xargs -0 sha256sum \
  > "$RUN/source-cache-checksums.sha256"

printf '%s\tsystems_suite\t%s\n' "$(date -u +%FT%TZ)" "$([[ $FAILURES -eq 0 ]] && echo complete || echo failed)" \
  | tee -a "$STATUS_FILE"
exit "$FAILURES"
