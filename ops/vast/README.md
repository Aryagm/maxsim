# VAST workflow

Use these helpers only for `bitmax` benchmark instances. Cleanup is ledger-based
and must never delete instances that were not created and recorded by this
project.

## Validate CLI

```bash
python -m pip install --upgrade vastai
python -m ops.vast.vast_workflow install-validate
```

## Search and create

```bash
python -m ops.vast.vast_workflow search 'gpu_name=RTX_4090 num_gpus>=1' --limit 10
python -m ops.vast.vast_workflow create <offer_id> \
  --hourly-cost <offer_hourly_cost> \
  --planned-hours 3 \
  --cap 25 \
  --git-sha "$(git rev-parse --short=12 HEAD)" \
  --role cuda-smoke
```

The create command labels instances as `bitmax-v0-<timestamp>-<gitsha>-<role>`
and appends them to `.vast/bitmax-instances.jsonl`.

## Remote commands

```bash
python -m ops.vast.vast_workflow remote-command <instance_id> \
  'cd bitmax && MAXSIM_BUILD_CUDA=1 python -m pip install -e ".[dev]" && pytest -m cuda'
```

## Cleanup

First save live instances from VAST to JSON. Then dry-run cleanup:

```bash
python -m ops.vast.vast_workflow destroy-owned --live-json live-instances.json
```

Only execute after reviewing the printed `vastai destroy instance <id>` commands:

```bash
python -m ops.vast.vast_workflow destroy-owned --live-json live-instances.json --execute
```
