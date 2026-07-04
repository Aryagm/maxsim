from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from ops.vast.vast_guard import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_SPEND_CAP,
    InstanceRecord,
    append_ledger,
    assert_spend_allowed,
    create_instance_command,
    destroy_instances,
    destroyable_owned_instances,
    load_ledger,
    make_label,
    run_vastai,
)


DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"


def cmd_install_validate(_: argparse.Namespace) -> int:
    print("Install or upgrade with: python -m pip install --upgrade vastai")
    print(run_vastai(["search", "offers", "--limit", "3"]))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    query = args.query or "gpu_name=RTX_4090 num_gpus>=1"
    command = ["search", "offers", query, "--limit", str(args.limit)]
    print(run_vastai(command))
    return 0


def cmd_create(args: argparse.Namespace) -> int:
    ledger = load_ledger(args.ledger)
    assert_spend_allowed(
        ledger,
        new_hourly_cost=args.hourly_cost,
        planned_hours=args.planned_hours,
        cap=args.cap,
    )

    label = make_label(args.git_sha, args.role)
    command = create_instance_command(args.offer_id, image=args.image, disk=args.disk, label=label)
    output = run_vastai(command)
    print(output)

    instance_id = _extract_instance_id(output)
    if instance_id is None:
        print("Could not infer instance id from VAST output; record it manually with record-created.", file=sys.stderr)
        print(f"label={label}", file=sys.stderr)
        return 2

    append_ledger(
        InstanceRecord(
            instance_id=instance_id,
            label=label,
            offer_id=args.offer_id,
            hourly_cost=args.hourly_cost,
            created_at=datetime.now(timezone.utc).isoformat(),
            role=args.role,
        ),
        args.ledger,
    )
    print(f"recorded instance {instance_id} in {args.ledger}")
    return 0


def cmd_record_created(args: argparse.Namespace) -> int:
    append_ledger(
        InstanceRecord(
            instance_id=args.instance_id,
            label=args.label,
            offer_id=args.offer_id,
            hourly_cost=args.hourly_cost,
            created_at=datetime.now(timezone.utc).isoformat(),
            role=args.role,
        ),
        args.ledger,
    )
    print(f"recorded instance {args.instance_id} in {args.ledger}")
    return 0


def cmd_remote(args: argparse.Namespace) -> int:
    ssh_target = run_vastai(["ssh-url", str(args.instance_id)]).strip()
    quoted = shlex.quote(args.command)
    print(f"ssh {ssh_target} {quoted}")
    return 0


def cmd_destroy_owned(args: argparse.Namespace) -> int:
    ledger = load_ledger(args.ledger)
    raw_live = json.loads(Path(args.live_json).read_text()) if args.live_json else []
    live = raw_live if isinstance(raw_live, list) else [raw_live]
    ids = destroyable_owned_instances(ledger, live)
    commands = destroy_instances(ids, dry_run=args.dry_run)
    for command in commands:
        print("vastai " + " ".join(command))
    if not args.dry_run:
        print(f"destroyed {len(commands)} owned maxsim VAST instance(s)")
    return 0


def _extract_instance_id(output: str) -> int | None:
    patterns = [
        r"instance\s+(\d+)",
        r"id['\"]?\s*[:=]\s*(\d+)",
        r"\b(\d{4,})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="maxsim VAST.ai workflow helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("install-validate")

    search = subparsers.add_parser("search")
    search.add_argument("query", nargs="?")
    search.add_argument("--limit", type=int, default=10)

    create = subparsers.add_parser("create")
    create.add_argument("offer_id", type=int)
    create.add_argument("--hourly-cost", type=float, required=True)
    create.add_argument("--planned-hours", type=float, default=3.0)
    create.add_argument("--cap", type=float, default=DEFAULT_SPEND_CAP)
    create.add_argument("--role", default="cuda-smoke")
    create.add_argument("--git-sha", default="nogit")
    create.add_argument("--image", default=DEFAULT_IMAGE)
    create.add_argument("--disk", type=int, default=80)
    create.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)

    record = subparsers.add_parser("record-created")
    record.add_argument("instance_id", type=int)
    record.add_argument("--label", required=True)
    record.add_argument("--offer-id", type=int, required=True)
    record.add_argument("--hourly-cost", type=float, required=True)
    record.add_argument("--role", default="")
    record.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)

    remote = subparsers.add_parser("remote-command")
    remote.add_argument("instance_id", type=int)
    remote.add_argument("command")

    destroy = subparsers.add_parser("destroy-owned")
    destroy.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    destroy.add_argument("--live-json", type=Path, required=True)
    destroy.add_argument("--execute", action="store_true")
    destroy.set_defaults(dry_run=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "execute", False):
        args.dry_run = False

    handlers = {
        "install-validate": cmd_install_validate,
        "search": cmd_search,
        "create": cmd_create,
        "record-created": cmd_record_created,
        "remote-command": cmd_remote,
        "destroy-owned": cmd_destroy_owned,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
