from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MAXSIM_LABEL_PREFIX = "maxsim-v0-"
DEFAULT_LEDGER_PATH = Path(".vast/maxsim-instances.jsonl")
DEFAULT_SPEND_CAP = 25.0


@dataclass(frozen=True)
class InstanceRecord:
    instance_id: int
    label: str
    offer_id: int
    hourly_cost: float
    created_at: str = ""
    role: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "label": self.label,
            "offer_id": self.offer_id,
            "hourly_cost": self.hourly_cost,
            "created_at": self.created_at,
            "role": self.role,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "InstanceRecord":
        return cls(
            instance_id=int(value["instance_id"]),
            label=str(value["label"]),
            offer_id=int(value["offer_id"]),
            hourly_cost=float(value["hourly_cost"]),
            created_at=str(value.get("created_at", "")),
            role=str(value.get("role", "")),
        )


def make_label(git_sha: str, role: str, *, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d%H%M%S")
    clean_sha = git_sha[:12] if git_sha else "nogit"
    clean_role = role.replace("_", "-")
    return f"{MAXSIM_LABEL_PREFIX}{timestamp}-{clean_sha}-{clean_role}"


def load_ledger(path: Path = DEFAULT_LEDGER_PATH) -> list[InstanceRecord]:
    if not path.exists():
        return []
    records: list[InstanceRecord] = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(InstanceRecord.from_json(json.loads(line)))
    return records


def append_ledger(record: InstanceRecord, path: Path = DEFAULT_LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")


def projected_spend(ledger: Iterable[InstanceRecord], *, new_hourly_cost: float, planned_hours: float) -> float:
    existing = sum(record.hourly_cost * planned_hours for record in ledger)
    return round(existing + new_hourly_cost * planned_hours, 2)


def assert_spend_allowed(
    ledger: Iterable[InstanceRecord],
    *,
    new_hourly_cost: float,
    planned_hours: float,
    cap: float = DEFAULT_SPEND_CAP,
) -> None:
    total = projected_spend(ledger, new_hourly_cost=new_hourly_cost, planned_hours=planned_hours)
    if total > cap:
        raise RuntimeError(f"projected VAST spend ${total:.2f} exceeds cap ${cap:.2f}")


def destroyable_owned_instances(
    ledger: Iterable[InstanceRecord],
    live_instances: Iterable[dict[str, Any]],
    *,
    prefix: str = MAXSIM_LABEL_PREFIX,
) -> list[int]:
    ledger_by_id = {record.instance_id: record for record in ledger if record.label.startswith(prefix)}
    destroyable: list[int] = []
    for live in live_instances:
        instance_id = int(live.get("id", live.get("instance_id", -1)))
        live_label = str(live.get("label", live.get("name", "")))
        if instance_id in ledger_by_id and live_label.startswith(prefix) and live_label == ledger_by_id[instance_id].label:
            destroyable.append(instance_id)
    return sorted(destroyable)


def run_vastai(args: list[str]) -> str:
    venv_cli = Path(sys.executable).with_name("vastai")
    executable = str(venv_cli) if venv_cli.exists() else "vastai"
    completed = subprocess.run([executable, *args], check=True, text=True, capture_output=True)
    return completed.stdout


def create_instance_command(offer_id: int, *, image: str, disk: int, label: str) -> list[str]:
    return [
        "create",
        "instance",
        str(offer_id),
        "--image",
        image,
        "--disk",
        str(disk),
        "--ssh",
        "--direct",
        "--label",
        label,
    ]


def destroy_instances(instance_ids: Iterable[int], *, dry_run: bool = True) -> list[list[str]]:
    commands = [["destroy", "instance", str(instance_id), "-y"] for instance_id in instance_ids]
    if not dry_run:
        for command in commands:
            run_vastai(command)
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safety helpers for maxsim VAST instances.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    label_parser = subparsers.add_parser("label")
    label_parser.add_argument("--git-sha", required=True)
    label_parser.add_argument("--role", required=True)

    spend_parser = subparsers.add_parser("check-spend")
    spend_parser.add_argument("--hourly-cost", type=float, required=True)
    spend_parser.add_argument("--planned-hours", type=float, required=True)
    spend_parser.add_argument("--cap", type=float, default=DEFAULT_SPEND_CAP)
    spend_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)

    args = parser.parse_args(argv)
    if args.command == "label":
        print(make_label(args.git_sha, args.role))
        return 0
    if args.command == "check-spend":
        assert_spend_allowed(
            load_ledger(args.ledger),
            new_hourly_cost=args.hourly_cost,
            planned_hours=args.planned_hours,
            cap=args.cap,
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
