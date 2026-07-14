"""SPEC-037's small audit convention; not a workflow runtime."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

STAGES = ("baseline", "inject", "judge", "case")


def stable_key(stage: str, node_id: str, payload: object) -> str:
    blob = json.dumps([stage, node_id, payload], ensure_ascii=False, sort_keys=True).encode()
    return "v1:" + hashlib.sha256(blob).hexdigest()


def node_paths(run_dir: Path, stage: str, node_id: str) -> tuple[Path, Path]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", node_id):
        raise ValueError(f"unsafe node id: {node_id!r}")
    stage_dir = Path(run_dir) / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir / f"{node_id}__INPUT.json", stage_dir / f"{node_id}__OUTPUT.json"


def journal(run_dir: Path, event: str, key: str, node_id: str, result: object = None) -> None:
    if event not in {"started", "result"}:
        raise ValueError(f"unknown journal event: {event}")
    row = {"type": event, "key": key, "node_id": node_id}
    if event == "result":
        row["result"] = result
    path = Path(run_dir) / "journal.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # ponytail: single-process K=1 append is enough for 037; add locking if writers become concurrent.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
