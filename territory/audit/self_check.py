#!/usr/bin/env python3
"""Small runnable check for the SPEC-037 territory contract."""

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from territory.audit import STAGES, journal, node_paths, stable_key

ROOT = Path(__file__).resolve().parents[1]
schema = json.loads((ROOT / "schemas/territory_brief.schema.json").read_text(encoding="utf-8"))
assert schema["required"] == ["route", "landmarks", "hazards", "blind_spots"]
assert STAGES == ("baseline", "inject", "judge", "case")

injector = (ROOT / "inject/run_territory_rollout.py").read_text(encoding="utf-8")
assert 'add_argument("--rollout' not in injector
assert "output/rollout" not in injector

with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp)
    input_path, output_path = node_paths(run, "inject", "q14_k1")
    key = stable_key("inject", "q14_k1", {"question": "q", "territory": "t"})
    journal(run, "started", key, "q14_k1")
    journal(run, "result", key, "q14_k1", {"output": str(output_path)})
    rows = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    assert [row["type"] for row in rows] == ["started", "result"]
    assert rows[0]["key"] == rows[1]["key"] and input_path.name.endswith("__INPUT.json")

print("SPEC-037 territory self-check: PASS")
