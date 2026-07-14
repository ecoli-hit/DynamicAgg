"""Frozen read-only inputs. The second-step injector must not import this module."""

from pathlib import Path

DYAGG_ROOT = Path("/data/xzc/DyAggAgent")
FROZEN_ROLLOUT_ROOT = DYAGG_ROOT / "output/rollout/GLM-4.7-Flash/researchrubrics"
FROZEN_ROLLOUT_DIRS = tuple(FROZEN_ROLLOUT_ROOT / f"iter{i}" for i in range(1, 9))
RESEARCHRUBRICS_DATASET = DYAGG_ROOT / "data/researchrubrics.jsonl"
