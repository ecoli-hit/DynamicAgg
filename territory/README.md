# Territory isolation contract (SPEC-037)

This directory is the isolated base for the two-step territory pipeline. SPEC-037
only establishes the contract and a K=1 injection smoke path; extraction logic is
reserved for SPEC-038.

## Two-step information flow

1. `extract/` may read the frozen eight-draft corpus and emits one bounded territory
   file with `route`, `landmarks`, `hazards`, and `blind_spots`.
2. `inject/run_territory_rollout.py` receives only the original question and that
   territory file. The territory file is the sole research-information channel.
   The second-step process must never open, enumerate, or receive a path to any of
   the eight source drafts.

The demo territory is capped at 1,200 English words and must begin with
`DEMO MANUAL ARTIFACT — NOT A SPEC-038 EXTRACTION`. Production word budgets and the
schema are finalized by SPEC-038.

## Directory responsibilities

| Path | Responsibility in SPEC-037 |
|---|---|
| `schemas/` | Draft contract for the four-part territory object. |
| `extract/` | Placeholder only; no extraction implementation in this spec. |
| `inject/` | Original `react_agent` wrapper; question + territory in, K=1 result out. |
| `audit/` | Lightweight stage names, paired node-I/O paths, and started/result journal helper. |
| `configs/paths.py` | Central read-only locations for the frozen corpus and dataset. The injector must not import it. |

## Lightweight audit convention

- Stages have fixed names: `baseline`, `inject`, `judge`, `case` (U01).
- Each node archives `<run>/<stage>/<id>__INPUT.json` and
  `<run>/<stage>/<id>__OUTPUT.json` (U05).
- `<run>/journal.jsonl` appends one `started` and one `result` event with the same
  stable `v1:<sha256>` key for every completed node (U09).

This is deliberately not a workflow runtime: no scheduler, resume engine,
persistence service, or extraction framework is introduced here.
