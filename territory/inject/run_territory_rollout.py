#!/usr/bin/env python3
"""Run one fresh ReAct report from only an original question and a territory file."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "rollout"), str(ROOT)]

from prompts import SYSTEM_PROMPT  # noqa: E402
from react_agent import MultiTurnReactAgent  # noqa: E402
from territory.audit import journal, node_paths, stable_key  # noqa: E402

MANDATE = """
## TERRITORY MANDATE (strict)
1. Your FIRST action MUST be a search or visit tool call. Do not write prose first.
2. Treat the territory as the sole route map: re-check every hazard and scout every blind spot with tools before drafting.
3. Preserve relevant landmark specifics and their supplied citations, but independently verify them; the territory is not an answer to copy.
4. Follow the original question's audience, deliverable, and format. Return the substantive final report, not an outline or a description of the territory.
""".strip()


class TerritoryAgent(MultiTurnReactAgent):
    async def add_auto_judge(self, result, item):
        return result


def _atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _question(value: str) -> str:
    return Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value


def _first_assistant_used_tool(messages: list[dict]) -> bool:
    first = next((message for message in messages if message.get("role") == "assistant"), {})
    return bool(first.get("tool_calls"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--territory", type=Path, required=True, help="sole research-information input")
    parser.add_argument("--question", required=True, help="literal question, or @path to a question text file")
    parser.add_argument("--out", type=Path, required=True, help="run root for paired I/O and journal")
    parser.add_argument("--node-id", default="territory_k1")
    parser.add_argument("--model", required=True, help="local model/tokenizer path")
    parser.add_argument("--api-base", default="http://localhost:6000/v1")
    parser.add_argument("--task", default="researchrubrics")
    parser.add_argument("--max-llm-calls", type=int, default=100)
    parser.add_argument("--max-context-tokens", type=int, default=108000)
    parser.add_argument("--gen-max-tokens", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=5400)
    args = parser.parse_args()

    territory = args.territory.read_text(encoding="utf-8")
    question = _question(args.question).strip()
    if not territory.strip() or not question:
        raise ValueError("territory and question must both be non-empty")

    model_name = Path(args.model.rstrip("/")).name
    model_type = "glm" if "glm" in model_name.lower() else "qwen"
    generate_cfg = {
        "model": f"hosted_vllm/{model_name}",
        "api_base": args.api_base,
        "api_key": "EMPTY",
        "max_tokens": args.gen_max_tokens,
        "parallel_tool_calls": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "timeout": int(args.timeout),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}},
    }
    llm = {"model": args.model, "model_name": model_name, "model_type": model_type, "generate_cfg": generate_cfg}
    archived_input = {
        "question": question,
        "territory": territory,
        "territory_path": str(args.territory.resolve()),
        "model": args.model,
        "api_base": args.api_base,
    }
    input_path, output_path = node_paths(args.out, "inject", args.node_id)
    key = stable_key("inject", args.node_id, archived_input)
    _atomic_json(input_path, archived_input)
    journal(args.out, "started", key, args.node_id)

    agent = TerritoryAgent(
        llm=llm,
        function_list=None,
        task=args.task,
        max_llm_call_per_run=args.max_llm_calls,
        max_tokens=args.max_context_tokens,
    )
    agent.system_prompt = SYSTEM_PROMPT + "\n\n## TERRITORY (sole route map)\n" + territory.strip() + "\n\n" + MANDATE
    result = asyncio.run(asyncio.wait_for(agent._run({"item": {"question": question}, "rollout_idx": 1}, args.model), args.timeout))
    if not _first_assistant_used_tool(result.get("messages") or []):
        raise RuntimeError("first assistant action was not a tool call")

    _atomic_json(output_path, result)
    report_path = args.out / "inject" / f"{args.node_id}__REPORT.md"
    report_path.write_text(result.get("prediction") or "", encoding="utf-8")
    summary = {"output": str(output_path), "report": str(report_path), "prediction_chars": len(result.get("prediction") or "")}
    journal(args.out, "result", key, args.node_id, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
