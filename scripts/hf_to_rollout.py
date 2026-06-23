#!/usr/bin/env python3
# 〔讲解〕模块说明:本文件是 scripts/ 数据准备环节的一个转换器。
# 〔讲解〕AggAgent 把每条 ReAct 搜索轨迹(rollout)发布成"扁平 parquet"——一行一条 rollout,
# 〔讲解〕结构化字段(messages/instance/cost...)被压成 JSON 字符串塞进单元格。
# 〔讲解〕但下游的 aggregation/aggregate.py、scripts/collect_trajs.py 需要的是"目录化布局":
# 〔讲解〕output/rollout/<MODEL>/<DATASET>/iter1..iterN/<question-id>.json(一题一文件、按 iter 分桶)。
# 〔讲解〕本脚本就负责把"一个 parquet"展开成这套磁盘目录结构,是聚合阶段的上游喂料环节。
"""Convert published AggAgent rollout parquets into the on-disk layout that
``aggregation/aggregate.py`` and ``scripts/collect_trajs.py`` expect.

The parquets we publish on Hugging Face (and ship in the BrowseComp tar) are
flat: one row per rollout, with the structured per-trajectory fields
(``messages``, ``instance``, ``cost``, ...) JSON-encoded as strings. The
codebase, however, expects one JSON file per (question, iter) pair under

    output/rollout/<MODEL>/<DATASET>/
        iter1/  <question-id>.json
        iter2/  <question-id>.json
        ...
        iterN/  <question-id>.json

This script materializes that layout from a single parquet.

Usage (local parquet)::

    python scripts/hf_to_rollout.py path/to/Qwen3.5-122B-A10B.parquet \
        --out output/rollout/Qwen3.5-122B-A10B/deepsearchqa

Usage (download from Hugging Face)::

    python scripts/hf_to_rollout.py \
        --repo yoonsanglee/deepsearchqa-react \
        --model Qwen3.5-122B-A10B \
        --out output/rollout/Qwen3.5-122B-A10B/deepsearchqa
"""
# 〔讲解〕from __future__ import annotations:让类型注解延迟求值(当作字符串),
# 〔讲解〕这样下面写 `Path | None` 这种新式联合类型语法在旧版 Python 上也不报错。
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

# 〔讲解〕这几个字段在 parquet 里是被 JSON 序列化成字符串存的,读出来后需要反序列化回 dict/list。
# 〔讲解〕下面 parse_record 会专门挑这些 key 做 json.loads 还原。
JSON_FIELDS = ("metadata", "instance", "cost", "messages", "debug_data", "auto_judge")


# 〔讲解〕load_rows:数据装载入口。两种来源二选一——本地 parquet 路径,或从 Hugging Face 下载。
# 〔讲解〕入参 parquet_path(本地文件)、repo(HF 数据集仓库名)、model(HF 里的 config 子集名)。
# 〔讲解〕返回:一个 list[dict],每个 dict 是一行原始 rollout 记录。
def load_rows(parquet_path: Path | None, repo: str | None, model: str | None):
    # 〔讲解〕优先走本地 parquet 分支:有路径就直接读文件,不联网。
    if parquet_path is not None:
        # 〔讲解〕惰性 import:只有真要读 parquet 时才加载 pyarrow,避免无谓依赖。
        import pyarrow.parquet as pq
        # 〔讲解〕read_table 读成 Arrow 表,to_pylist() 转成 Python 的 list[dict](每行一个字典)。
        return pq.read_table(parquet_path).to_pylist()
    # 〔讲解〕否则走 HF 在线分支:同样惰性 import datasets 库。
    from datasets import load_dataset
    # 〔讲解〕name=model 指定数据集内的具体配置子集(不同模型是不同 config);split="train" 取训练分片。
    ds = load_dataset(repo, name=model, split="train")
    # 〔讲解〕把每行转成普通 dict,统一成和本地分支一样的 list[dict] 返回格式。
    return [dict(row) for row in ds]


# 〔讲解〕parse_record:把一行原始记录里被压成字符串的 JSON 字段还原成真正的对象。
# 〔讲解〕入参 row 是一行(可能字段值是 JSON 字符串);返回还原后的 dict,可直接 dump 成结构化 JSON。
def parse_record(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        # 〔讲解〕只对 JSON_FIELDS 名单里、且当前值确实是字符串的字段尝试反序列化。
        if k in JSON_FIELDS and isinstance(v, str):
            try:
                out[k] = json.loads(v)
                # 〔讲解〕解析成功就用还原后的对象,continue 跳过下面的原样赋值。
                continue
            except json.JSONDecodeError:
                # 〔讲解〕容错:万一某字段不是合法 JSON(数据脏),不报错,落到下面原样保留字符串。
                pass
        # 〔讲解〕非 JSON 字段、或解析失败的字段,原值照搬。
        out[k] = v
    return out


# 〔讲解〕question_id:为每道题生成一个稳定的文件名 id(决定 <question-id>.json 的名字)。
# 〔讲解〕"稳定"很关键——同一道题每次运行都得到同一 id,这样多个 iter 才能用同名文件对齐成一组。
# 〔讲解〕入参 question(题目文本)、instance(该题的结构化元数据);返回一个可做文件名的字符串。
def question_id(question: str, instance: object) -> str:
    """Stable filename id. Prefer instance.id / instance.prompt_id when present
    (HealthBench, etc.); otherwise hash the question text."""
    # 〔讲解〕优先用数据集自带的稳定标识(HealthBench 等会带 id/prompt_id/qid),按优先级挨个找。
    if isinstance(instance, dict):
        for key in ("id", "prompt_id", "qid"):
            v = instance.get(key)
            # 〔讲解〕找到第一个非空字符串就直接拿来当 id。
            if isinstance(v, str) and v:
                return v
    # 〔讲解〕兜底:数据集没给现成 id,就对题目文本做 SHA1 哈希取前 12 位,保证同题同名、不同题大概率不撞。
    return "q_" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]


# 〔讲解〕main:命令行主流程。负责解析参数 → 装载数据 → 按题分组 → 建 iter 目录 → 逐题逐 iter 落盘 JSON。
def main():
    # 〔讲解〕用模块 docstring 当帮助文本;RawDescriptionHelpFormatter 保留 docstring 里的换行/缩进不被重排。
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 〔讲解〕互斥组 required=True:本地 parquet 位置参数 与 --repo 二者必须且只能选其一。
    src = ap.add_mutually_exclusive_group(required=True)
    # 〔讲解〕位置参数 parquet(nargs="?" 表示可选出现一次),本地文件路径。
    src.add_argument("parquet", nargs="?", type=Path, help="Path to a local parquet file")
    src.add_argument("--repo", help="Hugging Face dataset repo (e.g. yoonsanglee/deepsearchqa-react)")
    ap.add_argument("--model", help="Config name when using --repo (e.g. Qwen3.5-122B-A10B)")
    # 〔讲解〕--out 必填:输出 rollout 目录,iter1..iterN 子目录就建在它下面。
    ap.add_argument("--out", required=True, type=Path, help="Output rollout dir")
    args = ap.parse_args()

    # 〔讲解〕额外校验:走 HF 在线分支(--repo)时必须同时给 --model 指定子集,否则报错退出。
    if args.repo and not args.model:
        ap.error("--model is required with --repo")

    rows = load_rows(args.parquet, args.repo, args.model)
    print(f"Loaded {len(rows)} rows")

    # Group rollouts by question
    # 〔讲解〕按题目分组:同一道题的多条 rollout 收进一个 list,后面这些 rollout 会被分配到 iter1..iterN。
    # 〔讲解〕defaultdict(list) 让缺省值是空列表,直接 append 不用先判空。
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rec = parse_record(row)
        # 〔讲解〕用 rec["question"](题目文本)作分组键。
        groups[rec["question"]].append(rec)

    # 〔讲解〕n_iters = 所有题目里"rollout 条数最多"的那个数,决定一共要建几个 iter 目录。
    # 〔讲解〕(不同题可能 rollout 数不齐,取最大值保证目录够用。)
    n_iters = max(len(rs) for rs in groups.values())
    print(f"{len(groups)} unique questions, up to {n_iters} rollouts per question")

    # 〔讲解〕建输出根目录(parents=True 连父目录一起建,exist_ok=True 已存在不报错)。
    args.out.mkdir(parents=True, exist_ok=True)
    # 〔讲解〕预先建好 iter1..iterN 子目录(目录名从 1 计数,故用 i+1)。
    for i in range(n_iters):
        (args.out / f"iter{i + 1}").mkdir(exist_ok=True)

    written = 0
    # 〔讲解〕逐题落盘:每道题的多条 rollout 按顺序铺到 iter1、iter2... 同一 qid 文件名贯穿各 iter 目录。
    for q, rollouts in groups.items():
        # 〔讲解〕用该题第一条 rollout 的 instance 算出稳定文件名 id;同题各 iter 共用这个 qid。
        qid = question_id(q, rollouts[0].get("instance"))
        for i, rec in enumerate(rollouts):
            # 〔讲解〕目标路径:第 i 条 rollout 写进 iter{i+1}/<qid>.json。
            target = args.out / f"iter{i + 1}" / f"{qid}.json"
            with target.open("w") as f:
                # 〔讲解〕ensure_ascii=False:保留中文/非 ASCII 原字符,不转成 \uXXXX 转义,便于人读。
                json.dump(rec, f, ensure_ascii=False)
            written += 1

    print(f"Wrote {written} JSON files into {args.out}/iter1..iter{n_iters}")


if __name__ == "__main__":
    main()
