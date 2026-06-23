#!/usr/bin/env python3
# 〔讲解〕本文件属于 AggAgent 流水线的 scripts/(数据准备)环节,是一个命令行小工具。
# 〔讲解〕背景:rollout 阶段会对每道题跑 N 次 ReAct 搜索,把每次(iter)的结果各存一个 json,散落在 iter1/ iter2/ ... 子目录里。
# 〔讲解〕本工具的活儿:把"同一道题"散在各 iter 目录里的 N 个轨迹文件,挑出来收拢到一个扁平目录中。
# 〔讲解〕下游:收拢好的目录正好喂给聚合环节(/aggagent),让聚合智能体读这 N 条轨迹去合成一份答案。
"""
collect_trajs.py: Collect N trajectory files for a single problem (or all problems) into flat directories.

The rollout pipeline stores results as:
  <rollout_dir>/
    iter1/  run_<ts>_<hash>.json   (one file per problem)
    iter2/  run_<ts>_<hash>.json
    ...

Usage:
  # List all available questions
  python scripts/collect_trajs.py output/rollout/GLM-4.7-Flash/deepsearchqa --list

  # Collect trajectories for a question matching "MMORPG"
  python scripts/collect_trajs.py output/rollout/GLM-4.7-Flash/deepsearchqa "MMORPG"

  # Collect ALL problems (one subdirectory per problem)
  python scripts/collect_trajs.py output/rollout/GLM-4.7-Flash/deepsearchqa --all

  # Specify output directory explicitly
  python scripts/collect_trajs.py output/rollout/GLM-4.7-Flash/deepsearchqa "MMORPG" --out /tmp/mmorpg_trajs
  python scripts/collect_trajs.py output/rollout/GLM-4.7-Flash/deepsearchqa --all --out /tmp/all_trajs
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path


# 〔讲解〕职责:扫描 rollout 父目录,把它下面的每一个子目录(即 iter1/ iter2/ ...)收集成一个排好序的列表返回。
# 〔讲解〕入参 rollout_dir = 父目录;返回 = 子目录 Path 列表。若一个子目录都没有,直接报错退出(数据有问题)。
def find_iter_dirs(rollout_dir: Path) -> list[Path]:
    """Return sorted iter* leaf directories under rollout_dir."""
    dirs = []
    # 〔讲解〕sorted 保证 iter1 < iter2 < ... 的稳定顺序;只收目录,忽略散落的文件。
    for entry in sorted(rollout_dir.iterdir()):
        if entry.is_dir():
            dirs.append(entry)
    if not dirs:
        print(f"No subdirectories found in {rollout_dir}", file=sys.stderr)
        sys.exit(1)
    return dirs


# 〔讲解〕职责:打开一个轨迹 json,提取出这条轨迹对应的"题目文本"作为它的身份标识。
# 〔讲解〕返回该题文本;后续就是靠这个文本来判断"哪些文件属于同一道题"。
def load_question(filepath: Path) -> str:
    with open(filepath) as f:
        data = json.load(f)
    # 〔讲解〕容错取键:优先取 question 字段;没有就退而取 instance.id;再没有就返回空串(视为无法识别)。
    return data.get("question") or data.get("instance", {}).get("id") or ""


# 〔讲解〕职责:给单个 iter 目录建索引,返回 {题目文本: 文件路径} 字典。
# 〔讲解〕作用是把"这个 iter 里有哪些题、各自存在哪个文件"做成可快速查找的表。
def index_dir(dir_path: Path) -> dict[str, Path]:
    """Return {question: filepath} for all JSON files in a directory."""
    index = {}
    for f in sorted(dir_path.glob("*.json")):
        q = load_question(f)
        # 〔讲解〕只收能识别出题目的文件;空串(无法识别)直接跳过,不进索引。
        if q:
            index[q] = f
    return index


# 〔讲解〕职责:在某个 iter 的索引里,按用户给的关键词(子串、大小写不敏感)模糊找出唯一匹配的那道题。
# 〔讲解〕返回完整题目文本;0 个匹配返回 None;多个匹配则报歧义并退出(强制用户把关键词写精确)。
def find_matching_question(index: dict[str, Path], query: str) -> str | None:
    """Find the question that contains query (case-insensitive). Returns the full question."""
    query_lower = query.lower()
    # 〔讲解〕核心匹配:把题目和关键词都转小写,用 in 做子串包含判断,收集所有命中项。
    matches = [q for q in index if query_lower in q.lower()]
    if len(matches) == 1:
        return matches[0]
    # 〔讲解〕命中多于一个 = 关键词太宽泛,无法确定指哪道题,打印候选并退出让用户改精确。
    if len(matches) > 1:
        print(f"Ambiguous query '{query}' matches {len(matches)} questions:", file=sys.stderr)
        for m in matches:
            print(f"  {m[:100]}", file=sys.stderr)
        print("Refine your query to match exactly one.", file=sys.stderr)
        sys.exit(1)
    return None


# 〔讲解〕职责(本工具的核心动作):对"同一道题",从每个 iter 目录各取一个对应文件,统一拷进 out_dir。
# 〔讲解〕入参:question=目标题目文本, iter_dirs=所有 iter 目录, out_dir=收拢目标目录;返回=实际拷贝的文件数。
def collect_one(
    question: str,
    iter_dirs: list[Path],
    out_dir: Path,
) -> int:
    """Copy one file per iter for `question` into out_dir. Returns number of files copied."""
    # 〔讲解〕parents=True 递归建父目录, exist_ok=True 已存在不报错。
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    n_copied = 0
    # 〔讲解〕逐个 iter:先给该 iter 建索引,再在里面找这道题对应的文件。
    for iter_dir in iter_dirs:
        index = index_dir(iter_dir)
        matched = find_matching_question(index, question)
        # 〔讲解〕某些 iter 可能跑挂了/没这道题 → 记下缺失的 iter 名,跳过即可,不中断整体收集。
        if matched is None:
            missing.append(iter_dir.name)
            continue
        src = index[matched]
        # 〔讲解〕目标文件名加 iter 前缀(如 iter1_run_xxx.json),避免不同 iter 同名文件互相覆盖,也便于溯源。
        dst = out_dir / f"{iter_dir.name}_{src.name}"
        # 〔讲解〕copy2 复制内容并保留元数据(修改时间等),原文件不动。
        shutil.copy2(src, dst)
        n_copied += 1
    if missing:
        print(f"  Warning: not found in {missing}", file=sys.stderr)
    return n_copied


# 〔讲解〕职责:命令行入口。解析参数,按三种模式分流:--list 列题 / --all 全量收集 / 单题查询收集。
def main():
    parser = argparse.ArgumentParser(
        description="Collect per-problem trajectories from iter* rollout directories."
    )
    parser.add_argument("rollout_dir", help="Parent rollout directory (contains iter1/, iter2/, ...)")
    # 〔讲解〕query 用 nargs="?" 设为可选位置参数:--list/--all 模式下可以不给关键词。
    parser.add_argument("query", nargs="?", help="Question substring to match (case-insensitive)")
    parser.add_argument("--out", help="Output directory (leaf dir for single query; parent dir for --all)")
    parser.add_argument("--list", action="store_true", help="List all available questions and exit")
    parser.add_argument("--all", action="store_true", help="Collect all problems (one subdir per problem)")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    if not rollout_dir.is_dir():
        print(f"Error: {rollout_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    iter_dirs = find_iter_dirs(rollout_dir)
    print(f"Found {len(iter_dirs)} iter directories: {[d.name for d in iter_dirs]}")

    # 〔讲解〕约定用第一个 iter 目录作为"题目全集"的基准:列题、遍历全量题都以它为准。
    first_index = index_dir(iter_dirs[0])

    # 〔讲解〕模式一 --list:只把基准 iter 里的所有题目编号打印出来,方便用户挑关键词,然后直接 return。
    if args.list:
        print(f"\nAvailable questions ({len(first_index)} in {iter_dirs[0].name}):")
        # 〔讲解〕enumerate(..., 1) 从 1 开始编号;q[:120] 截断长题目避免刷屏。
        for i, q in enumerate(sorted(first_index), 1):
            print(f"  [{i:3d}] {q[:120]}")
        return

    # --all: collect every problem into its own subdirectory
    # 〔讲解〕模式二 --all:对基准 iter 里的每一道题都各建一个子目录并收集,实现一次性全量整理。
    if args.all:
        questions = sorted(first_index.keys())
        # 〔讲解〕未显式给 --out 时,默认输出到 rollout 父目录旁的 trajs_all/。
        base_out = Path(args.out) if args.out else rollout_dir.parent / "trajs_all"
        print(f"\nCollecting {len(questions)} problems → {base_out}/")
        total = 0
        for i, question in enumerate(questions):
            # 〔讲解〕slug = 由题目生成的目录名片段:转小写、空格换下划线、截前 40 字符。
            slug = question.lower().replace(" ", "_")[:40]
            # 〔讲解〕再做一遍清洗:只保留字母数字和下划线,剔除会让目录名出错的特殊字符。
            slug = "".join(c if c.isalnum() or c == "_" else "" for c in slug)
            # 〔讲解〕每题一个子目录 problem_000_xxx,编号补零对齐便于排序浏览。
            prob_dir = base_out / f"problem_{i:03d}_{slug}"
            n = collect_one(question, iter_dirs, prob_dir)
            print(f"  [{i:3d}] {question[:80]} → {prob_dir.name}/ ({n} trajs)")
            total += n
        print(f"\nDone. {len(questions)} problems, {total} files → {base_out}")
        # 〔讲解〕贴心提示:整理完直接告诉用户怎么对其中一题跑聚合(下游 /aggagent 命令)。
        print(f"\nRun aggregation on one problem with:")
        print(f"  /aggagent {base_out}/problem_000_...")
        return

    # Single query mode
    # 〔讲解〕模式三(默认):单题模式。既不是 --list 也不是 --all,就必须给关键词,否则报错退出。
    if not args.query:
        print("Error: provide a query string, --list, or --all", file=sys.stderr)
        sys.exit(1)

    matched_question = find_matching_question(first_index, args.query)
    if matched_question is None:
        print(f"No question matching '{args.query}' found in {iter_dirs[0].name}", file=sys.stderr)
        print("Use --list to see all available questions.", file=sys.stderr)
        sys.exit(1)

    print(f"\nMatched: {matched_question[:120]}")

    # 〔讲解〕单题模式的默认输出名:用关键词(而非完整题目)生成 slug,前 30 字符。
    slug = args.query.lower().replace(" ", "_")[:30]
    out_dir = Path(args.out) if args.out else rollout_dir.parent / f"trajs_{slug}"

    n = collect_one(matched_question, iter_dirs, out_dir)
    # 〔讲解〕一个文件都没收到 = 这道题在所有 iter 都缺,属异常,报错退出。
    if n == 0:
        print("Error: no files collected.", file=sys.stderr)
        sys.exit(1)

    print(f"\nCollected {n} trajectories → {out_dir}")
    print(f"\nRun aggregation with:")
    print(f"  /aggagent {out_dir}")


if __name__ == "__main__":
    main()
