#!/usr/bin/env python3
# 〔讲解〕本文件是 aggagent(聚合智能体)环节的一个【命令行小工具】。
# 〔讲解〕在 AggAgent 两阶段 MoA 流程里:rollout/ 先跑出多条 ReAct 搜索轨迹(每条存成一个 .json),
# 〔讲解〕aggagent 这个"会调工具的聚合智能体"需要去读这些轨迹再融合成一份答案。本脚本就是它能调用的工具之一:
# 〔讲解〕给定一个存放轨迹 json 的目录,把每条轨迹的【最终答案(最后一条消息)】和【元数据(步数/token/用过哪些工具)】打印出来。
# 〔讲解〕上游=rollout 产出的 trajectory json;下游=聚合智能体读这里打印的 stdout 来决策。
"""
get_solution: Retrieve final content from trajectory JSON files.

Usage:
  python get_solution.py <trajectories_dir>              # all trajectories + metadata
  python get_solution.py <trajectories_dir> <traj_id>   # single trajectory (1-based)
"""
import json
import os
import re
import sys


# 〔讲解〕从一条 message(对话消息字典)里抽出纯文本内容。
# 〔讲解〕职责:屏蔽消息 content 的多种格式差异 —— 有时是纯字符串,有时是 [{"text": ...}] 这种富文本列表。
# 〔讲解〕入参 message=消息字典,key=要取哪个字段(默认 "content")。返回:抽取到的文本字符串(取不到则空串)。
def get_content(message, key="content"):
    """Extract text content from a message dict (handles string or list formats)."""
    value = message.get(key, "")
    # 〔讲解〕情况一:content 本身就是字符串,直接返回。
    if isinstance(value, str):
        return value
    # 〔讲解〕情况二:content 是非空列表(OpenAI 富文本格式),取第一个 block 的 text 字段。
    if isinstance(value, list) and value:
        if isinstance(value[0], dict):
            # 〔讲解〕`or ""` 防御:text 为 None 时退化成空串,避免后续拼接报错。
            text = value[0].get("text") or ""
            # 〔讲解〕只有在取主 content 时才给文本加"工具调用/工具响应"前缀标签,方便聚合智能体看懂这条消息的来历。
            if key == "content":
                name = message.get("name")            # 〔讲解〕name 存在 → 这是某工具返回的结果。
                recipient = message.get("recipient")  # 〔讲解〕recipient 存在 → 这是发往某工具的调用。
                if name:
                    text = f"[Tool Response: {name}]\n{text}"
                elif recipient:
                    text = f"[Tool Call: {recipient}]\n{text}"
            return text
    return ""


# 〔讲解〕粗略估算一批消息的 token 数(不调真分词器,用"字符数 / 每 token 字符数"近似)。
# 〔讲解〕作用:给聚合智能体一个"这条轨迹大概多长"的量级感,用于元数据展示。
# 〔讲解〕chars_per_token=4.0 是英文场景常用经验值(约 4 字符 ≈ 1 token);返回整数 token 估值。
def count_tokens_approx(messages, chars_per_token=4.0):
    total_chars = 0
    for msg in messages:
        # 〔讲解〕把 role/思维链(reasoning)/正文 content 这几个文本字段的字符都累加进来。
        for key in ["role", "reasoning_content", "reasoning", "content"]:
            value = msg.get(key)
            if isinstance(value, str):
                total_chars += len(value)
        # 〔讲解〕工具调用单独算:tool_calls 是结构化 JSON,序列化后按其字符长度计入(ensure_ascii=False 保中文按 1 字符算,更贴近真实)。
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None:
            total_chars += len(json.dumps(tool_calls, ensure_ascii=False))
    return int(total_chars / chars_per_token)


# 〔讲解〕把所有轨迹的元数据汇总成一段可读文本(每条一行:步数、估算 token、用过的工具及次数)。
# 〔讲解〕入参 trajectories=轨迹列表(每条是一串 message);返回:多行字符串。注释里说明它和 tools.format_metadata 保持一致。
def format_metadata(trajectories):
    """Format trajectory metadata — mirrors tools.format_metadata."""
    blocks = []
    for i, traj in enumerate(trajectories):
        num_steps = len(traj)                      # 〔讲解〕步数 = 这条轨迹里消息的条数。
        approx_tokens = count_tokens_approx(traj)   # 〔讲解〕复用上面的估算函数。

        # 〔讲解〕统计这条轨迹各工具被调用了几次,key=工具名,value=次数。
        tool_counts: dict = {}
        for msg in traj:
            # 〔讲解〕只看 assistant 发出且带 tool_calls 的消息 —— 即"模型主动调工具"这一步。
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tc = msg["tool_calls"][0]          # 〔讲解〕只取第一个工具调用(本框架一步通常只调一个工具)。
                func = tc.get("function", {})
                if isinstance(func, dict):
                    name = func.get("name")
                    if name:
                        # 〔讲解〕字典累加计数的惯用写法:取旧值(没有则 0)再 +1。
                        tool_counts[name] = tool_counts.get(name, 0) + 1

        # 〔讲解〕把计数排序后拼成 "search×3, browse×1" 这种串;一个工具都没用就显示 "none"。
        tool_str = ", ".join(f"{n}×{c}" for n, c in sorted(tool_counts.items())) if tool_counts else "none"
        # 〔讲解〕`{approx_tokens:,}` 给数字加千分位逗号,更易读;`i + 1` 是因为对外用 1-based 编号。
        blocks.append(f"Trajectory {i + 1}: {num_steps} steps, ~{approx_tokens:,} tokens | tools: {tool_str}")

    return "\n".join(blocks)


# 〔讲解〕从目录里加载所有轨迹 json。职责:扫目录 → 逐个读 json → 拆出 messages(对话)和 question(问题)。
# 〔讲解〕入参 traj_dir=轨迹目录路径;返回二元组 (trajectories, questions),两者下标一一对应。
def load_trajectories(traj_dir):
    # 〔讲解〕只挑目录下以 .json 结尾的文件并排序,保证每次读取顺序稳定(对应稳定的 1-based 编号)。
    files = sorted([
        os.path.join(traj_dir, f)
        for f in os.listdir(traj_dir)
        if f.endswith(".json")
    ])
    # 〔讲解〕目录里没有任何 json 属于异常 —— 打到 stderr 并以非零码退出,让调用方(聚合智能体/脚本)感知失败。
    if not files:
        print(f"No JSON files found in {traj_dir}", file=sys.stderr)
        sys.exit(1)

    trajectories = []
    questions = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        # 〔讲解〕轨迹 json 的约定结构:messages=对话列表,question=原始问题;用 .get 给默认值防字段缺失。
        trajectories.append(data.get("messages", []))
        questions.append(data.get("question", ""))
    return trajectories, questions


# 〔讲解〕命令行入口。两种用法:只给目录=打印全部轨迹的元数据+答案;再给一个编号=只打印那一条轨迹的完整答案。
# 〔讲解〕这是聚合智能体作为"工具"实际执行的命令所触发的逻辑。
def main():
    # 〔讲解〕参数校验:至少要有目录参数(argv[0] 是脚本名,argv[1] 才是目录)。
    if len(sys.argv) < 2:
        print("Usage: get_solution.py <trajectories_dir> [trajectory_id]")
        sys.exit(1)

    traj_dir = sys.argv[1]
    # 〔讲解〕可选的第二个参数=要单独查看的轨迹编号(1-based);没给则为 None,表示"看全部"。
    trajectory_id = int(sys.argv[2]) if len(sys.argv) > 2 else None

    trajectories, questions = load_trajectories(traj_dir)
    n = len(trajectories)

    # 〔讲解〕分支一:未指定编号 → 输出"元数据总览 + 每条轨迹的最终答案"。
    if trajectory_id is None:
        # Print metadata header + all solutions
        # 〔讲解〕取第一个非空的 question 作为本批问题(同一批轨迹通常对应同一个问题)。
        question = next((q for q in questions if q), "")
        print(f"TRAJECTORY METADATA ({n} trajectories):")
        if question:
            print(f"Question: {question}")
        print()
        print(format_metadata(trajectories))
        print()
        print("=" * 60)
        print("SOLUTIONS")
        print("=" * 60)
        for i, traj in enumerate(trajectories):
            # 〔讲解〕核心:一条轨迹的"答案"=它最后一条消息 traj[-1] 的文本内容;空轨迹则给空串。
            content = get_content(traj[-1]) if traj else ""
            print(f"\n--- Trajectory {i + 1} ---")
            # 〔讲解〕答案过长(>2000 字符)就截断 + 标注 [... truncated],避免一次性灌爆聚合智能体的上下文。
            if len(content) > 2000:
                print(content[:2000] + "\n[... truncated]")
            else:
                print(content)
    # 〔讲解〕分支二:指定了编号 → 只看那一条,且不截断(给全文)。
    else:
        # 〔讲解〕编号必须落在 1..n 区间内,越界就报错退出。
        if trajectory_id < 1 or trajectory_id > n:
            print(f"Error: trajectory_id must be 1-{n}")
            sys.exit(1)
        # 〔讲解〕1-based 编号转 0-based 下标取出对应轨迹。
        traj = trajectories[trajectory_id - 1]
        content = get_content(traj[-1]) if traj else ""
        print(f"Trajectory {trajectory_id} (total steps: {len(traj)}):")
        print(content)


if __name__ == "__main__":
    main()
