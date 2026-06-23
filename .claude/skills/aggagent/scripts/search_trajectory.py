#!/usr/bin/env python3
"""
search_trajectory: ROUGE-L search through a trajectory's steps.

Usage:
  python search_trajectory.py <trajectories_dir> <trajectory_id> "<query>" [--role tool|assistant] [--k N]

trajectory_id is 1-based.
--role tool      restricts search to actual tool responses (ground truth observations)
--role assistant restricts search to agent reasoning steps
--k N            number of results to return (default 5, max 10)
"""
#〔讲解〕本文件是 aggagent/scripts/ 下的命令行小工具,服务于「聚合」环节。
#〔讲解〕一条 ReAct trajectory(rollout 阶段跑出来的搜索轨迹)往往几十步、上万 token,聚合智能体不可能整条读完。
#〔讲解〕本工具用 ROUGE-L 文本相似度在一条轨迹的所有 step 里检索 query,返回最相关的几步,等价于给 agent 一个"轨迹内全文搜索"。
#〔讲解〕上游:rollout 阶段产出的 trajectory JSON(每个文件含 messages 列表);下游:aggagent 聚合智能体把本工具当 tool 调用,据此定位证据再写答案。
#〔讲解〕注:文件内多处注释说"mirrors XXX",即本独立脚本是在复刻 aggagent 框架里某个 Tool 类的逻辑,保证 CLI 与框架内表现一致。
import argparse
import json
import os
import re
import sys

#〔讲解〕优先尝试用工业级的 rouge_score 库算 ROUGE-L;装不上则在 except 里退化成标准库 difflib(见下)。这种"软依赖"写法让脚本在缺包环境也能跑。
try:
    from rouge_score import rouge_scorer as _rouge_scorer
    #〔讲解〕创建一个只算 rougeL 的打分器;use_stemmer=False 表示不做词干还原(如 running→run),按原词面匹配,保证可复现、不引入额外语言学处理。
    _scorer = _rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    #〔讲解〕核心相似度函数:返回 query 相对 text 的 ROUGE-L recall(召回)。
    #〔讲解〕选 recall 而非 f1:衡量"query 里的词有多少能在 text 中找到",即"这段文本是否覆盖了我要找的内容",正合检索语义。
    def rouge_l_recall(query, text):
        if not query or not text:
            return 0.0
        #〔讲解〕score(query, text) 把 query 当 reference、text 当候选;取 ['rougeL'].recall 这一标量作为相关度分。
        return _scorer.score(query, text)['rougeL'].recall

except ImportError:
    import difflib

    #〔讲解〕降级实现:rouge_score 没装时用标准库 difflib 顶替,签名/返回值与上面完全一致,调用方无感知。
    def rouge_l_recall(query, text):
        """Fallback when rouge_score is not installed."""
        if not query or not text:
            return 0.0
        #〔讲解〕SequenceMatcher.ratio() 给出两串的整体相似比(0~1);先 lower() 做大小写无关匹配。它不是真 ROUGE-L,但同为"越像分越高",够用作兜底。
        return difflib.SequenceMatcher(None, query.lower(), text.lower()).ratio()


#〔讲解〕从一条 message(轨迹里的一步,dict 结构)中抽取可读文本。
#〔讲解〕难点在于 content 字段格式不统一:可能是纯字符串,也可能是 OpenAI 风格的"内容块列表"[{type,text},...],本函数把两种都归一成一段文本。
#〔讲解〕入参 key 决定抽哪个字段(默认 "content";调用方也会传 "reasoning_content" 抽思考过程)。
def get_content(message, key="content"):
    """Extract text content from a message dict (handles string or list formats)."""
    value = message.get(key, "")
    #〔讲解〕情况一:字段本身就是字符串,直接返回。
    if isinstance(value, str):
        return value
    #〔讲解〕情况二:字段是非空列表(内容块格式),只取第一个块的 text。
    if isinstance(value, list) and value:
        if isinstance(value[0], dict):
            #〔讲解〕or "" 防御:块里没有 text 键或值为 None 时,回退成空串而非报错。
            text = value[0].get("text") or ""
            #〔讲解〕仅当抽的是主内容(content)时,给文本加上来源前缀标签,让后续打印/检索能看出这步是"工具响应"还是"工具调用"。
            if key == "content":
                name = message.get("name")
                recipient = message.get("recipient")
                #〔讲解〕有 name → 这是某工具返回的观测结果,打 [Tool Response: 工具名] 前缀。
                if name:
                    text = f"[Tool Response: {name}]\n{text}"
                #〔讲解〕否则有 recipient → 这是发给某工具的调用,打 [Tool Call: 目标] 前缀。
                elif recipient:
                    text = f"[Tool Call: {recipient}]\n{text}"
            return text
    #〔讲解〕兜底:格式不认识(如 None、空列表、列表首元素不是 dict)一律返回空串,保证调用处不会崩。
    return ""


#〔讲解〕把过长文本截断到前 max_words 个"词",末尾补省略标记。
#〔讲解〕目的:检索结果只给 agent 看摘要,避免一步几千词把上下文撑爆。docstring 说明它复刻框架里 tools.truncate_text 的行为。
def truncate_text(text, max_words=150):
    """Truncate text to first N words — mirrors tools.truncate_text."""
    if not text:
        return ""
    count = 0
    #〔讲解〕\S+ 匹配每一段非空白字符(即一个"词");逐个数词,数到第 max_words 个就在该词结束位置(m.end())切断。
    for m in re.finditer(r'\S+', text):
        count += 1
        if count == max_words:
            #〔讲解〕text[:m.end()] 保留到第 max_words 个词为止,再拼接截断提示。
            return text[:m.end()] + '\n[... truncated]'
    #〔讲解〕循环正常走完说明总词数 < max_words,无需截断,原样返回。
    return text


#〔讲解〕加载一个目录下的所有轨迹文件。
#〔讲解〕返回值是 list[list[message]]:外层每个元素对应一条轨迹,内层是该轨迹的 messages(步骤)列表。
#〔讲解〕sorted 保证文件顺序稳定,从而"第几条轨迹"的编号在多次运行间一致(下游用 1-based id 选轨迹靠的就是这个顺序)。
def load_trajectories(traj_dir):
    files = sorted([
        os.path.join(traj_dir, f)
        for f in os.listdir(traj_dir)
        if f.endswith(".json")
    ])
    #〔讲解〕目录里没有任何 .json 直接报错退出,避免后面拿空列表做无意义检索。
    if not files:
        print(f"No JSON files found in {traj_dir}", file=sys.stderr)
        sys.exit(1)
    trajectories = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        #〔讲解〕每个 JSON 取出 "messages" 字段(轨迹步骤);缺失则用空列表兜底,不让单个坏文件中断整批加载。
        trajectories.append(data.get("messages", []))
    return trajectories


#〔讲解〕主流程:解析命令行 → 选定一条轨迹 → 给每一步打 ROUGE-L 分 → 排序取 Top-k → 打印。
def main():
    #〔讲解〕argparse 定义 CLI 接口:三个位置参数(轨迹目录、轨迹编号、查询词)+ 两个可选参数(--role 过滤角色、--k 返回条数)。
    parser = argparse.ArgumentParser(description="ROUGE-L search through trajectory steps")
    parser.add_argument("traj_dir", help="Directory containing trajectory JSON files")
    parser.add_argument("trajectory_id", type=int, help="Trajectory index (1-based)")
    parser.add_argument("query", help="Search term or phrase")
    #〔讲解〕--role 只允许 tool / assistant:tool=只搜工具真实返回(客观观测,即"ground truth"),assistant=只搜 agent 自己的推理步。默认 None=全搜。
    parser.add_argument("--role", choices=["tool", "assistant"], default=None,
                        help="Filter to 'tool' (environment observations) or 'assistant' steps only")
    parser.add_argument("--k", type=int, default=5, help="Max matches to return (default 5, max 10)")
    args = parser.parse_args()

    trajectories = load_trajectories(args.traj_dir)
    n = len(trajectories)

    #〔讲解〕轨迹编号是 1-based 且需落在 [1, n];越界直接报错退出,防止下一行索引出错。
    if args.trajectory_id < 1 or args.trajectory_id > n:
        print(f"Error: trajectory_id must be 1-{n}")
        sys.exit(1)

    #〔讲解〕用户给的是 1-based 编号,这里 -1 转成 Python 的 0-based 下标,取出目标轨迹。
    traj = trajectories[args.trajectory_id - 1]
    #〔讲解〕硬上限:即使用户传了很大的 --k,也最多返回 10 条,防止刷屏 / 撑爆 agent 上下文。
    max_results = min(args.k, 10)

    # Score each step — mirrors SearchTrajectoriesTool._score_traj
    #〔讲解〕逐步打分。这段是检索核心,复刻框架内 SearchTrajectoriesTool._score_traj 的算分逻辑。
    scored = []
    for step_idx, step in enumerate(traj):
        #〔讲解〕若指定了 --role 且当前步角色不符,跳过(role 缺失时用 "" 兜底,自然不匹配 tool/assistant)。
        if args.role is not None and step.get("role", "") != args.role:
            continue
        #〔讲解〕分别抽出这一步的三路文本:主内容。
        content = get_content(step) or ""
        #〔讲解〕推理过程:不同模型字段名不一(reasoning_content / reasoning),用 or 依次回退取到哪个算哪个。
        reasoning = get_content(step, "reasoning_content") or get_content(step, "reasoning") or ""
        #〔讲解〕工具调用结构(函数名+参数):序列化成 JSON 字符串好做文本匹配;ensure_ascii=False 保留中文等原字符;无 tool_calls 则置空串。
        tool_calls_str = json.dumps(step.get("tool_calls"), ensure_ascii=False) if step.get("tool_calls") else ""

        #〔讲解〕一步的最终得分 = 三路文本各自与 query 的 ROUGE-L recall 取最大值。
        #〔讲解〕含义:只要 query 命中了内容、推理、或工具调用三者之一,这步就算相关,不要求三处都中。
        score = max(
            rouge_l_recall(args.query, content),
            rouge_l_recall(args.query, reasoning),
            rouge_l_recall(args.query, tool_calls_str),
        )
        #〔讲解〕只保留有正相关的步(score>0),连同原始下标和整步数据存起来,供后续排序与打印。
        if score > 0:
            scored.append((score, step_idx, step))

    #〔讲解〕按分数降序排(-x[0] 实现从高到低),分高的相关步排在前面。
    scored.sort(key=lambda x: -x[0])

    #〔讲解〕一条都没命中:给出友好提示(带上 role 过滤说明)后直接返回,不打印结果表。
    if not scored:
        role_msg = f" (role={args.role})" if args.role else ""
        print(f"No matches found for '{args.query}'{role_msg} in trajectory {args.trajectory_id}")
        return

    #〔讲解〕实际展示条数 = min(上限, 命中数),避免命中不足 k 条时打印越界。
    n_shown = min(max_results, len(scored))
    print(f"Top {n_shown} matches for '{args.query}' in trajectory {args.trajectory_id}"
          + (f" [role={args.role}]" if args.role else "") + ":")
    print()

    #〔讲解〕遍历 Top-k 命中步,逐条格式化打印(角色、分数、推理、首个工具调用、内容)。
    for score, step_idx, step in scored[:max_results]:
        role = step.get("role", "")
        content = get_content(step)
        reasoning = get_content(step, "reasoning_content") or get_content(step, "reasoning")
        tool_calls = step.get("tool_calls")

        #〔讲解〕step_idx+1 把 0-based 下标转回 1-based 步号给人看;score 保留三位小数。
        print(f"  Step {step_idx + 1} [{role}]  score={score:.3f}")
        if reasoning:
            print(f"  [reasoning] {truncate_text(reasoning)}")
        #〔讲解〕tool_calls[:1] 只打印第一个工具调用(一步可能并发多个调用),避免输出过长。
        if tool_calls:
            print(f"  [tool_calls] {json.dumps(tool_calls[:1], ensure_ascii=False)}")
        if content:
            print(f"  {truncate_text(content)}")
        print()


if __name__ == "__main__":
    main()
