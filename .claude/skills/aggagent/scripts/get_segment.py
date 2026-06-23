#!/usr/bin/env python3
#〔讲解〕本文件 = 一个命令行小工具,用来「截取并打印一条 ReAct 轨迹里连续若干步(最多 5 步)的内容」。
#〔讲解〕在 AggAgent pipeline 里属于辅助/调试环节:rollout 阶段会把每条搜索轨迹存成 JSON(里面是 messages 列表);
#〔讲解〕本脚本就是给人(或聚合智能体)按 trajectory_id + 步号区间「翻看某几步在干什么」。
#〔讲解〕它刻意复刻了 aggagent 工具 GetSegmentTool 的截断/裁剪逻辑(见下文多处 "mirrors ..."),
#〔讲解〕这样命令行看到的内容,与聚合智能体真正读到的内容保持一致。上游=rollout 产出的轨迹目录;下游=人工排查或对照工具行为。
"""
get_segment: Read a contiguous range of steps from a trajectory (max 5).

Usage:
  python get_segment.py <trajectories_dir> <trajectory_id> <start_step> <end_step>

trajectory_id is 1-based. start_step and end_step are 1-based step indices.
end_step - start_step must be <= 4 (enforced automatically).
"""
import json
import os
import re
import sys


#〔讲解〕从一条「消息(message)」里把文本内容抽出来。轨迹里每一步就是一个 message dict。
#〔讲解〕入参 message=单步字典; key=要取哪个字段(默认 "content",也会被复用来取 reasoning)。返回值=纯文本字符串。
#〔讲解〕难点在于:同一字段在不同模型输出里可能是「纯字符串」也可能是「list of dict」(OpenAI 风格的多段内容),要兼容两种格式。
def get_content(message, key="content"):
    """Extract text content from a message dict (handles string or list formats)."""
    value = message.get(key, "")
    #〔讲解〕情况一:字段直接就是字符串,原样返回。
    if isinstance(value, str):
        return value
    #〔讲解〕情况二:字段是非空列表(多段内容块),这里只取第 0 块——简化处理,假设主要文本在第一段。
    if isinstance(value, list) and value:
        if isinstance(value[0], dict):
            #〔讲解〕从内容块里取 "text" 字段;`or ""` 防止 text 为 None 时后面字符串拼接报错。
            text = value[0].get("text") or ""
            #〔讲解〕仅当抽的是正文 content 时,才给文本加「来源标签」,让人一眼看出这步是工具返回还是工具调用。
            if key == "content":
                #〔讲解〕name 存在 => 这是某工具的「返回结果」;recipient 存在 => 这是发往某工具的「调用」。二者用于区分 ReAct 里的 action/observation。
                name = message.get("name")
                recipient = message.get("recipient")
                if name:
                    #〔讲解〕给工具响应加前缀 "[Tool Response: 工具名]",便于人工阅读时辨认。
                    text = f"[Tool Response: {name}]\n{text}"
                elif recipient:
                    #〔讲解〕给工具调用加前缀 "[Tool Call: 接收方]"。
                    text = f"[Tool Call: {recipient}]\n{text}"
            return text
    #〔讲解〕兜底:既不是字符串也不是可解析的列表(如空列表/None),返回空串,保证调用方拿到的总是 str。
    return ""


#〔讲解〕把长文本截断到前 N 个「词」,超出部分用省略标记替代。作用:轨迹里某些步(尤其网页抓取结果)极长,打印前先压缩。
#〔讲解〕注释明说它「mirrors tools.truncate_text」——即与正式工具里的截断逻辑保持一致,保证调试视图≈智能体视图。max_words 默认 600。
def truncate_text(text, max_words=600):
    """Truncate text to first N words — mirrors tools.truncate_text."""
    if not text:
        return ""
    count = 0
    #〔讲解〕`\S+` 匹配「连续非空白字符」=一个词;用 finditer 逐个数词,而不是 split,这样能拿到每个词的结束位置 m.end()。
    for m in re.finditer(r'\S+', text):
        count += 1
        #〔讲解〕数到第 max_words 个词时,从原文截到该词结尾(保留完整词,不切半个单词),并贴上截断提示后立即返回。
        if count == max_words:
            return text[:m.end()] + '\n[... truncated]'
    #〔讲解〕循环走完都没到上限 => 文本本就不超长,整段原样返回。
    return text


#〔讲解〕加载一个目录下的所有轨迹文件。入参 traj_dir=轨迹目录;返回值=一个 list,每个元素是「某条轨迹的 messages 列表」。
#〔讲解〕约定:目录里每个 .json 文件 = 一条 rollout 轨迹;按文件名排序后,顺序即对应后面的 1-based trajectory_id。
def load_trajectories(traj_dir):
    #〔讲解〕列出目录里所有 .json 文件并排序,拼成绝对路径列表。排序很关键——决定了 trajectory_id 与文件的稳定对应关系。
    files = sorted([
        os.path.join(traj_dir, f)
        for f in os.listdir(traj_dir)
        if f.endswith(".json")
    ])
    #〔讲解〕目录里一个 json 都没有 => 报错并以退出码 1 终止(写到 stderr,符合命令行工具惯例)。
    if not files:
        print(f"No JSON files found in {traj_dir}", file=sys.stderr)
        sys.exit(1)
    trajectories = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        #〔讲解〕每个文件取出其 "messages" 字段(轨迹的核心:一步步的对话/动作记录);取不到则用空列表占位,避免后续 KeyError。
        trajectories.append(data.get("messages", []))
    return trajectories


#〔讲解〕程序主入口:解析命令行参数 -> 定位指定轨迹 -> 钳制步号区间 -> 逐步打印。无返回值,结果直接 print 到 stdout。
def main():
    #〔讲解〕参数校验:期望 4 个位置参数(脚本名算 argv[0],所以总长需 >=5);不足则打印用法并退出。
    if len(sys.argv) < 5:
        print("Usage: get_segment.py <trajectories_dir> <trajectory_id> <start_step> <end_step>")
        sys.exit(1)

    #〔讲解〕依次取四个参数:轨迹目录、轨迹编号、起始步、结束步;后三者转 int(命令行传进来都是字符串)。
    traj_dir = sys.argv[1]
    trajectory_id = int(sys.argv[2])
    start_step = int(sys.argv[3])
    end_step = int(sys.argv[4])

    trajectories = load_trajectories(traj_dir)
    n = len(trajectories)

    #〔讲解〕校验轨迹编号是否在 1..n 范围内(用户用的是 1-based 计数)。越界则报错退出。
    if trajectory_id < 1 or trajectory_id > n:
        print(f"Error: trajectory_id must be 1-{n}")
        sys.exit(1)

    #〔讲解〕把 1-based 的 trajectory_id 转成 0-based 的 list 下标取出目标轨迹;n_steps=这条轨迹一共有多少步。
    traj = trajectories[trajectory_id - 1]
    n_steps = len(traj)

    # Clamp and enforce max-5 window — mirrors GetSegmentTool.call
    #〔讲解〕以下四步「钳制」逻辑刻意复刻正式工具 GetSegmentTool.call,保证调试与线上行为一致:
    #〔讲解〕①把 start/end 夹到合法范围 [1, n_steps],防止用户传的步号越界(min 防超上界,max(1,..) 防小于 1)。
    start_step = max(1, min(start_step, n_steps))
    end_step = max(1, min(end_step, n_steps))
    #〔讲解〕②若起点反而比终点大(用户写反了),就把起点拉回等于终点,退化成只看 1 步。
    if start_step > end_step:
        start_step = end_step
    #〔讲解〕③强制窗口最多 5 步:end-start>4 说明跨度超 5,直接把终点压到 start+4(即 start..start+4 共 5 步)。
    if end_step - start_step > 4:
        end_step = start_step + 4

    #〔讲解〕打印一行表头,告诉读者「看的是第几条轨迹的第几到第几步,总共多少步」。注意 –(en dash)是字符串里的连字符,非减号。
    print(f"Trajectory {trajectory_id}, steps {start_step}–{end_step} (of {n_steps} total):")
    print()

    #〔讲解〕遍历选中的步:range(start-1, end) 把 1-based 的闭区间 [start, end] 转成 0-based 的半开区间下标。
    for step_idx in range(start_step - 1, end_step):
        step = traj[step_idx]
        #〔讲解〕role=这步的角色(user/assistant/tool 等),用于表头展示。
        role = step.get("role", "")
        #〔讲解〕content=这步的正文(经 get_content 处理,可能已带 [Tool Call/Response] 前缀)。
        content = get_content(step)
        #〔讲解〕reasoning=模型的思维链。不同模型字段名不一(reasoning_content / reasoning),这里用 `or` 依次回退取到哪个算哪个。
        reasoning = get_content(step, "reasoning_content") or get_content(step, "reasoning")
        #〔讲解〕tool_calls=这步发起的工具调用(原始结构,通常是 list of dict),后面原样 JSON 打印。
        tool_calls = step.get("tool_calls")

        #〔讲解〕每步先打一行分隔头 "--- Step N [role] ---",step_idx+1 转回 1-based 给人看。
        print(f"--- Step {step_idx + 1} [{role}] ---")
        #〔讲解〕有思维链才打,并截断,避免刷屏。
        if reasoning:
            print(f"[reasoning]\n{truncate_text(reasoning)}")
        #〔讲解〕有工具调用才打;ensure_ascii=False 让中文等非 ASCII 字符正常显示,而非转成 \uXXXX。
        if tool_calls:
            print(f"[tool_calls] {json.dumps(tool_calls, ensure_ascii=False)}")
        #〔讲解〕有正文才打,同样截断。
        if content:
            print(truncate_text(content))
        print()


#〔讲解〕标准入口守卫:仅当本文件被直接运行(而非被 import)时才执行 main()。
if __name__ == "__main__":
    main()
