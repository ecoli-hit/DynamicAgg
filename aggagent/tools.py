# 〔讲解〕 本文件 = 「聚合智能体(aggagent)的工具箱」。
# 〔讲解〕 在 AggAgent 两阶段 MoA 流程里:rollout/ 先跑出多条 ReAct 搜索轨迹(trajectory),
# 〔讲解〕 然后聚合阶段把这些轨迹交给一个会「调工具」的聚合智能体读取、对比、再综合出一份答案。
# 〔讲解〕 本文件定义的就是这个聚合智能体能调用的工具:get_solution(读各轨迹最终答案)、
# 〔讲解〕 get_segment(读某轨迹连续若干步全文)、search_trajectory(在某轨迹里按关键词检索步骤)、
# 〔讲解〕 finish(提交最终聚合结果)。这些工具都遵循 qwen_agent 的 BaseTool 接口,通过 register_tool 注册后供 LLM function-calling 调用。
import re
import json
from typing import Optional, Union
# 〔讲解〕 qwen_agent 是阿里 Qwen 的智能体框架;BaseTool 是工具基类,register_tool 把工具注册进全局工具表,LLM 才能按名字调用。
from qwen_agent.tools.base import BaseTool, register_tool
# 〔讲解〕 rouge_score 提供 ROUGE-L 指标(基于最长公共子序列的文本重合度),后面用它给「检索词 vs 轨迹步骤文本」打相似度分。
from rouge_score import rouge_scorer as _rouge_scorer

# 〔讲解〕 预先建一个全局打分器,只算 rougeL,use_stemmer=False 表示不做词干还原(不把 running/run 当同词),保持精确匹配语义。
_scorer = _rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


# 〔讲解〕 工具函数:从一条「消息字典」里把纯文本内容抠出来。轨迹里每一步都是一个 message dict,
# 〔讲解〕 但 content 字段格式不统一(可能是字符串,也可能是 [{"text": ...}] 这种 OpenAI 多模态列表),本函数统一成字符串。
# 〔讲解〕 key 参数允许复用本函数去取别的字段(如 reasoning_content)。
def _get_content(message: dict, key: str = "content") -> str:
    """Extract text content from a message dict (handles string or list formats)."""
    value = message.get(key, "")
    # 〔讲解〕 情况一:已经是纯字符串,直接返回。
    if isinstance(value, str):
        return value
    # 〔讲解〕 情况二:是列表且非空(多模态格式),取第一个元素里的 text。
    if isinstance(value, list) and value:
        if isinstance(value[0], dict):
            # 〔讲解〕 .get("text") 可能返回 None,用 `or ""` 兜底成空串,避免后续字符串拼接报错。
            text = value[0].get("text") or ""
            # 〔讲解〕 仅当取的是主 content 时,给文本加上来源前缀,让聚合 LLM 一眼看出这步是「工具调用」还是「工具返回」。
            if key == "content":
                # 〔讲解〕 recipient = 这步调用了哪个工具;name = 这步是哪个工具的返回结果。
                recipient = message.get("recipient")
                name = message.get("name")
                # 〔讲解〕 有 name → 这是工具的「返回」,打 [Tool Response: xxx] 标签。
                if name:
                    text = f"[Tool Response: {name}]\n{text}"
                # 〔讲解〕 否则有 recipient → 这是模型发起的「工具调用」,打 [Tool Call: xxx] 标签。
                elif recipient:
                    text = f"[Tool Call: {recipient}]\n{text}"
            return text
    # 〔讲解〕 其它无法识别的格式一律返回空串,保证调用方拿到的永远是 str。
    return ""


# 〔讲解〕 工具函数:算「query 在 text 中的 ROUGE-L 召回率」。召回率 = query 里有多少内容能在 text 里找到,
# 〔讲解〕 越高说明这段 text 越覆盖了检索词,用作 search_trajectory 的相关性排序依据。
def _rouge_l_recall(query: str, text: str) -> float:
    """ROUGE-L recall of query against text."""
    # 〔讲解〕 任一为空则相似度为 0,避免空串送进打分器。
    if not query or not text:
        return 0.0
    # 〔讲解〕 score(target, prediction) 返回各指标对象,.recall 取召回率分量。这里 query 当 target、text 当 prediction。
    return _scorer.score(query, text)['rougeL'].recall


# 〔讲解〕 工具函数:把长文本截断到前 max_words 个「词」(\S+ 连续非空白即一词),防止单步内容过长撑爆 LLM 上下文。
def truncate_text(text: str, max_words: int = 150) -> str:
    """Truncate text to first n words."""
    if not text:
        return ""
    count = 0
    # 〔讲解〕 用正则迭代逐个匹配「词」,数到第 max_words 个就在该词结尾处切断并追加省略标记。
    for m in re.finditer(r'\S+', text):
        count += 1
        if count == max_words:
            # 〔讲解〕 m.end() 是这个词在原串里的结束下标,切到这里能保留完整单词,不会切半个词。
            return text[:m.end()] + '\n[... truncated]'
    # 〔讲解〕 词数本就不足 max_words,原样返回。
    return text


# 〔讲解〕 工具函数:粗估一条轨迹(消息列表)的 token 数。不调真分词器,用「总字符数 / 每 token 字符数」近似,
# 〔讲解〕 只为在 metadata 里给 LLM 一个量级感(这条轨迹大概多长),不要求精确。
def _count_tokens_approx(messages: list, chars_per_token: float = 4.0) -> int:
    """Approximate token count for a list of messages."""
    total_chars = 0
    for msg in messages:
        # 〔讲解〕 把这几类可能含文本的字段长度都累加进去(role/思维链/正文)。
        for key in ["role", "reasoning_content", "reasoning", "content"]:
            value = msg.get(key)
            if isinstance(value, str):
                total_chars += len(value)
        # 〔讲解〕 tool_calls 是结构化对象,序列化成 JSON 字符串再计长度;ensure_ascii=False 让中文按 1 字符算而非 \uXXXX。
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None:
            total_chars += len(json.dumps(tool_calls, ensure_ascii=False))
    # 〔讲解〕 英文经验值约 4 字符 ≈ 1 token,故除以 4.0 取整。
    return int(total_chars / chars_per_token)


# 〔讲解〕 工具函数:把「所有轨迹的概览信息」格式化成一段文本,塞进聚合智能体的 user prompt,
# 〔讲解〕 让它在调工具细看前先有个全局印象:每条轨迹多少步、约多少 token、用了哪些工具各几次。
def format_metadata(trajectories: list) -> str:
    """Format trajectory metadata for inclusion in user prompt."""
    blocks = []
    # 〔讲解〕 enumerate 给每条轨迹一个从 0 起的下标 i,展示时统一 +1 改成「人类习惯的从 1 起编号」。
    for i, traj in enumerate(trajectories):
        num_steps = len(traj)
        approx_tokens = _count_tokens_approx(traj)

        # 〔讲解〕 统计这条轨迹里各工具被调用的次数:遍历每步,只看 assistant 发起的 tool_calls。
        tool_counts: dict[str, int] = {}
        for msg in traj:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # 〔讲解〕 一步通常只发一个工具调用,取第 0 个即可。
                tc = msg["tool_calls"][0]
                func = tc.get("function", {})
                # 〔讲解〕 兼容性处理:function 有时不是 dict 而是带 .to_dict() 的对象(不同 SDK 返回类型不一),尝试转成 dict,失败则置空。
                if not isinstance(func, dict):
                    try:
                        func = func.to_dict()
                    except Exception:
                        func = {}
                name = func.get("name")
                # 〔讲解〕 累加计数:tool_counts[name] += 1 的字典安全写法。
                if name:
                    tool_counts[name] = tool_counts.get(name, 0) + 1

        # 〔讲解〕 把计数拼成 "search×3, finish×1" 这种串(按工具名排序保证稳定);没调过任何工具则显示 "none"。
        tool_str = ", ".join(f"{n}×{c}" for n, c in sorted(tool_counts.items())) if tool_counts else "none"
        # 〔讲解〕 {approx_tokens:,} 的 ":," 是千位分隔符,如 12,345,更易读。
        blocks.append(f"Trajectory {i + 1}: {num_steps} steps, ~{approx_tokens:,} tokens | tools: {tool_str}")

    # 〔讲解〕 各轨迹概览块之间空一行隔开。
    return "\n\n".join(blocks)


# 〔讲解〕 工具类:get_solution —— 聚合智能体最常用的第一步工具,直接取出每条轨迹「最后一步」的内容,
# 〔讲解〕 即各 rollout 给出的最终答案。@register_tool 把它以名字 "get_solution" 注册进框架;allow_overwrite=True 允许覆盖同名旧注册(防重复 import 报错)。
@register_tool("get_solution", allow_overwrite=True)
class GetSolutionTool(BaseTool):
    name = "get_solution"
    # 〔讲解〕 description 会作为工具说明喂给 LLM,告诉它这个工具干啥、返回什么结构,LLM 据此决定何时调用。
    description = "Retrieves the final content from trajectories' last step. Returns a list of {trajectory_id, content} entries."
    # 〔讲解〕 parameters 是 JSON Schema,声明入参:trajectory_id 选填(不填=取全部轨迹),required 为空表示可无参调用。
    parameters = {
        "type": "object",
        "properties": {
            "trajectory_id": {"type": "integer", "description": "Trajectory index. Omit to retrieve all trajectories."}
        },
        "required": []
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    # 〔讲解〕 call() 是工具真正被执行时的入口。params 是 LLM 传来的参数;
    # 〔讲解〕 关键设计:轨迹数据不放在 params 里,而是由框架通过 **kwargs 注入(kwargs["trajectories"]),工具按需读取。
    def call(self, params: Union[str, dict], **kwargs) -> list:
        trajectories = kwargs.get("trajectories", [])
        # 〔讲解〕 params 可能是 dict 也可能是裸字符串,这里只在 dict 时取 trajectory_id,否则视为没传(=取全部)。
        trajectory_id = params.get("trajectory_id") if isinstance(params, dict) else None

        # 〔讲解〕 指定了某条轨迹:先校验编号合法(1~n),再把它转成「(0基下标, 轨迹)」单元素列表,后续循环统一处理。
        if trajectory_id is not None:
            n = len(trajectories)
            if trajectory_id < 1 or trajectory_id > n:
                # 〔讲解〕 越界直接返回错误字符串(而非抛异常),让 LLM 看到提示后自我纠正重调。
                return f"[get_solution] 'trajectory_id' must be 1-{n}"
            # 〔讲解〕 LLM 给的是 1 基编号,内部列表是 0 基,故 -1 取真实元素。
            trajs = [(trajectory_id - 1, trajectories[trajectory_id - 1])]
        else:
            # 〔讲解〕 没指定:取全部,enumerate 自动配上 0 基下标。
            trajs = list(enumerate(trajectories))

        results = []
        for i, traj in trajs:
            # 〔讲解〕 取最后一步 traj[-1] 的内容;traj 为空则为 None,再 `or ""` 兜底成空串。
            content = (_get_content(traj[-1]) if traj else None) or ""
            # 〔讲解〕 对外编号 +1 还原成 1 基。
            results.append({"trajectory_id": i + 1, "content": content})
        return results

    # 〔讲解〕 把本工具导出成 OpenAI function-calling 规范的定义(type/function/strict 结构),供注册到 LLM 的可用工具列表。
    def get_tool_definitions(self):
        # 〔讲解〕 .copy() 避免改到类属性原 parameters;additionalProperties=False 禁止 LLM 传未声明的多余字段。
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            # 〔讲解〕 strict=True 开启严格模式,要求模型输出严格符合 schema(OpenAI structured outputs 特性)。
            "strict": True,
        }


# 〔讲解〕 工具类:get_segment —— 读「某条轨迹里一段连续步骤(最多 5 步)」的较完整内容。
# 〔讲解〕 典型用法:先用 search_trajectory 定位到关键步,再用本工具看那一步及其前后文,理解上下文。
@register_tool("get_segment", allow_overwrite=True)
class GetSegmentTool(BaseTool):
    name = "get_segment"
    description = "Reads the full content of a contiguous range of steps from a trajectory (max 5). Use after search_trajectory to inspect a step in full with its surrounding context."
    # 〔讲解〕 入参:轨迹号 + 起止步(均含端点),且要求 end-start ≤ 4(即最多 5 步);三者都必填。
    parameters = {
        "type": "object",
        "properties": {
            "trajectory_id": {"type": "integer", "description": "Trajectory index."},
            "start_step": {"type": "integer", "description": "Start step (inclusive)."},
            "end_step": {"type": "integer", "description": "End step (inclusive); end_step - start_step ≤ 4."}
        },
        "required": ["trajectory_id", "start_step", "end_step"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        # 〔讲解〕 直接按 key 取三个必填参数;若 params 不是 dict 或缺字段会抛异常,被 except 捕获后返回格式错误提示给 LLM。
        try:
            trajectory_id = params["trajectory_id"]
            start_step = params["start_step"]
            end_step = params["end_step"]
        except Exception:
            return "[get_segment] Invalid request format: Input must be a JSON object containing 'trajectory_id', 'start_step', and 'end_step' field"

        trajectories = kwargs.get("trajectories", [])
        n_traj = len(trajectories)
        # 〔讲解〕 校验轨迹编号合法。
        if trajectory_id < 1 or trajectory_id > n_traj:
            return f"[get_segment] 'trajectory_id' must be 1-{n_traj}"
        traj = trajectories[trajectory_id - 1]
        n = len(traj)
        # 〔讲解〕 防御式夹取:把起止步钳制到 [1, n] 区间,即使 LLM 给了越界值也不会出错。
        start_step = max(1, min(start_step, n))
        end_step = max(1, min(end_step, n))
        # 〔讲解〕 起 > 止 则纠正为相等(至少返回一步)。
        if start_step > end_step:
            start_step = end_step
        # 〔讲解〕 强制执行「最多 5 步」上限:跨度超 4 就把终点压回 start+4。
        if end_step - start_step > 4:
            end_step = start_step + 4
        # 〔讲解〕 转成 0 基下标用于切片/索引。
        start_0 = start_step - 1
        end_0 = end_step - 1

        result = []
        # 〔讲解〕 逐步收集 [start_0, end_0] 区间内每一步的精简信息。
        for step_idx in range(start_0, end_0 + 1):
            step = traj[step_idx]
            # 〔讲解〕 每步先放对外的 1 基 step 号和角色(assistant/tool 等)。
            entry = {"step": step_idx + 1, "role": step.get("role", "")}
            content = _get_content(step)
            # 〔讲解〕 思维链字段两种命名兼容:先试 reasoning_content,没有再试 reasoning。
            reasoning = _get_content(step, "reasoning_content") or _get_content(step, "reasoning")
            tool_calls = step.get("tool_calls")
            # 〔讲解〕 三类信息「有才放」,且正文/思维链各截到 600 词,避免段落过长。
            if content:
                entry["content"] = truncate_text(content, 600)
            if reasoning:
                entry["reasoning"] = truncate_text(reasoning, 600)
            # 〔讲解〕 tool_calls 是结构化数据,原样保留不截断(LLM 需要看到完整的调用参数)。
            if tool_calls:
                entry["tool_calls"] = tool_calls
            result.append(entry)
        return result

    # 〔讲解〕 同 GetSolutionTool,导出 function-calling 定义。
    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


# 〔讲解〕 工具类:search_trajectory —— 在「单条轨迹」内按关键词/短语检索,返回相关性最高的若干步。
# 〔讲解〕 这是聚合智能体「在长轨迹里定位证据」的核心检索工具,相关性用前面的 ROUGE-L 召回率衡量。
@register_tool("search_trajectory", allow_overwrite=True)
class SearchTrajectoriesTool(BaseTool):
    name = "search_trajectory"
    description = "Searches for keywords or phrases within a single trajectory. Returns top matching steps ranked by relevance score."
    # 〔讲解〕 入参:轨迹号 + 检索词(必填);role 可选(只搜 tool 步=真实环境观测,或只搜 assistant 步);k=返回条数(默认 5,上限 10)。
    parameters = {
        "type": "object",
        "properties": {
            "trajectory_id": {"type": "integer", "description": "Trajectory index to search within."},
            "query": {"type": "string", "description": "Search term or phrase."},
            "role": {"type": "string", "enum": ["tool", "assistant"], "description": "Optional. Filter to 'tool' steps (actual environment observations) or 'assistant' steps only. Omit to search all steps."},
            "k": {"type": "integer", "description": "Max matches to return (default 5, max 10).", "default": 5}
        },
        "required": ["trajectory_id", "query"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    # 〔讲解〕 内部方法:给一条轨迹的每一步打分。返回 (score, 轨迹下标, 步下标, 步本身) 的四元组列表。
    # 〔讲解〕 role_filter 不为空时只看该角色的步。
    def _score_traj(self, traj_idx, traj, query, role_filter=None):
        scored = []
        for step_idx, step in enumerate(traj):
            # 〔讲解〕 角色过滤:不匹配就跳过这步。
            if role_filter is not None and step.get("role", "") != role_filter:
                continue
            content = _get_content(step) or ""
            reasoning_content = _get_content(step, "reasoning_content") or _get_content(step, "reasoning") or ""
            # 〔讲解〕 tool_calls 序列化成字符串后也参与检索(便于按工具名/参数匹配);无则空串。
            tool_calls_str = json.dumps(step.get("tool_calls"), ensure_ascii=False) if step.get("tool_calls") else ""

            # 〔讲解〕 一步的分数 = 「正文 / 思维链 / 工具调用串」三者各自与 query 的 ROUGE-L 召回率中的最大值。
            # 〔讲解〕 即只要任一部分命中检索词就算这步相关,取最强的那一处当代表分。
            score = max(
                _rouge_l_recall(query, content),
                _rouge_l_recall(query, reasoning_content),
                _rouge_l_recall(query, tool_calls_str),
            )
            # 〔讲解〕 只保留有任何重合(>0)的步,完全不相关的丢弃。
            if score > 0:
                scored.append((score, traj_idx, step_idx, step))
        return scored

    def call(self, params: Union[str, dict], **kwargs) -> str:
        # 〔讲解〕 query 必填,缺失则返回格式错误提示。
        try:
            query = params["query"]
        except Exception:
            return "[search_trajectory] Invalid request format: Input must be a JSON object containing 'query' field"

        trajectory_id = params.get("trajectory_id")
        # 〔讲解〕 trajectory_id 也必填(本工具只搜单条轨迹,不允许全局搜)。
        if trajectory_id is None:
            return "[search_trajectory] 'trajectory_id' is required"
        # 〔讲解〕 返回条数取 min(用户给的 k, 10),硬上限 10 防止返回过多。
        max_results = min(params.get("k", 5), 10)
        role_filter = params.get("role", None)
        trajectories = kwargs.get("trajectories", [])
        n = len(trajectories)

        if trajectory_id < 1 or trajectory_id > n:
            return f"[search_trajectory] 'trajectory_id' must be 1-{n}"

        # 〔讲解〕 对目标轨迹打分(注意 -1 换成 0 基下标),再按分数降序排列(-x[0] 即取负实现从高到低)。
        scored = self._score_traj(trajectory_id - 1, trajectories[trajectory_id - 1], query, role_filter=role_filter)
        scored.sort(key=lambda x: -x[0])

        matches = []
        # 〔讲解〕 取分数最高的前 max_results 条,组装成给 LLM 看的结果条目。
        for score, traj_idx, step_idx, step in scored[:max_results]:
            content = _get_content(step)
            reasoning_content = _get_content(step, "reasoning_content") or _get_content(step, "reasoning")
            tool_calls = step.get("tool_calls")
            # 〔讲解〕 每条命中先带定位信息:轨迹号、步号(均 1 基)、角色、保留 3 位小数的相关性分。
            match_entry = {
                "trajectory_id": traj_idx + 1,
                "step": step_idx + 1,
                "role": step.get("role", ""),
                "score": round(score, 3),
            }
            # 〔讲解〕 正文/思维链按默认 150 词截断(检索结果只给摘要,要看全文用 get_segment)。
            if content:
                match_entry["content"] = truncate_text(content)
            if reasoning_content:
                match_entry["reasoning"] = truncate_text(reasoning_content)
            if tool_calls:
                match_entry["tool_calls"] = tool_calls
            matches.append(match_entry)

        # 〔讲解〕 一条都没命中:返回友好提示(带上 role 过滤条件方便 LLM 判断要不要去掉过滤再搜)。
        if not matches:
            role_msg = f" (role={role_filter})" if role_filter else ""
            return f"[search_trajectory] No matches found for '{query}'{role_msg} in trajectory {trajectory_id}"
        return matches

    # 〔讲解〕 同上,导出 function-calling 定义。
    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


# 〔讲解〕 工具类:finish —— 聚合智能体的「收尾/交卷」工具,提交最终综合结果并结束整个聚合过程。
# 〔讲解〕 它支持两种产出形态(variant):默认 = 短答案(solution+reason);"long_form" = 长篇报告(solution_report+reason)。
# 〔讲解〕 还针对 Qwen 模型做了特殊适配(纯文本 Explanation/Exact Answer 格式,而非 XML 标签)。
@register_tool("finish", allow_overwrite=True)
class FinishTool(BaseTool):
    name = "finish"

    # 〔讲解〕 类级常量:三个可能用到的字段及其给 LLM 的说明。注意 solution/solution_report 都强调「自洽、不得提及轨迹/聚合」——
    # 〔讲解〕 因为最终答案要像是独立写出来的,不能暴露背后是多轨迹聚合得来的。
    _PROPERTIES = {
        "solution": {"type": "string", "description": "A comprehensive, standalone solution as a single string with two XML sections: <explanation>detailed reasoning leading to the answer</explanation><answer>the exact answer</answer>. The explanation must be self-contained and make sense without any reference to trajectories or aggregation."},
        "solution_report": {"type": "string", "description": "The synthesized long-form response. Write it as a complete, standalone report — do not reference trajectories, agents, or aggregation. Cite using <cite url=\"...\">...</cite> if necessary."},
        "reason": {"type": "string", "description": "Meta-reasoning explaining your decision: how you evaluated the trajectories, what evidence you relied on, and how you resolved any conflicts or inconsistencies."},
    }

    # 〔讲解〕 类级常量:给 Qwen 模型用的 solution 字段说明。要求用「Explanation: ... / Exact Answer: ...」纯文本格式
    # 〔讲解〕 (而非 XML 标签),因为 Qwen 对这种格式更稳。下方 call() 会据此用不同正则去校验。
    _QWEN_SOLUTION_DESCRIPTION = (
        "A comprehensive, standalone solution as a single string in the following format:\n"
        "Explanation: {detailed reasoning leading to the answer}\n"
        "Exact Answer: {the exact answer}\n"
        "The explanation must be self-contained and make sense without any reference to trajectories or aggregation."
    )

    # 〔讲解〕 构造时按 variant(产出形态)和 model(模型名)动态决定:必填字段、description、以及 parameters schema。
    def __init__(self, cfg: Optional[dict] = None, variant: str = "", model: str = ""):
        # variant: "" (default) or "long_form"
        super().__init__(cfg)
        self.variant = variant
        # 〔讲解〕 模型名里含 "qwen" 就启用 Qwen 专用的纯文本格式(影响 schema 说明与后续校验分支)。
        self.use_qwen_solution = "qwen" in model.lower()
        # 〔讲解〕 长报告形态:要 solution_report + reason。
        if variant == "long_form":
            required = ["solution_report", "reason"]
            self.description = "Submits the final synthesized long-form response."
        else:
            # 〔讲解〕 默认形态:要 solution + reason。
            required = ["solution", "reason"]
            self.description = "Submits the final synthesized solution."
        # 〔讲解〕 从全量 _PROPERTIES 里只挑本形态需要的字段,组成本次 schema 的 properties。
        properties = {k: v for k, v in self._PROPERTIES.items() if k in required}
        # 〔讲解〕 若是 Qwen 且本形态要 solution,就把 solution 的说明替换成 Qwen 专用版(纯文本格式)。
        if self.use_qwen_solution and "solution" in required:
            # 〔讲解〕 dict(...) 复制一份再改,避免污染上一步从 _PROPERTIES 取出的引用。
            properties = dict(properties)
            properties["solution"] = {"type": "string", "description": self._QWEN_SOLUTION_DESCRIPTION}
        # 〔讲解〕 拼出最终 JSON Schema 作为实例属性(注意:这里覆盖的是实例的 parameters,非类属性)。
        self.parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    # 〔讲解〕 call():收卷时校验 LLM 提交的内容是否合规,合规则返回结构化结果(框架据此终止聚合循环),不合规则返回错误提示让其重交。
    def call(self, params: Union[str, dict], **kwargs) -> str:
        required = self.parameters["required"]
        # 〔讲解〕 先查必填字段有没有全交齐。
        missing = [f for f in required if f not in params]
        if missing:
            return f"[finish] Invalid request format: missing field(s): {', '.join(missing)}"
        # 〔讲解〕 长报告形态:只要求 solution_report 非空白;对外统一把它装进 "solution" 键返回(下游接口一致)。
        if self.variant == "long_form":
            if not params.get("solution_report", "").strip():
                return "[finish] Invalid format: 'solution_report' must not be empty."
            return {"solution": params["solution_report"], "reason": params.get("reason", "")}
        # default: solution + reason
        # 〔讲解〕 默认形态:要校验 solution 内部结构是否含「解释」和「确切答案」两部分。
        solution = params.get("solution", "")
        if self.use_qwen_solution:
            # 〔讲解〕 Qwen 分支:用正则抓纯文本格式。Explanation 抓到下一个 "\nExact Answer:" 或文末(\Z)为止;re.DOTALL 让 . 也能匹配换行。
            explanation_match = re.search(r'Explanation:\s*(.*?)(?=\nExact Answer:|\Z)', solution, re.DOTALL)
            # 〔讲解〕 Exact Answer 抓其后第一个非空白字符(\S)起到文末的内容。
            answer_match = re.search(r'Exact Answer:\s*(\S.*)', solution, re.DOTALL)
            # 〔讲解〕 解释段缺失或全空白 → 报错。.group(1) 取正则第一个捕获组的内容。
            if not explanation_match or not explanation_match.group(1).strip():
                return "[finish] Invalid solution format: missing or empty 'Explanation:' section."
            # 〔讲解〕 答案段缺失或全空白 → 报错。
            if not answer_match or not answer_match.group(1).strip():
                return "[finish] Invalid solution format: missing or empty 'Exact Answer:' section."
        else:
            # 〔讲解〕 非 Qwen 分支:改用 XML 标签格式,抓 <explanation>...</explanation> 和 <answer>...</answer> 内部内容。
            explanation_match = re.search(r'<explanation>(.*?)</explanation>', solution, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
            if not explanation_match or not explanation_match.group(1).strip():
                return "[finish] Invalid solution format: missing or empty <explanation>...</explanation> section."
            if not answer_match or not answer_match.group(1).strip():
                return "[finish] Invalid solution format: missing or empty <answer>...</answer> section."
        # 〔讲解〕 校验通过:把所有必填字段原样打包成 dict 返回(即最终被采纳的聚合结果)。
        return {f: params[f] for f in required}

    # 〔讲解〕 同上,导出 function-calling 定义(注意这里读的是实例级 self.parameters,即构造时按 variant/model 定制后的那份)。
    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }
