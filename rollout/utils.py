#〔讲解〕本文件 = rollout 阶段的「工具箱/账房」。rollout 负责跑 ReAct 搜索轨迹(模型一边想一边调检索工具)。
#〔讲解〕这里不跑轨迹本身,只提供三类辅助:① 给一条轨迹算钱(token 费 + 工具调用费);
#〔讲解〕② 把每一步消耗的 token 数记进 debug_data 账本;③ 把对话 messages 整理成 tokenizer 能吃的格式。
#〔讲解〕上游 = react_agent 跑轨迹时会调本文件记 token / 算成本;下游 = 实验报告读这些成本数做 cost 对比。
"""Rollout utilities for AggAgent."""

import json

#〔讲解〕模型单价表:key=模型名,value=(输入价, 输出价),单位是「每 100 万 token 多少美元」。
#〔讲解〕大模型按 token 收费且输入比输出便宜,这里把各模型的官方定价硬编码进来供算账查表用。
# Model costs per 1M tokens (input, output) in USD
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "GLM-4.7-Flash": (0.07, 0.40),
    "MiniMax-M2.5": (0.30, 1.20),
    "gpt-oss-120b": (0.15, 0.60),
    "gemini-3-flash-preview": (0.5, 3.0),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.1": (1.25, 10),
    "Qwen3.5-122B-A10B": (0.26, 2.08),
}

#〔讲解〕工具调用单价:搜索(search)和抓网页(scrape/visit)各自「每 1000 次调用多少美元」。
#〔讲解〕ReAct 轨迹会反复调外部检索 API,这部分钱要单独算,不算在模型 token 费里。
# Tool usage costs per 1K calls in USD
SEARCH_COST_PER_K = 0.5
SCRAPE_COST_PER_K = 0.83


#〔讲解〕给「单条 rollout 轨迹」算一份成本明细。
#〔讲解〕入参:debug_data=这条轨迹跑下来的账本(含每步 token、工具调用次数);model_name=用哪个模型(查单价表);
#〔讲解〕task=任务名,某些任务(browsecomp-plus)用本地工具不花钱,需特判。
#〔讲解〕返回:{"rollout": token费, "tool": 工具费},单位美元;模型不在价表里就全返 0。
#〔讲解〕角色:它是 cost 统计的最小单元,实验报告把每条轨迹的成本汇总起来做 RMB/USD 真实货币对比。
def compute_rollout_cost(debug_data: dict, model_name: str, task: str | None = None) -> dict:
    """
    Compute cost breakdown for a single rollout.

    Args:
        debug_data: The debug_data dict (token_lengths_each_step, tool_usage)
        model_name: Model basename (e.g. "GLM-4.7-Flash") for cost lookup
        task: Task name; browsecomp-plus uses local FAISS tools (no tool cost)

    Returns:
        {"rollout": token_cost, "tool": tool_cost} in USD, or zeros if model unknown.
    """
    #〔讲解〕先查单价表;查不到(返回 None)说明这个模型没登记定价,直接返 0 成本,避免后面拿 None 做乘法报错。
    costs = MODEL_COSTS.get(model_name)
    if not costs:
        return {"rollout": 0.0, "tool": 0.0}

    #〔讲解〕把 (输入价, 输出价) 元组拆成两个变量,方便下面分别算输入费和输出费。
    input_cost_per_m, output_cost_per_m = costs

    # --- token cost (incremental input + output) ---
    #〔讲解〕下面这段是全文件最绕的地方,核心是「增量计费」。
    #〔讲解〕背景:ReAct 每一步都把之前所有对话重新喂给模型,所以每步记录的 input_tokens 是【累计值】(越来越大,
    #〔讲解〕因为历史越积越长),不是这一步新增的。如果直接把每步 input_tokens 相加会严重重复计费。
    #〔讲解〕做法:只累加相邻两步 input 的【差值】(本步累计 - 上步累计),还原出每步真正新读入的 token 量。
    total_input_delta = 0  #〔讲解〕累计「输入增量」总和(用于算输入费)。
    total_output = 0       #〔讲解〕累计输出 token 总和(输出每步独立产生,可直接相加)。
    prev_input = 0         #〔讲解〕记上一步的累计 input,用来和本步相减求增量。
    try:
        for step in debug_data.get("token_lengths_each_step", []):
            curr_input = step.get("input_tokens", 0)  #〔讲解〕本步的累计输入 token。
            #〔讲解〕本步增量 = 本步累计 - 上步累计;max(...,0) 兜底:万一出现回退/异常导致差值为负,按 0 计不倒扣。
            total_input_delta += max(curr_input - prev_input, 0)
            prev_input = curr_input  #〔讲解〕把「上一步累计」更新成本步,供下一轮循环相减。
            total_output += step.get("output_tokens", 0)
    #〔讲解〕账本字段缺失或类型不对(KeyError/TypeError)时整体跳过,保证算账不会因脏数据崩掉。
    except (KeyError, TypeError):
        pass

    #〔讲解〕token 总费 = 输入增量×输入单价 + 输出总量×输出单价,再÷100万(因为单价是「每百万 token」)换算成美元。
    token_cost = (
        total_input_delta * input_cost_per_m + total_output * output_cost_per_m
    ) / 1_000_000

    # --- tool call cost (external API tools only) ---
    # browsecomp-plus uses local FAISS-based tools, so no tool cost applies.
    #〔讲解〕工具费:browsecomp-plus 这个任务用的是本地 FAISS 向量检索(自己跑,不调付费 API),所以它工具费=0,直接跳过。
    tool_cost = 0.0
    if task != "browsecomp-plus":
        try:
            tool_usage = debug_data.get("tool_usage", {})  #〔讲解〕取工具调用次数统计字典。
            search_count = tool_usage.get("search", 0)     #〔讲解〕搜索调用次数。
            scrape_count = tool_usage.get("visit", 0)      #〔讲解〕注意:抓网页的工具内部叫 "visit",对应 scrape 单价。
            #〔讲解〕工具总费 = 搜索次数×搜索单价 + 抓页次数×抓页单价,再÷1000(单价是「每千次调用」)换成美元。
            tool_cost = (
                search_count * SEARCH_COST_PER_K + scrape_count * SCRAPE_COST_PER_K
            ) / 1_000
        except (KeyError, TypeError):
            pass

    return {"rollout": token_cost, "tool": tool_cost}


#〔讲解〕往账本 debug_data["token_lengths_each_step"] 追加「某一步用了多少 token」的记录。
#〔讲解〕入参:iteration=第几步;response=模型返回对象(可能带 usage);也可直接传 input_tokens/output_tokens。
#〔讲解〕角色:rollout 每跑完一步就调一次,把 token 消耗逐步记下,供上面的 compute_rollout_cost 事后算账。
#〔讲解〕设计要点:四种来源按优先级取 token 数,取不到就记 None(占位但不计费)。
def store_token_length(
    debug_data: dict,
    iteration: int,
    response=None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Append a token-length entry to debug_data["token_lengths_each_step"]."""
    #〔讲解〕情况1:没有 response,或 response 是纯字符串(没 usage 信息)→ token 记 None。
    if response is None or isinstance(response, str):
        token_entry = {"iteration": iteration, "input_tokens": None, "output_tokens": None}
    #〔讲解〕情况2:调用方已经显式把 token 数算好传进来 → 优先用传入值(覆盖 response 里的)。
    elif input_tokens is not None or output_tokens is not None:
        token_entry = {"iteration": iteration, "input_tokens": input_tokens, "output_tokens": output_tokens}
    #〔讲解〕情况3:response 自带 usage 字段(OpenAI 风格返回)→ 从中读 prompt/completion token 数。
    #〔讲解〕getattr(..., None) 是防御写法:字段不存在也返 None 而不报错。
    elif hasattr(response, "usage") and response.usage:
        token_entry = {
            "iteration": iteration,
            "input_tokens": getattr(response.usage, "prompt_tokens", None),
            "output_tokens": getattr(response.usage, "completion_tokens", None),
        }
    #〔讲解〕情况4:兜底,啥都取不到就记 None。
    else:
        token_entry = {"iteration": iteration, "input_tokens": None, "output_tokens": None}
    debug_data["token_lengths_each_step"].append(token_entry)


#〔讲解〕把对话 messages 改写成 tokenizer.apply_chat_template 能正确处理的格式。
#〔讲解〕用途:很多模型有自己的 chat 模板,要求字段名/数据结构符合特定约定,本函数做这层适配。
#〔讲解〕入参:messages=原始对话列表;model_type=模型家族(glm/qwen/minimax 等),不同家族处理方式不同。
#〔讲解〕返回:整理后的新列表(不改原 messages,全程拷贝出新对象)。
def prepare_messages_for_tokenization(messages: list, model_type: str | None) -> list:
    """
    Normalize messages for tokenizer.apply_chat_template.

    Renames "reasoning" → "reasoning_content" and, for GLM/Qwen/MiniMax models,
    deserializes tool_call arguments from JSON strings to dicts.
    """
    #〔讲解〕第一步通用处理:逐条拷贝消息,把字段名 "reasoning" 改成模板要求的 "reasoning_content"。
    #〔讲解〕dict(msg) 是浅拷贝,保证不污染原始数据;pop 取出旧 key 同时删掉它。
    normalized = []
    for msg in messages:
        prep = dict(msg)
        if "reasoning" in prep:
            prep["reasoning_content"] = prep.pop("reasoning")
        normalized.append(prep)

    #〔讲解〕第二步家族特判:glm/qwen/minimax 这几类模型,要求 tool_call 的 arguments 是【dict 对象】而非 JSON 字符串。
    #〔讲解〕(很多 API 返回的 arguments 是序列化好的字符串,这里要反序列化回来。)
    if model_type in ("glm", "qwen", "minimax"):
        prepared = []
        #〔讲解〕zip 同时遍历原始 msg(读 role/tool_calls 判断条件)和已改名的 prep(作为输出基底)。
        for msg, prep in zip(messages, normalized):
            #〔讲解〕只处理「assistant 角色且发起了工具调用」的消息,其余原样放行。
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                #〔讲解〕逐个工具调用重建:用 {**tc} 复制原字段,再把内层 function.arguments 用 json.loads 从字符串解析成 dict。
                #〔讲解〕{**tc, "function": {...}} 这种「展开+覆盖」写法 = 拷贝原结构同时只替换 function 这一项,不动其他字段。
                tool_calls_list = [
                    {
                        **tc,
                        "function": {
                            **tc["function"],
                            "arguments": json.loads(tc["function"]["arguments"]),
                        },
                    }
                    for tc in msg["tool_calls"]
                ]
                #〔讲解〕用解析后的 tool_calls 覆盖 prep 里的同名字段,得到这条消息的最终版本。
                prepared.append({**prep, "tool_calls": tool_calls_list})
            else:
                prepared.append(prep)
        return prepared

    #〔讲解〕非上述家族:不需要 tool_call 反序列化,直接返回只做过改名的 normalized。
    return normalized
