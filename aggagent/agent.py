#〔讲解〕本文件是 AggAgent「聚合智能体」的核心驱动类,处于两阶段 MoA pipeline 的「第二阶段(聚合)」。
#〔讲解〕第一阶段(rollout/)已经跑出 N 条独立的 ReAct 搜索轨迹;本文件让一个会调工具的 LLM
#〔讲解〕去「阅读」这 N 条轨迹(用 get_solution/search_trajectory/get_segment 等工具按需翻看),
#〔讲解〕再综合出一份最终答案,最后调 finish 工具提交。上游=rollout 产出的 trajectories;
#〔讲解〕下游=evaluation/ 拿这份答案去判分。依赖 .tools(工具实现)与 .prompts(提示词模板)。
"""AggAgent: tool-calling agent for trajectory aggregation."""

import json
import os
import random
import re
import time

#〔讲解〕litellm:统一各家大模型 API 的调用层。一套 completion() 接口即可切 vLLM 本地 / GPT / Gemini / MiniMax 等。
import litellm

#〔讲解〕导入四个「聚合工具」+ 两个辅助函数。这些是 agent 能调用的「动作」:
#〔讲解〕GetSolutionTool=取各轨迹最终答案; GetSegmentTool=读某轨迹连续若干步原文;
#〔讲解〕SearchTrajectoriesTool=在某轨迹内关键词检索; FinishTool=提交最终解(终止信号)。
#〔讲解〕format_metadata=把 N 条轨迹的概要(步数/token 数)拼成给 LLM 看的元信息;
#〔讲解〕_count_tokens_approx=按字符数粗估 token 量(无需真分词器,chars/4 近似)。
from .tools import (
    GetSolutionTool, GetSegmentTool,
    SearchTrajectoriesTool, FinishTool,
    format_metadata, _count_tokens_approx,
)
#〔讲解〕导入提示词常量。系统/用户提示有多个变体:普通版、Qwen 版、长文本报告版(healthbench 等)。
#〔讲解〕FINAL_MESSAGE=当上下文快爆时,逼模型「别再翻轨迹了,立刻 finish 给答案」的最后通牒文本。
from .prompts import (
    SYSTEM_PROMPT_AGGAGENT, USER_PROMPT_AGGAGENT,
    SYSTEM_PROMPT_AGGAGENT_QWEN,
    SYSTEM_PROMPT_AGGAGENT_REPORT, USER_PROMPT_AGGAGENT_REPORT,
    FINAL_MESSAGE,
)

#〔讲解〕「长文本任务」白名单:这两类任务答案是大段报告,要走专门的 report 版提示词与 finish 字段。
LONG_FORM_TASKS = {"healthbench", "researchrubrics"}


#〔讲解〕AggAgent:聚合智能体主类。对外只暴露 run(question, trajectories)->dict;
#〔讲解〕内部维护一个「LLM ↔ 工具」的多轮对话循环,直到模型调 finish 或上下文耗尽。
class AggAgent:
    """
    AggAgent: agentic aggregation over a set of parallel trajectories.

    Usage::

        agent = AggAgent(model="GLM-4-Flash", api_base="http://...", task="browsecomp")
        result = agent.run(question="...", trajectories=[[msg, ...], [msg, ...]])
        # {"solution": "...", "reason": "..."}
        # or on failure: {"solution": None, "reason": None, "error": "..."}
    """

    #〔讲解〕构造函数:配置模型与运行环境,并实例化本次可用的四个工具。
    #〔讲解〕入参:model=模型名(决定调用路由与提示词变体); api_base=本地 vLLM 地址;
    #〔讲解〕max_context_tokens=上下文预算上限(默认 100K),超了就强制收尾; task=任务名(决定是否走长文本分支);
    #〔讲解〕llm_kwargs=外部直接传入的 litellm 参数(给了就完全覆盖默认调用配置)。
    def __init__(
        self,
        model: str = "",
        api_base: str | None = None,
        max_context_tokens: int = 100 * 1024,
        task: str = "",
        llm_kwargs: dict | None = None,
    ):
        self.model = model
        self.api_base = api_base or ""
        self.max_context_tokens = max_context_tokens
        self.task = task
        self.llm_kwargs = llm_kwargs or {}

        #〔讲解〕据任务类型选 FinishTool 的变体:长文本任务的 finish 收 solution_report 字段,普通任务收 solution。
        variant = "long_form" if task in LONG_FORM_TASKS else ""
        #〔讲解〕组装本次 agent 可调用的工具清单(顺序即注册顺序)。这四个就是聚合智能体的全部「动作空间」。
        tools = [
            GetSolutionTool(),
            SearchTrajectoriesTool(),
            GetSegmentTool(),
            FinishTool(variant=variant, model=model),
        ]

        #〔讲解〕name->tool 的查表字典,后续按模型返回的 tool_name 快速定位到具体工具对象。
        self.tool_map = {tool.name: tool for tool in tools}
        #〔讲解〕把每个工具转成 OpenAI function-calling 规范的 JSON schema 列表,作为 tools 参数喂给 LLM。
        self.tool_description = [self.tool_map[t].get_tool_definitions() for t in self.tool_map]

    #〔讲解〕run:对外主入口。把外部传入的「轨迹列表」包装成内部格式,跑完整聚合循环,做结果合法性校验。
    #〔讲解〕返回成功={"solution":..,"reason":..};失败={"solution":None,"reason":None,"error":..}。
    def run(self, question: str, trajectories: list[list[dict]]) -> dict:
        """
        Synthesize a final answer from N parallel trajectories.

        Args:
            question: The task/question string.
            trajectories: List of N trajectories, each a list of message dicts
                          with keys like "role", "content", "tool_calls".

        Returns:
            On success: {"solution": str, "reason": str}
            On failure: {"solution": None, "reason": None, "error": str}
        """
        #〔讲解〕统一内部数据形态:每条轨迹包成 {"messages": traj},便于 _run 与各工具按 key 取用。
        run_results = [{"messages": traj} for traj in trajectories]
        #〔讲解〕调内部循环,output 形如 {"result":.., "messages":.., "stats":..}。
        output = self._run(question, run_results)
        result = output.get("result")
        #〔讲解〕合法性闸门:result 必须是含 "solution" 键的 dict,否则视为「没产出有效解」而返回错误结构。
        if result is None or not isinstance(result, dict) or "solution" not in result:
            return {"solution": None, "reason": None, "error": "Agent did not produce a valid solution"}
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    #〔讲解〕粗估当前对话总 token:消息正文近似量 + 工具 schema 的字符量折算。用来判断是否逼近上下文上限。
    def count_tokens_approx(self, messages, chars_per_token=4.0):
        #〔讲解〕工具描述本身也占 prompt,所以要把它序列化后的字符数也算进去(否则会低估而触发模型端报错)。
        tool_chars = len(json.dumps(self.tool_description, ensure_ascii=False))
        return _count_tokens_approx(messages, chars_per_token) + int(tool_chars / chars_per_token)

    #〔讲解〕清洗工具名:去掉除字母数字下划线连字符外的字符。防止模型吐出带空格/标点的非法函数名导致匹配失败。
    def sanitize_tool_name(self, name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]', '', name)

    #〔讲解〕把 litellm 返回的 assistant message 对象转成可写回 messages 列表的标准 dict(含 tool_calls)。
    #〔讲解〕关键点:不同模型族对「思维链字段」命名不同,需分别处理,否则回填历史时字段不被模型接受。
    def _message_to_dict(self, assistant_message):
        content = assistant_message.get("content") or ""
        #〔讲解〕reasoning_content=模型的思考过程(推理型模型才有),用 or "" 兜底防 None。
        reasoning_content = assistant_message.get("reasoning_content") or ""
        #〔讲解〕只取首个 tool_call([:1]):本框架强制单工具串行(见下方 parallel_tool_calls=False),一次只执行一个动作。
        tool_calls_list = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    #〔讲解〕函数名同样经清洗,保证与 tool_map 的 key 对齐。
                    "name": self.sanitize_tool_name(tc.function.name),
                    "arguments": tc.function.arguments,
                }
            }
            for tc in assistant_message.tool_calls[:1]
        ] if assistant_message.tool_calls else []
        #〔讲解〕Gemini 分支:它的思维链以结构化 thinking_blocks 形式返回,且历史字段名用 reasoning_content,需原样保留。
        if "gemini" in self.model:
            thinking_blocks = assistant_message.get("thinking_blocks") or []
            return {"role": "assistant", "content": content, "reasoning_content": reasoning_content,
                    "thinking_blocks": thinking_blocks, "tool_calls": tool_calls_list}
        #〔讲解〕其他模型:思维链回填字段名用 reasoning(litellm 对非 Gemini 模型的约定)。
        else:
            return {"role": "assistant", "content": content, "reasoning": reasoning_content,
                    "tool_calls": tool_calls_list}

    #〔讲解〕call_server:真正发起一次 LLM 调用,带重试。是 agent 与模型对话的「单步通信」原子操作。
    #〔讲解〕only_finish=True 时只暴露 finish 工具(用于上下文爆掉后逼模型直接收尾,不许再翻轨迹)。
    def call_server(self, messages, max_tries=5, only_finish=False):
        base_sleep_time = 1
        #〔讲解〕据 only_finish 决定本次给模型看的工具集:全部工具 vs 仅 finish。
        tools = self.tool_description if not only_finish else [self.tool_map["finish"].get_tool_definitions()]
        #〔讲解〕若外部显式传了 llm_kwargs,则以它为准只补 messages/tools;否则走下面这套按模型族定制的默认参数。
        if self.llm_kwargs:
            litellm_kwargs = {**self.llm_kwargs, "messages": messages,"tools": tools}
        else:
            #〔讲解〕默认配置:走本地 vLLM(hosted_vllm/ 前缀 + EMPTY key + api_base)。
            #〔讲解〕temperature=1.0/top_p=0.95=保留一定随机性; parallel_tool_calls=False=强制一次只调一个工具(对应 [:1])。
            litellm_kwargs = {
                "model": "hosted_vllm/" + self.model,
                "api_key": "EMPTY",
                "api_base": self.api_base,
                "messages": messages,
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 10000,
                "parallel_tool_calls": False,
                "tools": tools,
            }
            #〔讲解〕以下按模型族打补丁:换调用路由、换 key 来源、加/删特定参数。每家 API 习惯不同,逐一适配。
            #〔讲解〕Gemini:换 gemini/ 路由 + 环境变量 key,删掉本地相关参数,开低强度推理,Gemini 不支持 parallel_tool_calls 故删。
            if "gemini" in self.model.lower():
                litellm_kwargs["model"] = "gemini/" + self.model
                litellm_kwargs["api_key"] = os.getenv("GEMINI_API_KEY")
                del litellm_kwargs["api_base"]
                litellm_kwargs["reasoning_effort"] = "low"
                del litellm_kwargs["parallel_tool_calls"]
            #〔讲解〕oss(开源推理模型,如 gpt-oss):开高强度推理。
            elif "oss" in self.model.lower():
                litellm_kwargs["reasoning_effort"] = "high"
            #〔讲解〕GPT:换 openai/ 路由 + OPENAI_API_KEY,删本地 api_base;OpenAI 新模型不接受 top_p 与默认温度组合,故删 top_p。
            elif "gpt" in self.model.lower():
                litellm_kwargs["model"] = "openai/" + self.model
                litellm_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
                del litellm_kwargs["api_base"]
                del litellm_kwargs["top_p"]
            #〔讲解〕MiniMax:通过 extra_body 开启「推理/答案分离」返回。
            elif "minimax" in self.model.lower():
                litellm_kwargs["extra_body"] = {"reasoning_split": True}

        #〔讲解〕重试循环:最多 max_tries 次。只有「拿到含 tool_calls 的回复」才算成功;没调工具视为无效,继续重试。
        for attempt in range(max_tries):
            try:
                response = litellm.completion(**litellm_kwargs)
                #〔讲解〕本框架要求模型每步都必须调用工具(纯文字回复不算数),故以 tool_calls 存在与否作为成功判据。
                if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                    return response
            #〔讲解〕单独捕获「上下文超长」错误:不可通过重试解决,直接返回哨兵字符串让上层走收尾逻辑。
            except litellm.BadRequestError as e:
                if "context length" in str(e).lower() or "input tokens" in str(e).lower():
                    print(f"Context length exceeded: {e}")
                    return "ContextLengthError"
                print(f"Error: {e}")
            except Exception as e:
                print(f"Error: {e}")

            #〔讲解〕指数退避 + 抖动重试:sleep = base*2^attempt + 随机(0,1),上限 30s。避免对服务端密集打点。
            if attempt < max_tries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 30)
                time.sleep(sleep_time)

        #〔讲解〕重试全部失败,返回哨兵字符串 "Server error"(上层据 isinstance(str) 判定为失败)。
        return "Server error"

    #〔讲解〕统一封装内部循环的返回壳:result=最终解或 None; messages=完整对话记录; stats=统计信息。
    def _make_output(self, result, messages, stats):
        return {"result": result, "messages": messages, "stats": stats}

    #〔讲解〕_run:聚合的核心驱动循环——「构造初始提示 → 反复(让模型调工具→执行工具→回填结果) → 直到 finish/超限」。
    #〔讲解〕这是整个 ReAct 式聚合的引擎,理解本函数即理解 AggAgent 全貌。
    def _run(self, question, run_results):
        #〔讲解〕从内部结构里抽出纯轨迹列表;format_metadata 生成给模型的轨迹概览(几条、各多少步/token)。
        trajectories = [r.get("messages", []) for r in run_results]
        metadata = format_metadata(trajectories)
        iteration = 0
        #〔讲解〕硬性上限:最多 100 轮工具交互,防止模型陷入无限翻轨迹而不收尾。
        MAX_ITERATIONS = 100

        #〔讲解〕按「任务类型/模型族」三分支选择提示词模板,并把 question/metadata/轨迹条数填入用户提示。
        #〔讲解〕长文本任务→report 版; Qwen 模型→Qwen 专用系统提示; 其余→通用版。
        if self.task in LONG_FORM_TASKS:
            system_prompt = SYSTEM_PROMPT_AGGAGENT_REPORT
            user_prompt = USER_PROMPT_AGGAGENT_REPORT.format(question=question, metadata=metadata, traj_N=len(trajectories))
        elif "qwen" in self.model.lower():
            system_prompt = SYSTEM_PROMPT_AGGAGENT_QWEN
            user_prompt = USER_PROMPT_AGGAGENT.format(question=question, metadata=metadata, traj_N=len(trajectories))
        else:
            system_prompt = SYSTEM_PROMPT_AGGAGENT
            user_prompt = USER_PROMPT_AGGAGENT.format(question=question, metadata=metadata, traj_N=len(trajectories))

        #〔讲解〕messages=喂给模型的「工作上下文」(会被裁剪/收尾); full_messages=完整留痕(用于返回与调试,不裁剪)。
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        full_messages = list(messages)

        #〔讲解〕统计盘:轮数、服务端错误数、工具调用错误数、各工具调用次数、是否触上下文上限、每步 token 用量。
        stats = {
            "iterations": 0,
            "server_errors": 0,
            "tool_call_errors": 0,
            "tool_calls": {},
            "context_limit_reached": False,
            "token_usage_each_step": [],
        }

        #〔讲解〕内嵌小工具:把每次 LLM 调用的 input/output token 数记进 stats(供后续 cost 统计)。
        #〔讲解〕response 是哨兵字符串(出错)时记 None;否则从 response.usage 取 prompt/completion tokens。
        def store_token_usage(response, iteration):
            if response is None or isinstance(response, str):
                entry = {"iteration": iteration, "input_tokens": None, "output_tokens": None}
            elif hasattr(response, "usage") and response.usage:
                entry = {
                    "iteration": iteration,
                    "input_tokens": getattr(response.usage, "prompt_tokens", None),
                    "output_tokens": getattr(response.usage, "completion_tokens", None),
                }
            else:
                entry = {"iteration": iteration, "input_tokens": None, "output_tokens": None}
            stats["token_usage_each_step"].append(entry)

        #〔讲解〕主循环:每一轮 = 一次「模型决策(调哪个工具)+ 执行该工具 + 把结果喂回」。
        while iteration < MAX_ITERATIONS:
            iteration += 1
            stats["iterations"] = iteration
            #〔讲解〕让模型基于当前 messages 决定下一步动作(返回带 tool_calls 的 response,或哨兵字符串)。
            response = self.call_server(messages)
            store_token_usage(response, iteration)

            #〔讲解〕上下文超长哨兵:直接终止,标记 context_limit_reached,返回空解。
            if response == "ContextLengthError":
                stats["context_limit_reached"] = True
                return self._make_output(None, full_messages, stats)

            #〔讲解〕其他哨兵字符串(如 "Server error"):记一次服务端错误,把错误当作 assistant 消息写入,然后进入下一轮重试。
            if isinstance(response, str):
                stats["server_errors"] += 1
                error_message = {"role": "assistant", "content": response}
                messages.append(error_message)
                full_messages.append(error_message)
                continue

            #〔讲解〕拿到正常回复:取出 assistant message,转成 dict 并同时写入工作上下文与完整留痕。
            assistant_message = response.choices[0].message
            assistant_dict = self._message_to_dict(assistant_message)
            messages.append(assistant_dict)
            full_messages.append(assistant_dict)

            #〔讲解〕执行模型本轮请求的工具(同样只取首个 [:1],对应单工具串行约定)。
            for tool_call in assistant_message.tool_calls[:1] if assistant_message.tool_calls else []:
                tool_name = self.sanitize_tool_name(tool_call.function.name)
                #〔讲解〕累计该工具被调次数(统计模型的行为分布)。
                stats["tool_calls"][tool_name] = stats["tool_calls"].get(tool_name, 0) + 1
                try:
                    #〔讲解〕模型把参数以 JSON 字符串形式给出,需先 parse 再传给工具;结果再序列化回字符串以喂回模型。
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_result = self.custom_call_tool(tool_name, tool_args, trajectories=trajectories)
                    tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                #〔讲解〕参数非法 JSON 或工具执行抛错:记一次 tool_call 错误,把报错文本作为工具结果回喂(让模型自我纠正)。
                except Exception as e:
                    print(e)
                    stats["tool_call_errors"] += 1
                    tool_result = None
                    tool_result_str = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field'

                #〔讲解〕终止判定:模型调了 finish 且返回的是 dict,即视为「已产出最终解」,立刻带 result 退出整个循环。
                if tool_name == "finish" and isinstance(tool_result, dict):
                    return self._make_output(tool_result, full_messages, stats)

                #〔讲解〕非 finish 工具:把结果以 role="tool" 消息回填(必须带 tool_call_id 以与刚才的调用配对),进入下一轮。
                tool_message = {"role": "tool", "tool_call_id": tool_call.id, "name": tool_name,
                                "content": tool_result_str}
                messages.append(tool_message)
                full_messages.append(tool_message)

            #〔讲解〕本轮结束后检查上下文预算:若已超 max_context_tokens,进入「强制收尾」流程(下面这整段)。
            if self.count_tokens_approx(messages) > self.max_context_tokens:
                stats["context_limit_reached"] = True
                #〔讲解〕追加 FINAL_MESSAGE 最后通牒:告知模型已达上限,必须立刻调 finish 给出最可能的答案。
                final_msg = {"role": "user", "content": FINAL_MESSAGE}
                messages.append(final_msg)
                full_messages.append(final_msg)
                #〔讲解〕only_finish=True:本次只放出 finish 工具,断绝模型继续翻轨迹的可能,强制其收口。
                response = self.call_server(messages, only_finish=True)
                store_token_usage(response, iteration)
                #〔讲解〕收尾这一次调用又失败:记错误、写留痕,直接返回空解(放弃)。
                if isinstance(response, str):
                    stats["server_errors"] += 1
                    err_msg = {"role": "assistant", "content": response}
                    messages.append(err_msg)
                    full_messages.append(err_msg)
                    return self._make_output(None, full_messages, stats)

                #〔讲解〕收尾调用成功:取出 assistant 消息并回填(变量名 assistant_dict2 区别于主循环那次)。
                assistant_message = response.choices[0].message
                assistant_dict2 = self._message_to_dict(assistant_message)
                messages.append(assistant_dict2)
                full_messages.append(assistant_dict2)
                #〔讲解〕执行收尾这一步的工具调用(理应是 finish)。逻辑与主循环里那段对称,只是上下文已锁定为仅 finish。
                for tc in assistant_message.tool_calls[:1] if assistant_message.tool_calls else []:
                    tn = self.sanitize_tool_name(tc.function.name)
                    stats["tool_calls"][tn] = stats["tool_calls"].get(tn, 0) + 1
                    try:
                        tool_args = json.loads(tc.function.arguments)
                        tool_result = self.custom_call_tool(tn, tool_args, trajectories=trajectories)
                        tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                    except Exception as e:
                        print(e)
                        stats["tool_call_errors"] += 1
                        tool_result = None
                        tool_result_str = "Error: Tool call is not a valid JSON."
                    #〔讲解〕收尾时若成功 finish,返回最终解;否则下一行直接落到返回空解(放弃,不再给第二次机会)。
                    if tn == "finish" and isinstance(tool_result, dict):
                        return self._make_output(tool_result, full_messages, stats)
                #〔讲解〕收尾后仍未拿到有效 finish:返回空解结束。
                return self._make_output(None, full_messages, stats)

        #〔讲解〕跑满 MAX_ITERATIONS 仍未 finish:视为未产出有效解,返回空 result。
        return self._make_output(None, full_messages, stats)

    #〔讲解〕工具调度器:按 tool_name 在 tool_map 里找到对应工具对象并调用其 call();找不到则返回错误字符串。
    #〔讲解〕**kwargs 会把 trajectories=.. 透传给各工具(工具据此真正去读对应轨迹内容)。
    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in self.tool_map:
            return self.tool_map[tool_name].call(tool_args, **kwargs)
        return f"Error: Tool {tool_name} not found"
