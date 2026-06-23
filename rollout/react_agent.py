# 〔讲解〕 模块说明:本文件是「rollout 阶段」的核心——一个会循环调工具的 ReAct 智能体。
# 〔讲解〕 在 AggAgent 两阶段 MoA 流程里属于第一环:对每道题反复「思考→调检索/网页工具→读结果→再思考」,直到给出答案,产出一条完整轨迹(messages + debug_data)。
# 〔讲解〕 上游=run_multi_react.py(批量调度器)把每道题的数据喂进来;下游=rollout 结果落盘成 json,供 aggregation/ 第二环聚合。
# 〔讲解〕 关键依赖:qwen_agent 提供 Agent 基类与消息 schema;litellm 统一各家 LLM 的调用接口;tools/ 提供 search/visit 等真实工具;evaluation/ 在轨迹末尾顺手打个自动分。
import json
import json5
import os
import asyncio
import copy
import random
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union
# 〔讲解〕 FnCallAgent=qwen_agent 里「支持函数/工具调用」的智能体基类,本类继承它(但其实大部分逻辑本文件自己重写了)。
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.tools import BaseTool
# 〔讲解〕 litellm=统一封装 OpenAI/GLM/Claude 等多家模型 API 的库,后面用它的 acompletion 异步调模型。
import litellm
# 〔讲解〕 AutoTokenizer=HuggingFace 分词器,这里只用来「数 token」以判断是否超上下文,不做生成。
from transformers import AutoTokenizer
# 〔讲解〕 prompts 是同目录的 prompts.py,里面定义了 SYSTEM_PROMPT / SYSTEM_PROMPT_DR 等提示词常量。
from prompts import *
# 〔讲解〕 从 evaluation 包拿两样东西:get_task_instruction(按任务返回追加在题目后的指令)、get_evaluator(按任务返回打分器)。
from evaluation import get_task_instruction, get_evaluator

# 〔讲解〕 下面两个常量是「上下文/步数用尽」时塞给模型的提示文本,逼它停止调工具、直接给最终答案。
TRUNCATED_MESSAGE = """
--- Maximum Length Limit Reached ---
You have reached the maximum length limit.
The response is truncated."""
FINAL_MESSAGE = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer."

# 〔讲解〕 导入四个工具实现:tool_search/tool_visit 是通用搜索+网页访问;带 _bcp 后缀的是 browsecomp-plus 任务专用的检索/取文档工具。
from tools.tool_search import *
from tools.tool_visit import *
from tools.tool_search_bcp import *
from tools.tool_get_document_bcp import *
# 〔讲解〕 三个工具函数:compute_rollout_cost=按真实定价算这条轨迹花了多少钱;store_token_length=记录每步 token 数;prepare_messages_for_tokenization=把 messages 整理成分词器能吃的格式。
from utils import compute_rollout_cost, store_token_length, prepare_messages_for_tokenization


# 〔讲解〕 本类=多轮 ReAct 智能体,是本文件主角。职责:对单道题跑完整个「想→调工具→读→再想」的循环并产出结果。
# 〔讲解〕 继承 FnCallAgent 主要为复用其初始化骨架,但核心方法(call_server/_run/custom_call_tool)都在本类重写,行为以本文件为准。
class MultiTurnReactAgent(FnCallAgent):
    # 〔讲解〕 构造函数:从 llm 配置里取出生成参数、模型类型/名、分词器,并按 task 决定挂哪套工具。
    # 〔讲解〕 入参 function_list=可用工具名;llm=模型配置字典(含 generate_cfg、model 路径等);kwargs 里还带 task/max_tokens 等运行参数。
    def __init__(self, function_list: Optional[List[Union[str, Dict, BaseTool]]] = None, llm: Optional[Dict] = None, **kwargs):
        # 〔讲解〕 generate_cfg=传给 litellm 的生成参数(temperature、model 名等),后面每次调模型都会 copy 一份再补 messages/tools。
        self.llm_generate_cfg = llm["generate_cfg"] if llm else {}
        # 〔讲解〕 model_type 用来区分模型家族(如 "glm"),影响数 token 时的 chat_template 特殊参数。
        self.model_type = llm.get("model_type") if llm else None
        self.model_name = llm.get("model_name") if llm else None
        # 〔讲解〕 用 llm["model"] 指定的路径加载分词器;注意这只为本地精确数 token,真正生成走 litellm 远程接口。
        self.tokenizer = AutoTokenizer.from_pretrained(llm["model"]) if llm else None
        self.task = kwargs.get("task", None)
        # 〔讲解〕 单条轨迹最多调几次 LLM(防死循环的硬上限),默认 100 次。
        self.max_llm_call_per_run = kwargs.get("max_llm_call_per_run", 100)
        # 〔讲解〕 上下文 token 上限,超过就触发「截断+逼答」流程,默认 108000。
        self.max_tokens = kwargs.get("max_tokens", 108000)
        self.system_prompt = SYSTEM_PROMPT

        # 〔讲解〕 按任务类型分流挂载工具:browsecomp-plus 用 BCP 专用工具;一批检索类任务用通用 Search+Visit;其余任务不挂工具(纯靠模型回答)。
        if self.task == "browsecomp-plus":
            self.tool_class = [SearchBCP(), GetDocumentBCP()]
            self.function_list = ["search_bcp", "get_document_bcp"]
        elif self.task in ["browsecomp", "deepsearchqa", "hle", "healthbench", "researchrubrics"]:
            self.tool_class = [Search(), Visit()]
            self.function_list = ["visit", "search"]
        else:
            self.tool_class = []
            self.function_list = []

        # 〔讲解〕 tool_map:工具名→工具实例,后面按模型给出的工具名 O(1) 查到实例去执行。
        self.tool_map = {tool.name: tool for tool in self.tool_class} if self.tool_class else {}
        # 〔讲解〕 tool_description:把每个工具的「函数签名描述」收集成列表(OpenAI function-calling 格式),传给模型让它知道能调什么。
        self.tool_description = [self.tool_map[tool].get_tool_definitions() for tool in self.tool_map] if self.tool_class else []

    # 〔讲解〕 调一次模型(带重试)。职责:把 messages(+可选 tools)发给 litellm,失败则指数退避重试,拿到「有效」回复才返回。
    # 〔讲解〕 入参 no_tool_calls=True 时不传 tools(用在最后逼模型直接出答案、不许再调工具)。返回:成功=litellm response 对象;彻底失败="Server error" 字符串。
    async def call_server(self, msgs: List[Dict[str, Any]], max_tries: int = 3, no_tool_calls: bool = False, tool_description: List[Dict[str, Any]] = None) -> Union[Any, str]:
        # 〔讲解〕 copy 一份生成配置再往里塞本次的 messages/tools,避免污染 self.llm_generate_cfg 原字典。
        kwargs = self.llm_generate_cfg.copy()
        kwargs["messages"] = msgs
        if not no_tool_calls and tool_description:
            kwargs["tools"] = tool_description

        base_sleep_time = 1
        response = None
        for attempt in range(max_tries):
            try:
                # 〔讲解〕 真正的异步模型调用;await 让出事件循环,使多道题的 rollout 能并发跑。
                response = await litellm.acompletion(**kwargs)
                content = response.choices[0].message.content
                # 〔讲解〕 reasoning_content=部分模型(如思考型)单独返回的「思维链」字段,可能为空。
                reasoning_content = response.choices[0].message.get("reasoning_content") or ""
                # 〔讲解〕 下面三选一即视为「有效回复」并直接返回:有正文内容 / 有工具调用 / 有思维链。三者全空则视为坏回复,继续重试。
                if content and content.strip():
                    return response
                elif hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                    return response
                elif reasoning_content and reasoning_content.strip():
                    return response
            except Exception as e:
                # 〔讲解〕 把异常和(若有)HTTP 响应体打出来便于排查,但不中断——交给外层的重试循环。
                print(f"Error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        print(f"Response dump: {e.response.text if hasattr(e.response, 'text') else e.response}")
                    except:
                        print(f"Response object: {e.response}")
            if attempt < max_tries - 1:
                # 〔讲解〕 指数退避 + 随机抖动,封顶 30s:重试越多等越久,抖动避免多个请求同时重试撞在一起。
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 30)
                await asyncio.sleep(sleep_time)

        if response:
            print(f"Response dump: {response.model_dump() if hasattr(response, 'model_dump') else response}")
        # 〔讲解〕 重试全部失败:返回字符串哨兵,外层 _run 会据此判断 isinstance(response, str) 走出错分支。
        return "Server error"

    # 〔讲解〕 数 token:用本地分词器套上 chat 模板算出当前对话(+工具描述)占多少 token,用于判断是否超上下文。
    # 〔讲解〕 关键:add_generation_prompt=True 表示把「待模型续写」的占位也算进去,数得更接近真实请求长度。
    def count_tokens(self, messages: List[Dict[str, Any]], tool_description: Optional[List[Dict[str, Any]]] = None, reasoning_effort: str = 'high', add_generation_prompt: bool = True) -> int:
        # 〔讲解〕 先把 messages 规整成分词器能吃的纯文本结构(不同 model_type 处理略有差异)。
        prepared_messages = prepare_messages_for_tokenization(messages, self.model_type)
        apply_template_kwargs = {
            "tokenize": True,
            "reasoning_effort": reasoning_effort,
            "add_generation_prompt": add_generation_prompt
        }
        if tool_description is not None: apply_template_kwargs["tools"] = tool_description
        # 〔讲解〕 GLM 系列模板要额外开思考开关,否则模板渲染出的 token 数与实际不符。
        if self.model_type == "glm": apply_template_kwargs["chat_template_kwargs"] = {"enable_thinking": True, "clear_thinking": False}
        # 〔讲解〕 apply_chat_template 把消息按模型对话格式拼接并分词;返回可能是 token id 列表,也可能是带 input_ids 的字典,故下面分两种取长度。
        tokens = self.tokenizer.apply_chat_template(prepared_messages, **apply_template_kwargs)
        if isinstance(tokens, list): return len(tokens)
        return len(tokens["input_ids"])

    # 〔讲解〕 清洗工具名:有些模型会把特殊标记(如 <|...|>)粘进工具名里,这里切掉这些标记并去空白,得到干净的工具名。
    def sanitize_tool_name(self, tool_name: str) -> str:
        return tool_name.split('<|')[0].split('|>')[0].strip()

    # 〔讲解〕 判断模型这段回复里是否「已经给出答案」,仅用于统计 termination 标签,不影响流程。
    # 〔讲解〕 healthbench/researchrubrics 是开放式长答,任何非空回复都算答案;其余任务靠是否出现 "Answer:" 等标记词来粗判。
    def answer_in_content(self, content):
        """Check if answer is in content. Just for statistics."""
        if not content: return False
        if self.task in ["healthbench", "researchrubrics"]: return True
        return "Answer:" in content or "Answer**" in content or "Exact Answer" in content

    # 〔讲解〕 轨迹跑完后顺手做一次「自动判分」,把分数挂到结果里。这样 rollout 阶段就带了初步评分,聚合/分析时可直接用。
    # 〔讲解〕 入参 result=本条轨迹的结果字典;item=题目原始数据(含标准答案/rubric)。返回:在 result 前面拼上 auto_judge 字段的新字典。
    async def add_auto_judge(self, result, item):
        prediction = result.get("prediction", "") or ""
        # 〔讲解〕 healthbench 打分需要知道实际问给模型的多轮 prompt,这里把它带给打分器。
        if self.task in ["healthbench"]: item["actual_queried_prompt_messages"] = self.actual_queried_prompt_messages
        evaluator = get_evaluator(self.task)

        # 〔讲解〕 三条路径都最终回填 auto_judge:① 预测为空→以 error 模式打分;② 正常→用预测打分;③ 打分过程报错→兜底以 error 模式打分。无论如何不让异常冒泡中断整条轨迹。
        try:
            if not prediction:
                judge_result = await evaluator.compute_score(
                    prediction="", item=item, error=True, err_msg="No prediction found"
                )
                # 〔讲解〕 {"auto_judge": ..., **result}:把判分放在最前,再展开原 result 各字段(键不重复故无覆盖)。
                result = {"auto_judge": judge_result, **result}
                return result
            judge_result = await evaluator.compute_score(prediction, item)
            result = {"auto_judge": judge_result, **result}
        except Exception as e:
            judge_result = await evaluator.compute_score(
                prediction="", item=item, error=True, err_msg=f"Error: {str(e)}"
            )
            result = {"auto_judge": judge_result, **result}
        return result

    # 〔讲解〕 ★核心方法★:对单道题跑完整条 ReAct 轨迹。这是 rollout 的主循环所在。
    # 〔讲解〕 入参 data=单题数据(data['item'] 含 question/prompt 等);model=本次用的模型 id。返回:含 prediction/messages/cost/auto_judge 等的结果字典。
    async def _run(self, data: dict, model: str, context_manager=None, **kwargs) -> dict:
        self.model = model
        question = data['item']['question']
        task_instruction = get_task_instruction(self.task)
        # 〔讲解〕 若该任务有附加指令,就拼到题目后面;无则空串。注意前面加了两个换行做分隔。
        TASK_INSTRUCTION = f"\n\n{task_instruction}" if task_instruction else ""
        # 〔讲解〕 按任务构造初始消息列表(system + user):
        # 〔讲解〕 healthbench 把基础 system 与 deep-research system(SYSTEM_PROMPT_DR)拼接,且用户轮直接用数据自带的多轮 prompt;
        # 〔讲解〕 researchrubrics 同样拼接 system,但用户轮是单条 question;其余任务用基础 system + (question+附加指令)。
        if self.task == "healthbench":
            messages = [{"role": "system", "content": self.system_prompt + "\n\n" + SYSTEM_PROMPT_DR}] + data['item']["prompt"]
            # 〔讲解〕 深拷贝留存「实际问给模型的 prompt」,后面 add_auto_judge 打分要用(浅拷贝会被后续 messages 变动污染)。
            self.actual_queried_prompt_messages = copy.deepcopy(messages)
        elif self.task == "researchrubrics":
            messages = [{"role": "system", "content": self.system_prompt + "\n\n" + SYSTEM_PROMPT_DR}, {"role": "user", "content": question}]
        else:
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": question+TASK_INSTRUCTION}]
        # 〔讲解〕 debug_data 是「轨迹全记录」:每步 token 数、各工具调用次数、逐条完整消息——既供调试也供后续算成本/分析。
        debug_data = {
            "token_lengths_each_step": [],
            "tool_usage": {},
            "full_messages": [],
        }
        # 〔讲解〕 把初始 system/user 消息也登记进 full_messages,标 iteration=0(尚未进入循环)。
        if not debug_data["full_messages"]:
            for msg in messages:
                debug_data["full_messages"].append({**msg, "iteration": 0})

        iteration = 0
        # 〔讲解〕 termination=轨迹结束原因标签(error/no_tool_call/max_exceed/answer),初始 unknown,循环里据情况改写。
        termination = "unknown"
        # 〔讲解〕 记开始时刻,结尾算耗时;用事件循环时钟而非 wall-clock,适配 async。
        t_start = asyncio.get_event_loop().time()
        # 〔讲解〕 ReAct 主循环:每轮 = 让模型说一次话;它若调工具就执行并把结果喂回,再进下一轮;它若直接作答或资源耗尽就跳出。
        while True:
            iteration += 1
            # 〔讲解〕 第①步:把当前完整对话发给模型,带上工具描述让它可以决定调工具还是直接答。
            response = await self.call_server(messages, tool_description=self.tool_description)
            store_token_length(debug_data, iteration, response)

            ### Error
            # 〔讲解〕 call_server 失败会返回字符串 "Server error";此时把它当作一条 assistant 消息记下,标记 error 并终止。
            if isinstance(response, str):
                error_message = {"role": "assistant", "content": response}
                messages.append(error_message)
                debug_data["full_messages"].append({**error_message, "iteration": iteration, "raw_response": response})
                termination = "error"
                break

            # add content, reasoning, tool_calls, and tool_responses
            # 〔讲解〕 第②步:解析模型这轮回复,拆出正文 content 与可选的思维链 reasoning,组装成标准 assistant 消息字典。
            assistant_message = response.choices[0].message
            content = (assistant_message.get("content") or "").strip()
            message_dict = {"role": "assistant", "content": content}
            reasoning_content = (assistant_message.get("reasoning_content") or "").strip()
            if reasoning_content: message_dict["reasoning"] = reasoning_content
            # Store raw response for debugging
            # 〔讲解〕 把模型原始返回(序列化后)留底,排查问题时能看到模型到底返回了什么。
            raw_response = response.model_dump() if hasattr(response, 'model_dump') else str(response)

            ### Tool Call
            # 〔讲解〕 第③步:看模型是否要调工具。tool_calls 非空=进入「执行工具」分支;为空=进入「直接作答/无调用」分支。
            tool_calls = assistant_message.get("tool_calls") or []
            if tool_calls:
                # 〔讲解〕 把模型给出的工具调用规整成标准结构;注意 [:1] —— 本框架强制「一次只执行一个工具」(即使模型一口气要求多个,也只取第一个)。
                tool_calls_list = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": self.sanitize_tool_name(tc.function.name),
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in tool_calls[:1] ### Only call one tool at a time
                ]
                message_dict["tool_calls"] = tool_calls_list
                messages.append(message_dict)
                debug_data["full_messages"].append({**message_dict, "iteration": iteration, "raw_response": raw_response})

                # 〔讲解〕 真正执行这个工具(同样 [:1] 只取第一个),把工具返回作为 role=tool 的消息追加进对话,供下一轮模型阅读。
                for tool_call in tool_calls[:1]: ### Only call one tool at a time
                    tool_name = self.sanitize_tool_name(tool_call.function.name)
                    try:
                        # 〔讲解〕 工具参数是模型生成的 JSON 字符串,parse 成 dict 再交给 custom_call_tool 执行;结果转回 JSON 字符串塞进对话。
                        tool_args = json.loads(tool_call.function.arguments)
                        result = await self.custom_call_tool(tool_name, tool_args)
                        result = json.dumps(result, ensure_ascii=False)
                        # 〔讲解〕 统计该工具被调用次数(+1),用于后续分析模型的工具使用习惯。
                        debug_data["tool_usage"][tool_name] = debug_data["tool_usage"].get(tool_name, 0) + 1
                    except Exception as e:
                        # 〔讲解〕 参数不是合法 JSON 等情况:不崩溃,而是把报错文本当作工具返回喂回模型,让它自己纠正下一步。
                        print(e)
                        result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field'
                    tool_message = {"role": "tool", "tool_call_id": tool_call.id, "name": tool_name, "content": result}
                    messages.append(tool_message)
                    debug_data["full_messages"].append({**tool_message, "iteration": iteration})
            else:
                # 〔讲解〕 没有工具调用:把这条 assistant 消息记下。
                messages.append(message_dict)
                debug_data["full_messages"].append({**message_dict, "iteration": iteration, "raw_response": raw_response})
                # 〔讲解〕 终止判定:若有正文内容(模型给了答案)就正常收尾 no_tool_call;
                # 〔讲解〕 特例——若「没正文 也 没工具调用」(content 空 且 reasoning 空)也终止,避免空转死循环;
                # 〔讲解〕 但若「只有思维链、没正文、也没工具调用」(content 空但 reasoning 非空),则不终止,让循环再给模型一次机会继续输出。
                if content or not reasoning_content:
                    termination = "no_tool_call"
                    break

            # 〔讲解〕 第④步:每轮末尾数一次当前对话总 token,用于下面判断是否超限。
            token_count = self.count_tokens(messages, tool_description=self.tool_description)

            ### Exceed Token Limit or max LLM calls
            # 〔讲解〕 触发「资源耗尽」收尾:token 超上限 或 调用次数到顶。此时不再让模型自由调工具,而是逼它基于已有信息直接给最终答案。
            if token_count >= self.max_tokens or iteration >= self.max_llm_call_per_run:
                # 〔讲解〕 把最后一条(通常是超长的工具返回)内容替换成截断提示,既省 token 又告知模型「材料被截断了」。
                messages[-1]['content'] = TRUNCATED_MESSAGE
                debug_data["full_messages"][-1]["content"] = TRUNCATED_MESSAGE
                # 〔讲解〕 追加一条 user 指令(FINAL_MESSAGE)明确要求停止调工具、直接作答。
                messages.append({"role": "user", "content": FINAL_MESSAGE})
                debug_data["full_messages"].append({"role": "user", "content": FINAL_MESSAGE, "iteration": iteration})
                # 〔讲解〕 再调一次模型,这次 no_tool_calls=True 禁用工具,拿它的最终答案。
                response = await self.call_server(messages, no_tool_calls=True, tool_description=self.tool_description)
                store_token_length(debug_data, iteration, response)
                raw_response = response.model_dump() if hasattr(response, 'model_dump') else str(response)
                # 〔讲解〕 这次逼答也可能失败(返回字符串);分两种情况组装最终 assistant 消息。
                if isinstance(response, str): message_dict = {"role": "assistant", "content": response}
                else:
                    assistant_message = response.choices[0].message
                    content = (assistant_message.get("content") or "").strip()
                    message_dict = {"role": "assistant", "content": content}
                    reasoning_content = (assistant_message.get("reasoning_content") or "").strip()
                    if reasoning_content: message_dict["reasoning"] = reasoning_content

                messages.append(message_dict)
                debug_data["full_messages"].append({**message_dict, "iteration": iteration, "raw_response": raw_response})

                termination = "max_exceed"
                break

        # 〔讲解〕 循环结束:算总耗时;把最后一条消息内容作为本题的最终预测答案。
        time = asyncio.get_event_loop().time() - t_start
        prediction = messages[-1]['content']
        # 〔讲解〕 若预测里识别出答案标记,把 termination 改写为 "answer"(比 no_tool_call 更精确地反映「确实给了答案」)。
        if self.answer_in_content(prediction): termination = "answer"
        # 〔讲解〕 打包整条轨迹的结果:题目、原始数据、预测、结束原因、耗时、真实货币成本、完整对话、调试记录。这就是落盘 json 的主体。
        result = {
            "question": data["item"].get("question", ""),
            "instance": data["item"],
            "prediction": prediction,
            "termination": termination,
            "time": time,
            # 〔讲解〕 按 debug_data 里各步 token 数 + 模型定价,折算这条轨迹的真实花费(项目偏好用货币而非裸 token 计成本)。
            "cost": compute_rollout_cost(debug_data, self.model_name, self.task),
            "messages": messages,
            "debug_data": debug_data,
        }
        # 〔讲解〕 最后挂上自动判分,得到带 auto_judge 的完整结果再返回给上游调度器。
        result = await self.add_auto_judge(result, data["item"])
        return result

    # 〔讲解〕 工具执行的薄封装:按工具名从 tool_map 取实例并 await 调用,找不到则返回错误文本。
    # 〔讲解〕 关键细节:把 tool_args 整个再塞进它自己的 "params" 键——这是为兼容工具实现里通过 params 取参的约定。
    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in self.tool_map:
            tool_args["params"] = tool_args
            raw_result = await self.tool_map[tool_name].call(tool_args, **kwargs)
            result = raw_result
            return result
        else: return f"Error: Tool {tool_name} not found"
