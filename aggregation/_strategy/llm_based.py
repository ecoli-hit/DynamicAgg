#〔讲解〕本文件是 AggAgent「第二阶段:聚合」里最轻量的两种聚合策略实现。
#〔讲解〕上游:rollout/ 已经为每道题跑了 N 条独立 ReAct 搜索轨迹(每条含 messages/prediction/auto_judge)。
#〔讲解〕本文件:把同一道题里 k 条轨迹「合并成一个最终答案」,有两种做法:
#〔讲解〕   SolAgg = 直接把各轨迹的最终预测(prediction)丢给一个 LLM 让它整合裁决;
#〔讲解〕   SummAgg = 先把每条长轨迹用 LLM 压缩成简报(report),再整合(省 token,更聚焦)。
#〔讲解〕注意:这两种是「纯 LLM 一次性整合」,不会调工具读轨迹——那种「会调工具的聚合智能体」在 aggagent/ 里,不在本文件。
#〔讲解〕下游:整合出的答案交给 _compute_score(evaluation 判分)算对错,并落盘日志/统计用于 Sol@k / Summ@k 指标。
"""LLM-based aggregation strategies (SolAgg, SummAgg)."""

import json
import os
import re
import threading
import time
#〔讲解〕并发原语:用线程池并行处理多道题/多条轨迹摘要(都是网络 IO 密集,线程比进程合适);as_completed 让结果谁先完成谁先返回。
from concurrent.futures import ThreadPoolExecutor, as_completed
#〔讲解〕combinations:从 N 条轨迹里枚举所有 C(N,k) 个「选 k 条」的组合,这是 Sol@k 指标的核心采样单位。
from itertools import combinations

#〔讲解〕litellm 是统一 LLM 调用层,一套 API 兼容 OpenAI / 本地 vLLM 等多种后端(下面靠 model 前缀切换)。
import litellm
from tqdm import tqdm

#〔讲解〕从基类引入:Strategy(策略基类,提供 is_correct/get_n/rng/成本拆分等)、MODEL_COSTS(各模型单价表)、
#〔讲解〕CostBreakdown(rollout/tool/aggregation 三段成本的可加结构)、_compute_score(调评测判分)。
from .base import Strategy, MODEL_COSTS, CostBreakdown, _compute_score


#〔讲解〕REPORT_PROMPT:SummAgg 专用。要求 LLM 把一条「问题求解轨迹」蒸馏成结构化简报(规划/方法/最终推理),
#〔讲解〕并强制只能用轨迹内信息、用 <report></report> 包裹——便于后面正则抽取。下方三引号内是发给模型的原文,勿改。
REPORT_PROMPT = """Given the following problem-solving trajectory:
{traj}

Your task is to distill this trajectory into a concise yet sufficiently detailed problem-solving report.
You must only use the information contained within the provided trajectory — no additional or external information is allowed.

The report must include:

1. **Solution Planning**: Identify how the main problem is decomposed into subproblems, and describe the sequence and dependency relationships among those subproblems.
2. **Solution Methods**: For each subproblem, indicate which tools were invoked to solve it, the parameters used in those tool calls, and any resulting partial answers that contributed directly or indirectly to progress toward the final answer.
   _Do not repeat the full output of the tools; only include the specific fragments of tool results that were essential in deriving the subanswers._
3. **Final Reasoning**: Clearly outline the reasoning process by which the subproblems and their associated subanswers led to the derivation of the final answer.

Additional requirements:
- The report must remain **concise** and **focused**.
- Remove any content unrelated to problem-solving or any ineffective tool calls.
- Ensure the final report has clear logical structure, with each step traceable and analyzable.

Finally, present the complete report in **Markdown format**, and wrap the entire report content within <report> </report> tags.
""".strip()

#〔讲解〕INTEGRATE_PROMPT:整合阶段「短答案」模式(browsecomp 等有唯一确定答案的任务)。
#〔讲解〕核心约束:把多份报告当成多个独立团队的方案,要求 LLM 批判性择优、必须选一个、不许合并/泛化答案、不许调外部工具、用 <answer></answer> 包裹。
INTEGRATE_PROMPT = """You are tasked with solving the question: {question}.

Multiple independent teams have provided detailed process reports describing their approaches to solving this problem. As the final analyst, your role is to consolidate these reports, carefully examine the problem-solving methods they contain, and identify the key information obtained in each.

Your goal is to produce a final answer that perfectly resolves the question. Note that some of the reports may contain inconsistencies — you must critically evaluate which report(s) are reasonable and trustworthy.

If multiple reports reach the same conclusion, this increases the likelihood that the conclusion is correct; however, this is not guaranteed. You must still carefully verify and reflect to ensure that the final selected answer is truly the most accurate possible.

Wrap your final answer in <answer> </answer> tags and provide a brief explanation for your final answer in <explanation> </explanation> tags.

Important:
- Every question has a definitive, certain answer.
- You are not allowed to decline answering on the grounds of uncertainty.
- For any report that does not provide a clear and definite final answer, its confidence level should be significantly reduced.
- You must ultimately select one report as having the most correct answer.
- You are not allowed to call or use any external tools for verification. You must rely solely on the information already provided, conduct in-depth analysis, and then produce the final answer.
- You are not allowed to merge multiple different answers, nor are you allowed to produce an overly broad answer that attempts to encompass all candidate answers — such an answer should be eliminated first.
- You do not need to restate or summarize the reports; instead, provide a short-form answer that directly answers the question.

Below is the content of these reports:
""".strip()

#〔讲解〕INTEGRATE_PROMPT_REPORT:整合阶段「长报告」模式(healthbench / researchrubrics 等开放式、按 rubric 评分的任务)。
#〔讲解〕与上者区别:产出的是带引用的综合报告(用 <cite url="...">),要求只引用报告里出现过的真实 URL、禁止编造。
INTEGRATE_PROMPT_REPORT = """You are tasked with solving the task: {question}.

Multiple independent teams have provided detailed process reports describing their approaches to solving this problem. As the final analyst, your role is to consolidate these reports, carefully examine the problem-solving methods they contain, and identify the key information obtained in each.

Your goal is to produce a final report that perfectly resolves the problem. Note that some of the reports may contain inconsistencies — you must critically evaluate which report(s) are reasonable and trustworthy.

If multiple reports reach the same conclusion, this increases the likelihood that the conclusion is correct; however, this is not guaranteed. You must still carefully verify and reflect to ensure that the final report is truly the most accurate possible.

Produce a comprehensive final report that directly addresses the task. Cite every nontrivial claim using <cite url="...">...</cite> tags drawn only from URLs mentioned in the provided reports — never fabricate URLs or content.

Important:
- You are not allowed to decline answering on the grounds of uncertainty.
- For any report that does not provide a clear and definite solution, its confidence level should be significantly reduced.
- You are not allowed to call or use any external tools for verification. You must rely solely on the information already provided, conduct in-depth analysis, and then produce the final report.

Below is the content of these reports:
""".strip()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

#〔讲解〕_get_content:从一条 message 里安全取文本。难点在于「content 字段格式不统一」——
#〔讲解〕有的后端是纯字符串,有的是 OpenAI 多模态格式 [{"type":"text","text":...}]。这里两种都兼容。
def _get_content(message: dict, key: str = "content") -> str:
    """Extract content from a message dict (handles string or list formats)."""
    value = message.get(key, "")
    if isinstance(value, str):
        return value
    #〔讲解〕走到这里说明是 list 格式,取第一个元素的 text 字段。
    if isinstance(value, list) and value:
        if isinstance(value[0], dict):
            #〔讲解〕`or ""` 防 text 为 None;后面统一返回字符串不返回 None。
            text = value[0].get("text") or ""
            #〔讲解〕仅当取的是 content 字段时,补上工具调用/响应的可读前缀,让轨迹文本更清晰。
            if key == "content":
                recipient = message.get("recipient")
                name = message.get("name")
                #〔讲解〕有 name → 这是某工具的返回;有 recipient → 这是调用某工具的请求。二选一加标签。
                if name:
                    text = f"[Tool Response: {name}]\n{text}"
                elif recipient:
                    text = f"[Tool Call: {recipient}]\n{text}"
            return text
    return ""


#〔讲解〕_construct_interaction:把一条轨迹的 messages 列表渲染成「人能读、LLM 也好读」的对话文本,供 REPORT_PROMPT 蒸馏。
#〔讲解〕按 role 分别格式化:system/developer 直接跳过(对解题无信息),其余拼成 **User/Tool/Assistant** 段落。
def _construct_interaction(messages: list[dict]) -> str:
    """Convert a messages list into a readable trajectory string."""
    interaction = ""
    for r in messages:
        #〔讲解〕系统/开发者提示词不进轨迹文本——它们是固定模板,不含本题的求解信息。
        if r["role"] in ("developer", "system"):
            continue
        elif r["role"] == "user":
            interaction += f"**User:**\n{r['content']}\n\n"
        elif r["role"] == "tool":
            interaction += f"**Tool Response:**\n{r['content']}\n\n"
        elif r["role"] == "assistant":
            interaction += "**Assistant:**\n"
            #〔讲解〕assistant 一条消息可能同时含「思考/正文/工具调用」多个部分,逐一展开。
            #〔讲解〕reasoning_content 与 reasoning 是不同后端对「思维链」的两种字段名,都尝试取。
            if r.get("reasoning_content") is not None:
                interaction += f"*Thinking:* {r['reasoning_content']}\n"
            if r.get("reasoning") is not None:
                interaction += f"*Thinking:* {r['reasoning']}\n"
            if r.get("content") is not None:
                interaction += f"*Content:* {r['content']}\n"
            if r.get("tool_calls") is not None:
                #〔讲解〕只取第一个工具调用(本框架每步至多一个 tool_call)。
                func = r["tool_calls"][0]["function"]
                #〔讲解〕func 可能是 dict,也可能是 SDK 的对象;不是 dict 就转成 dict 再 json 序列化,保证可打印。
                if not isinstance(func, dict):
                    func = func.to_dict()
                interaction += f"*Tool Call:* {json.dumps(func)}\n"
            interaction += "\n"
    return interaction.strip()


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

#〔讲解〕_generate_compact_summary:SummAgg 的「压缩单条轨迹」函数。
#〔讲解〕入参 messages=一条轨迹;返回 (简报文本, token 用量)。用量用于后面折算聚合阶段成本。
def _generate_compact_summary(
    messages: list[dict], model: str = "gpt-4o-mini", api_base: str | None = None
) -> tuple[str, dict]:
    """Summarize a trajectory using REPORT_PROMPT. Returns (summary, usage)."""
    #〔讲解〕先把 messages 渲染成可读轨迹,再填进 REPORT_PROMPT 模板。
    trajectory = _construct_interaction(messages)
    prompt = REPORT_PROMPT.format(traj=trajectory)

    #〔讲解〕默认走本地 vLLM 后端,故给 model 加 "hosted_vllm/" 前缀(litellm 据前缀路由)。
    kwargs: dict = {
        "model": "hosted_vllm/" + model,
        "messages": [{"role": "user", "content": prompt}],
    }
    #〔讲解〕若模型是 OpenAI 系的 gpt-*(但排除开源的 gpt-oss),改走 "openai/" 路由 + 注入 API key + 限制最大输出。
    if "gpt" in model and "oss" not in model:
        kwargs["model"] = "openai/" + model
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        kwargs["max_tokens"] = 10000
    if api_base:
        kwargs["api_base"] = api_base

    content = None
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    max_retries = 3
    #〔讲解〕带重试的 LLM 调用:网络/限流偶发失败很常见,失败就退避重试,最多 3 次。
    for attempt in range(max_retries):
        try:
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content
            #〔讲解〕尽量从响应里抠真实 token 用量;getattr(..,0) or 0 双重兜底防字段缺失或为 None。
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                }
            #〔讲解〕拿到非空内容即视为成功,跳出重试循环。
            if content is not None:
                break
        except Exception as e:
            print(f"[generate_compact_summary] Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")
            #〔讲解〕线性退避:第 1 次失败睡 1s,第 2 次睡 2s;最后一次失败不再睡(直接退出)。
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
    #〔讲解〕重试耗尽仍无内容 → 返回空简报(调用方会跳过它,不计入 reports)。
    if not content:
        return "", usage
    #〔讲解〕从模型输出里抽 <report>...</report> 之间的正文;DOTALL 让 . 跨行匹配。抽到就用抽到的,抽不到就退回整段原文。
    match = re.search(r"<report>(.*?)</report>", content, re.DOTALL)
    if match:
        return match.group(1).strip(), usage
    return content, usage


#〔讲解〕_integrate_and_answer:整合阶段核心。把若干份报告(SolAgg 用原始 prediction / SummAgg 用蒸馏 report)
#〔讲解〕拼进整合 prompt,调一次 LLM 产出最终答案。返回 (答案文本, token 用量, 实际发出的 prompt)——prompt 也返回是为了落日志可复现。
def _integrate_and_answer(
    question: str,
    reports: list[str],
    model: str = "gpt-4o-mini",
    api_base: str | None = None,
    use_report_prompt: bool = False,
) -> tuple[str, dict, str]:
    """Integrate reports and produce a final answer. Returns (content, usage, prompt)."""
    #〔讲解〕据任务类型选模板:开放式报告任务用 REPORT 版(带引用、产长报告),短答案任务用普通版(选唯一答案)。
    prompt_template = INTEGRATE_PROMPT_REPORT if use_report_prompt else INTEGRATE_PROMPT
    prompt = prompt_template.format(question=question)
    #〔讲解〕把每份报告编号(从 1 开始)追加成 "## Report i" 小节,拼到 prompt 末尾。
    for i, report in enumerate(reports, 1):
        prompt += f"\n\n## Report {i}\n{report}"

    #〔讲解〕以下后端路由/重试逻辑与 _generate_compact_summary 完全一致(同一套 litellm 调用范式)。
    kwargs: dict = {
        "model": "hosted_vllm/" + model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if api_base:
        kwargs["api_base"] = api_base
    if "gpt" in model and "oss" not in model:
        kwargs["model"] = "openai/" + model
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        kwargs["max_tokens"] = 10000

    content = None
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                }
            if content is not None:
                break
        except Exception as e:
            print(f"[integrate_and_answer] Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))

    #〔讲解〕清掉控制字符(不可见的 \x00-\x1f 等),否则后面把答案写进 JSONL 日志时会破坏 JSON 序列化。
    # Strip control characters that break JSON serialization
    if content:
        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content)
    return content or "", usage, prompt


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------

#〔讲解〕SolAgg:Sol@k 策略本体,继承 Strategy 基类。
#〔讲解〕角色:对每道题,从 N 条轨迹中抽样若干个「k 条组合」,每个组合把各自最终预测交 LLM 整合→判分→统计正确率。
#〔讲解〕Sol@k = 这些组合判对的平均比例(类似 pass@k 的「整合版」)。SummAgg 继承它,只改成先压缩再整合。
class SolAgg(Strategy):
    """
    Sol@k (Solution Integration) strategy.

    For each sampled C(N,k) combination, collects predictions from each run,
    integrates them using an LLM, and checks correctness.
    """

    name = "SolAgg"

    #〔讲解〕构造:model=聚合用 LLM;task 决定判分器与 prompt 模板;max_workers=并行处理题目的线程数;
    #〔讲解〕output_dir=日志/统计落盘目录;skip_score=只整合不判分(省钱/调试);resume=断点续跑(读已有日志跳过已完成题)。
    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        task: str = "browsecomp",
        max_workers: int = 10,
        output_dir: str | None = None,
        skip_score: bool = False,
        resume: bool = False,
        **kwargs,
    ):
        super().__init__(task=task)
        self.model = model
        self.api_base = api_base
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.skip_score = skip_score
        self.resume = resume
        #〔讲解〕compact 是区分两策略的唯一开关:SolAgg=False(直接用 prediction),SummAgg 在子类把它置 True(先蒸馏 report)。
        self.compact = False  # overridden by SummAgg

        #〔讲解〕日志缓冲:多线程并发写,先攒在内存列表,定期 flush 到文件;_log_lock 保证线程安全。
        self._log_entries: list[dict] = []
        self._log_lock = threading.Lock()
        self._current_log_file: str | None = None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    #〔讲解〕_write_log:线程安全地把一条日志 append 进内存缓冲(还没落盘)。
    def _write_log(self, entry: dict):
        with self._log_lock:
            self._log_entries.append(entry)

    #〔讲解〕_flush_logs:把内存缓冲一次性追加写入 jsonl 文件并清空。无输出目录/无缓冲则直接返回。
    def _flush_logs(self):
        if not self.output_dir or not self._current_log_file:
            return
        with self._log_lock:
            if not self._log_entries:
                return
            #〔讲解〕以追加模式("a")写,每条一行 JSON;ensure_ascii=False 保留中文等非 ASCII 原样可读。
            with open(self._current_log_file, "a") as f:
                for entry in self._log_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._log_entries.clear()

    # ------------------------------------------------------------------
    # Cost helper
    # ------------------------------------------------------------------

    #〔讲解〕_calculate_llm_cost:按 token 用量 × 单价表算「聚合阶段」这一次 LLM 调用的钱(美元)。
    #〔讲解〕单价表里没有该模型(如本地自托管)就记 0;价格是「每百万 token」,故最后除以 1_000_000。
    def _calculate_llm_cost(self, usage: dict) -> float:
        """Calculate aggregation LLM cost from token usage."""
        if self.model not in MODEL_COSTS:
            return 0.0
        input_cost_per_m, output_cost_per_m = MODEL_COSTS[self.model]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        return (prompt_tokens * input_cost_per_m + completion_tokens * output_cost_per_m) / 1_000_000

    # ------------------------------------------------------------------
    # Per-problem processing
    # ------------------------------------------------------------------

    #〔讲解〕_process_single_problem:处理「一道题」的全流程(被线程池并行调用)。
    #〔讲解〕入参:problem_id;run_results=该题 N 条轨迹;sampled_combos=已抽样的若干 k 元组合(每个是轨迹下标元组)。
    #〔讲解〕返回 (该题在各组合上的平均得分, 平均聚合成本)。
    def _process_single_problem(
        self,
        problem_id: str,
        run_results: list[dict],
        sampled_combos: list[tuple],
    ) -> tuple[float, float]:
        """Process one problem over all sampled combos. Returns (score, avg_llm_cost)."""
        question = run_results[0].get("question", "")
        instance = run_results[0].get("instance", {})
        # item = {"question": question, **instance}
        #〔讲解〕item 是传给判分器的「题目+元信息」打包(展开整条轨迹记录,含答案/rubric 等判分所需字段)。
        item = {"question": question, **instance}

        #〔讲解〕healthbench 任务的「问题」其实是一段多轮对话(prompt 列表),这里把它拼成字符串当作 question 喂给整合 prompt。
        if self.task == "healthbench":
            question = "\n".join(
                str({"role": r["role"], "content": r["content"]})
                for r in run_results[0].get("instance", {}).get("prompt", [])
            )

        pass_count = 0
        total_llm_cost = 0.0

        #〔讲解〕逐个组合处理:每个组合是「从 N 条里选出的 k 条轨迹」,要独立整合出一个答案并判分。
        for combo in sampled_combos:
            #〔讲解〕按下标元组取出这 k 条轨迹记录。
            combo_runs = [run_results[i] for i in combo]
            combo_llm_cost = 0.0
            #〔讲解〕记录这 k 条「各自单独」的对错与判分,落日志用——可对比「单条 vs 整合后」谁更对(分析增益来源)。
            combo_correct = [self.is_correct(r) for r in combo_runs]
            combo_judgements = [r.get("auto_judge") for r in combo_runs]

            t_start = time.time()
            reports = []

            #〔讲解〕分叉:compact=True(SummAgg)先把每条轨迹并行压成 report;compact=False(SolAgg)直接拿现成 prediction。
            if self.compact:
                #〔讲解〕内部小函数:压缩单条轨迹;无 messages 则返回空+零用量。
                def _summarize(r):
                    messages = r.get("messages", [])
                    if messages:
                        return _generate_compact_summary(
                            messages, model=self.model, api_base=self.api_base
                        )
                    return None, {"prompt_tokens": 0, "completion_tokens": 0}

                #〔讲解〕k 条轨迹并行摘要(线程数=k);谁返回有效简报就收进 reports,并累加其压缩成本。
                with ThreadPoolExecutor(max_workers=len(combo_runs)) as pool:
                    futs = [pool.submit(_summarize, r) for r in combo_runs]
                    for fut in futs:
                        summary, usage = fut.result()
                        if summary:
                            reports.append(summary)
                            combo_llm_cost += self._calculate_llm_cost(usage)
            else:
                #〔讲解〕SolAgg 路径:直接收集每条轨迹的最终预测,空预测跳过。
                for r in combo_runs:
                    prediction = r.get("prediction", "")
                    if prediction:
                        reports.append(prediction)

            #〔讲解〕只要这个组合凑出了至少一份报告,就做整合 + 判分 + 记日志。
            if reports:
                #〔讲解〕一次 LLM 整合调用:据任务选 short-answer / report 模板(healthbench/researchrubrics 走 report 版)。
                integrated_response, usage, integrate_prompt = _integrate_and_answer(
                    question,
                    reports,
                    model=self.model,
                    api_base=self.api_base,
                    use_report_prompt=self.task in ("healthbench", "researchrubrics"),
                )
                elapsed = time.time() - t_start
                #〔讲解〕整合这一步的成本叠加到该组合成本(SummAgg 的话此时已含前面的压缩成本)。
                combo_llm_cost += self._calculate_llm_cost(usage)

                #〔讲解〕skip_score 模式只产答案不判分(省判分模型的钱/时间);否则调 _compute_score 判对错并累计通过数。
                if self.skip_score:
                    judge_result = None
                    is_correct = None
                else:
                    judge_result = _compute_score(integrated_response, item, self.task)
                    is_correct = self.is_correct({"auto_judge": judge_result})
                    #〔讲解〕is_correct 是 True/False,这里当 1/0 累加,统计该题判对的组合数。
                    pass_count += is_correct

                #〔讲解〕落一条完整日志:含整合答案、判分、成本/耗时、实际 prompt、组合下标、以及组合内各条单独对错——为后续分析留全证据。
                log_entry = {
                    "metadata": self._current_metadata,
                    "auto_judge": judge_result,
                    "question": question,
                    "instance": instance,
                    "prediction": integrated_response,
                    "is_correct": is_correct,
                    "aggregation_cost": combo_llm_cost,
                    "aggregation_time": elapsed,
                    "prompt": integrate_prompt,
                    "combo": list(combo),
                    "combo_correct": combo_correct,
                    "combo_judgements": combo_judgements,
                }
                self._write_log(log_entry)

            total_llm_cost += combo_llm_cost

        #〔讲解〕该题得分 = 判对组合数 / 组合数;平均成本同理。无组合则记 0,避免除零。
        score = pass_count / len(sampled_combos) if sampled_combos else 0.0
        avg_llm_cost = total_llm_cost / len(sampled_combos) if sampled_combos else 0.0
        return score, avg_llm_cost

    # ------------------------------------------------------------------
    # calculate_at_k
    # ------------------------------------------------------------------

    #〔讲解〕calculate_at_k:整个策略的对外入口。给定全部题目的 N 条轨迹结果和 k,算出 Sol@k(或 Summ@k)指标与总成本。
    #〔讲解〕流程:确定 N→枚举并抽样组合→准备日志/断点续跑→线程池并行跑各题→汇总得分与三段成本→写统计文件。
    def calculate_at_k(
        self, results: dict[str, list[dict]], k: int
    ) -> tuple[float, float]:
        if not results:
            return (0.0, 0.0)

        #〔讲解〕N = 每题的轨迹条数(基类据数据推断);k 不能超过 N。
        n = self.get_n(results)
        if k > n:
            raise ValueError(f"k ({k}) cannot be greater than N ({n})")

        #〔讲解〕枚举所有 C(N,k) 组合,但最多只抽 3 个(max_combos=3),否则组合爆炸、LLM 调用成本不可控。
        all_combos = list(combinations(range(n), k))
        max_combos = 3
        if len(all_combos) > max_combos:
            #〔讲解〕用基类带种子的 rng 抽样,保证不同运行可复现同一批组合。
            sampled_combos = self.rng.sample(all_combos, max_combos)
        else:
            sampled_combos = all_combos

        #〔讲解〕只保留「恰好有 N 条轨迹」的题,丢掉残缺题(避免组合下标越界、保证各题可比)。
        valid_problems = [(pid, rr) for pid, rr in results.items() if len(rr) == n]

        #〔讲解〕从任一题的轨迹里取 rollout 阶段的 metadata,叠加本次聚合配置(策略名/k/聚合模型),作为本批实验的统一元信息写进每条日志与统计。
        rollout_metadata = next(iter(results.values()))[0].get("metadata", {})
        self._current_metadata = {
            **rollout_metadata,
            "strategy": self.name,
            "k": k,
            "aggregation_model": self.model,
        }

        # Setup log file; load existing entries when resuming
        #〔讲解〕准备日志文件 + 断点续跑:把已完成题按 question 索引进 existing,后面跳过它们只补未完成的。
        existing: dict[str, list[dict]] = {}
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            #〔讲解〕文件名前缀据 compact 区分 summagg/solagg,k 入名,便于不同 k、不同策略的结果并存。
            label_lower = "summagg" if self.compact else "solagg"
            self._current_log_file = os.path.join(
                self.output_dir, f"{label_lower}_logs_k{k}.jsonl"
            )
            #〔讲解〕resume 且日志已存在 → 逐行读回旧记录,按 question 聚合(同一题可能有多条组合记录)。
            if self.resume and os.path.exists(self._current_log_file):
                with open(self._current_log_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        existing.setdefault(entry["question"], []).append(entry)
            #〔讲解〕没读到任何旧记录(非续跑或文件空)→ 用 "w" 清空/新建文件,从头开始写。
            if not existing:
                open(self._current_log_file, "w").close()

        problem_scores: list[float] = []
        problem_breakdowns: list[CostBreakdown] = []
        label = "SummAgg" if self.compact else "SolAgg"

        # Recover from existing log entries
        #〔讲解〕第一步:对已在日志里的题,直接从旧记录恢复得分与成本,不再重跑(省钱/省时,断点续跑的核心)。
        new_valid_problems = []
        for pid, rr in valid_problems:
            question = rr[0].get("question", "") if rr else ""
            if question in existing:
                entries = existing[question]
                #〔讲解〕该题旧记录里各组合的 is_correct(可能为 None,用 or 0 兜底)取平均,作为该题得分。
                combo_scores = [float(e.get("is_correct") or 0) for e in entries]
                problem_scores.append(sum(combo_scores) / len(combo_scores))
                #〔讲解〕聚合成本从旧日志直接平均取回。
                avg_cost = sum(e.get("aggregation_cost", 0.0) for e in entries) / len(entries)
                #〔讲解〕rollout/tool 两段成本日志里没存,需用当前轨迹重新拆分计算:逐组合累加再除组合数取平均。
                traj_breakdown = CostBreakdown()
                for combo in sampled_combos:
                    combo_runs = [rr[i] for i in combo]
                    traj_breakdown = traj_breakdown + self.calculate_combination_cost_breakdown(combo_runs)
                traj_breakdown = traj_breakdown / len(sampled_combos)
                #〔讲解〕拼出该题三段成本:rollout/tool 来自重算,aggregation 来自旧日志。
                problem_breakdowns.append(CostBreakdown(
                    rollout=traj_breakdown.rollout,
                    tool=traj_breakdown.tool,
                    aggregation=avg_cost,
                ))
            else:
                #〔讲解〕没跑过的题进 new_valid_problems,留给下面线程池真正去跑。
                new_valid_problems.append((pid, rr))

        if existing:
            print(f"Resuming: {len(existing)} already done, {len(new_valid_problems)} remaining")

        #〔讲解〕第二步:线程池并行处理所有「未跑过」的题,每题一个 future。
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_single_problem, pid, rr, sampled_combos): (pid, rr)
                for pid, rr in new_valid_problems
            }
            #〔讲解〕as_completed:谁先算完先处理其结果;tqdm 显示进度条。
            pbar = tqdm(as_completed(futures), total=len(futures), desc=f"{label}@{k}")
            for future in pbar:
                pid, rr = futures[future]
                score, llm_cost = future.result()
                problem_scores.append(score)

                #〔讲解〕同样为该题重算 rollout/tool 两段成本(各组合平均),聚合成本用本次实跑得到的 llm_cost。
                traj_breakdown = CostBreakdown()
                for combo in sampled_combos:
                    combo_runs = [rr[i] for i in combo]
                    traj_breakdown = traj_breakdown + self.calculate_combination_cost_breakdown(combo_runs)
                traj_breakdown = traj_breakdown / len(sampled_combos)

                problem_breakdowns.append(CostBreakdown(
                    rollout=traj_breakdown.rollout,
                    tool=traj_breakdown.tool,
                    aggregation=llm_cost,
                ))

                #〔讲解〕每完成 20 题落一次盘,防止中途崩溃丢失大量已算结果(配合 resume 可续跑)。
                if len(problem_scores) % 20 == 0:
                    self._flush_logs()

                #〔讲解〕进度条实时显示当前累计平均正确率(%)。
                running_avg = sum(problem_scores) / len(problem_scores) * 100
                pbar.set_postfix_str(f"{running_avg:.2f}%")

        #〔讲解〕循环结束把剩余缓冲日志全部落盘。
        self._flush_logs()

        #〔讲解〕最终指标 = 所有题得分的平均(即 Sol@k / Summ@k)。同时把每题得分缓存,供外部做显著性/方差分析。
        metric = sum(problem_scores) / len(problem_scores) if problem_scores else 0.0
        self._last_problem_scores = problem_scores

        #〔讲解〕把各题三段成本相加再除题数,得到「平均每题」的 rollout/tool/aggregation 成本拆分,缓存供外部读取。
        avg_breakdown = CostBreakdown()
        for b in problem_breakdowns:
            avg_breakdown = avg_breakdown + b
        if problem_breakdowns:
            avg_breakdown = avg_breakdown / len(problem_breakdowns)
        self._last_cost_breakdown = avg_breakdown

        #〔讲解〕写一份汇总统计 json:指标、题数、组合数、以及聚合成本的总额/每题/每组合三种口径(成本报告用真实货币而非 token)。
        if self.output_dir and problem_scores:
            n_probs = len(problem_scores)
            n_combos = len(sampled_combos)
            #〔讲解〕总聚合成本 = 平均每题聚合成本 × 题数。
            total_agg_cost = avg_breakdown.aggregation * n_probs
            stats = {
                "metadata": self._current_metadata,
                "metric_at_k": metric,
                "n_problems": n_probs,
                "combos_per_problem": n_combos,
                "total_combos": n_probs * n_combos,
                "aggregation_cost": {
                    "total": total_agg_cost,
                    "avg_per_problem": avg_breakdown.aggregation,
                    "avg_per_combo": total_agg_cost / (n_probs * n_combos) if n_combos else 0.0,
                },
            }
            stats_file = os.path.join(self.output_dir, f"{label_lower}_stats_k{k}.json")
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)

        #〔讲解〕返回 (指标, 平均每题总成本)。total 是 CostBreakdown 把 rollout+tool+aggregation 三段加总的属性。
        return (metric, avg_breakdown.total)


#〔讲解〕SummAgg:Summ@k 策略。完全复用 SolAgg 的全部流程,唯一差别是把 compact 打开——
#〔讲解〕即整合前先用 REPORT_PROMPT 把每条长轨迹压成简报,再交给 LLM 整合。目的:更省 token、信息更聚焦。
class SummAgg(SolAgg):
    """
    Summ@k (Compact Solution Integration) strategy.

    Same as Sol@k but first compresses each run's trajectory into a
    concise report using REPORT_PROMPT before integration.
    """

    name = "SummAgg"

    #〔讲解〕构造里只做一件事:先按父类初始化,再把 compact 置 True,从而切换到「先摘要后整合」路径。
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compact = True
