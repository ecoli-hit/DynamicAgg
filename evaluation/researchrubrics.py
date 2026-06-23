# 〔讲解〕 模块说明:
# 本文件是 evaluation/(判分环)里的一个 "Evaluator" 实现 —— ResearchRubrics 评分器。
# pipeline 上游:rollout/ + aggregation/aggagent/ 跑完后,会产出一份「最终答案文本」(prediction)。
# 本文件干的事:拿这份答案,用一个裁判大模型(LLM-as-judge)逐条对照「评分细则(rubric)」打分(每条 0/1 满足判定),
#               再按权重把各条聚合成总分和分轴(axis)分。
# 它移植(port)自一个叫 researchrubrics 的外部项目 —— 注释里反复出现的 "Ported from ..." 就是指那些原始 prompt 文件。
# 这是「评测/打分」环节,不参与生成答案,只负责给答案判分。

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# 〔讲解〕 base.py 里的 Evaluator 是所有评分器的抽象基类,提供了 async_complete(调裁判 LLM)、
#         extract_confidence 等通用能力;本类继承它,只补充 rubric 特有的打分逻辑。
from .base import Evaluator


# 〔讲解〕 把 .env 里的密钥/配置(如 OPENAI_API_KEY)读进环境变量,后面调 LLM 时要用。
load_dotenv()


# litellm._turn_on_debug()
# Ported from researchrubrics/src/prompts/system_prompt.txt
# 〔讲解〕 裁判 LLM 的 system prompt(系统人设)。核心规则:用二元判定(0.0 不满足 / 1.0 满足),
#         且对「负权重 rubric」(描述不良行为的条目)做了特殊说明 —— 这是本评分体系的关键设计,见 prompt 内文。
#         按铁律:只在赋值语句上方注释,绝不动三引号字符串内部一个字。
SYSTEM_PROMPT = """You are an expert evaluator tasked with assessing whether a document satisfies specific rubric criteria. Your evaluation must be precise, objective, and based solely on the evidence present in the document.

## Evaluation Framework

You will evaluate each rubric criterion using a binary satisfaction scale:

1. **Not Satisfied (Score: 0.0)**: The document fails to meet the criterion. Key elements are missing, incorrect, or inadequately addressed.

2. **Satisfied (Score: 1.0)**: The document fully meets the criterion. All required elements are present, well-developed, and appropriately detailed.

## Evaluation Process

1. **Understand the Criterion**: Carefully read and interpret what the rubric is asking for.

2. **Search for Evidence**: Systematically review the document for relevant content that addresses the criterion.

3. **Assess Completeness**: Evaluate whether the evidence satisfies or fails to satisfy the criterion.

4. **Provide Reasoning**: Explain your evaluation with specific references to the document content.

## Important Guidelines

- Base your evaluation ONLY on what is explicitly present in the document
- Do not make assumptions about implied or missing content
- Consider the quality, completeness, and relevance of the evidence
- Be consistent in your evaluation standards across all criteria
- Provide specific examples from the document to support your verdict

## Handling Negative-Weight Criteria

Some criteria describe **undesirable behaviors** and carry a **negative weight**. The Weight field in the rubric indicates this. For these criteria:

- **Satisfied (Score: 1.0)**: The undesirable behavior IS present in the document. This will reduce the overall score.
- **Not Satisfied (Score: 0.0)**: The undesirable behavior is ABSENT from the document. This is the desired outcome and does not penalize the score.

Your verdict must reflect whether the described behavior is literally present or absent in the document — not whether the document is generally good or bad.

Note: Example lists in these rubrics are intended to illustrate possible reasoning patterns or relevant topics. These example lists contain correct answers but are not exhaustive. Use them as guidance, but also make your own final judgment about what qualifies as correct when appropriate."""

# Ported from researchrubrics/src/prompts/user_prompt.txt
# 〔讲解〕 单条 rubric 的 user prompt 模板(不分块路径用)。占位符:文档内容 + rubric 标题/类别/权重。
#         注意 {{ }} 是转义后的真实大括号 —— 因为后面用 .format() 填充,模板里要展示的 JSON 范例必须双写花括号才不被当成占位符。
#         它要求裁判只输出一个 JSON(verdict/score/confidence/reasoning/evidence_quotes/missing_elements)。
USER_PROMPT_TEMPLATE = """## Document Content
{document_content}

## Rubric Criterion to Evaluate

**Title**: {rubric_title}
**Category**: {rubric_category}
**Weight**: {rubric_weight}

## Your Task

Evaluate whether the above document satisfies this specific rubric criterion.

## Required Response Format

Provide your evaluation in the following JSON format:

```json
{{
  "verdict": "[Not Satisfied/Satisfied]",
  "score": [0.0/1.0],
  "confidence": [0.0-1.0],
  "reasoning": "Detailed explanation with specific evidence from the document",
  "evidence_quotes": ["Direct quote 1", "Direct quote 2", ...],
  "missing_elements": ["Element 1 that would improve satisfaction", ...]
}}
```

Ensure your response is ONLY the JSON object, with no additional text."""

# Ported from researchrubrics/src/prompts/chunk_prompt_template.txt
# 〔讲解〕 文档太长、超模型上下文时走「分块(chunk)评估」路径用的模板。
#         它把文档切成多块,逐块让裁判抽取「与本条 rubric 相关的证据」,而非直接给 0/1 —— 先收集证据,最后再综合判定。
CHUNK_PROMPT_TEMPLATE = """You are evaluating a large document in chunks. This is chunk {chunk_num} of {total_chunks}.

## Previous Context Summary
{context_summary}

## Current Chunk Content
{chunk_content}

## Rubric Criterion
**Title**: {rubric_title}
**Category**: {rubric_category}

Please evaluate this chunk for evidence related to the rubric criterion. Your response should be in JSON format:

```json
{{
  "relevant_evidence": ["Evidence point 1", "Evidence point 2", ...],
  "satisfaction": true/false,
  "confidence_for_chunk": [0.0-1.0],
  "notes": "Any important observations"
}}
```"""

# Ported from researchrubrics/src/prompts/synthesis_prompt_template.txt
# 〔讲解〕 分块路径的「综合(synthesis)」模板:把上面各块抽到的证据汇总,让裁判据此给出最终 0/1 判定。
#         分块流程 = 多次 chunk 抽证据 → 一次 synthesis 出结论。
SYNTHESIS_PROMPT_TEMPLATE = """Based on the following evidence collected from the document:

Evidence points:
{all_evidence}

Evaluate whether the document satisfies the rubric criterion:
**Title**: {rubric_title}
**Category**: {rubric_category}

Provide your final evaluation in JSON format:
{{
  "verdict": "[Not Satisfied/Satisfied]",
  "score": [0.0/1.0],
  "confidence": [0.0-1.0],
  "reasoning": "Synthesis of evidence"
}}"""

# Tokens per chunk (rough estimate: 1 token ≈ 4 chars)
# 〔讲解〕 下面三个常量是分块判定的阈值(单位 token)。代码里用「字符数 // 4」粗略估 token 数。
# 〔讲解〕 每块最多塞多少 token(乘 4 换成字符数后用于切分)。
CHUNK_MAX_TOKENS = 100000
# 〔讲解〕 给输出/系统开销预留的 token,避免把上下文塞满到放不下回答。
CONTEXT_LIMIT_RESERVE = 50000
# 〔讲解〕 裁判模型的上下文上限;文档估算 token 超过 (上限-预留) 就触发分块。
MODEL_TOKEN_LIMIT = 200000


# 〔讲解〕 RubricResult:单条 rubric 评估完的结构化结果(用 dataclass 当轻量记录)。
#         字段含义:criterion 细则文本、verdict 判词、score 0/1 分、confidence 置信度、
#         weight 权重(可负)、axis 所属评估轴、success 是否成功、error 失败原因(可选)。
@dataclass
class RubricResult:
    criterion: str
    verdict: str
    score: float
    confidence: float
    reasoning: str
    weight: float
    axis: str
    success: bool
    error: Optional[str] = None


# 〔讲解〕 ResearchRubricsEvaluator:本文件主角,继承 Evaluator。
#         职责 = 把一份答案文档按一组 rubric 用 LLM 逐条判分,再加权聚合出总分/分轴分。
#         典型用法:await evaluator.compute_score(prediction, item, llm=...)。
class ResearchRubricsEvaluator(Evaluator):

    # 〔讲解〕 max_concurrent:同时并发评估多少条 rubric 的上限(后面用 asyncio.Semaphore 限流,防止把 API 打爆)。
    def __init__(self, max_concurrent: int = 20):
        self.max_concurrent = max_concurrent

    # ------------------------------------------------------------------
    # Evaluator interface
    # ------------------------------------------------------------------

    # 〔讲解〕 build_prompt:为「单条 rubric、不分块」场景拼出 user prompt。
    #         入参 item 这里没用到(传了空 {}),真正用的是 prediction(答案文档)和 rubric_item(细则字典)。
    #         .get(..., 默认值) 是为了字典缺字段时不报错。
    def build_prompt(self, prediction: str, item: Dict, rubric_item: Dict) -> str:
        """Build user prompt for a single rubric item (no-chunk path)."""
        return USER_PROMPT_TEMPLATE.format(
            document_content=prediction,
            rubric_title=rubric_item.get("criterion", ""),
            rubric_category=rubric_item.get("axis", ""),
            rubric_weight=rubric_item.get("weight", 0),
        )

    # 〔讲解〕 default_response:出错或无 rubric 时返回的「兜底空结果」,保证调用方拿到的字典结构始终一致(指标空、解释空、置信度 0)。
    def default_response(self, err_msg: str = "") -> Dict:
        return {
            "metrics": {},
            "readable_explanation_str": "",
            "rubric_items_with_grades": [],
            "judge_text": err_msg or "Error",
            "confidence": 0,
        }

    # 〔讲解〕 parse_response:基类接口的实现,把裁判输出文本解析成 JSON,内部转交 _parse_json。
    def parse_response(self, judge_text: str) -> Dict:
        return self._parse_json(judge_text)

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    # 〔讲解〕 _parse_json:把裁判 LLM 返回的文本稳健地解析成 dict。LLM 输出常不规范,这里做了两道容错。
    def _parse_json(self, text: str) -> Dict:
        # 〔讲解〕 第一道:去掉模型常爱包的 ```json ... ``` 代码围栏(开头的 ```json 和结尾的 ```)。
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fix invalid backslash escapes that LLMs sometimes emit inside JSON strings
            # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
            # 〔讲解〕 第二道:LLM 有时在字符串里写非法反斜杠转义(如单个 \ 后跟普通字母),导致 json 解析失败。
            #         这条正则:凡是后面不跟合法转义字符的反斜杠,统统改成双反斜杠(转义成字面 \)。
            #         (?!...) 是负向前瞻,意思「这个 \ 后面不是 " \ / b f n r t u 之一」。
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError as e:
                # 〔讲解〕 两道都失败就放弃,打印错误并返回空 dict(上游会据此判为评估失败/重试)。
                print(f"JSON decoding failed: {e}")
                return {}

    # ------------------------------------------------------------------
    # Chunking (mirrors researchrubrics workflow)
    # ------------------------------------------------------------------

    # 〔讲解〕 _chunk_content:把超长文档按「段落边界」切成多块,避免硬截断把句子切碎。返回块列表。
    def _chunk_content(self, content: str) -> List[str]:
        # 〔讲解〕 token 上限换算成字符上限(× 4 的粗估)。
        max_chars = CHUNK_MAX_TOKENS * 4
        # 〔讲解〕 文档本来就不超长,直接整篇作为唯一一块返回,省去切分。
        if len(content) <= max_chars:
            return [content]
        # 〔讲解〕 用空行(\n\n)切段落,然后贪心地往当前块累加,塞不下就开新块 —— 保证每块都不超 max_chars。
        paragraphs = content.split("\n\n")
        chunks, current = [], ""
        for para in paragraphs:
            if len(current) + len(para) < max_chars:
                current += para + "\n\n"
            else:
                # 〔讲解〕 当前块装不下这段了:先把已攒的 current 收尾入列,再用本段开启新块。
                if current:
                    chunks.append(current.strip())
                current = para + "\n\n"
        # 〔讲解〕 循环结束后把最后残留的 current 也收进去(否则末块会丢)。
        if current:
            chunks.append(current.strip())
        return chunks

    # ------------------------------------------------------------------
    # Single rubric evaluation
    # ------------------------------------------------------------------

    # 〔讲解〕 _evaluate_single:对「单条 rubric」做一次完整判定(文档不分块的常规路径)。
    #         返回一个 RubricResult。内含 3 次重试 + 指数退避,提升对 LLM 偶发抖动的鲁棒性。
    async def _evaluate_single(self, rubric: Dict, content: str, llm: str) -> RubricResult:
        """Evaluate one rubric item against content (no chunking)."""
        user_prompt = self.build_prompt(content, {}, rubric)
        # 〔讲解〕 标准的 chat 消息结构:system(裁判人设)+ user(本条 rubric 的具体问题)。
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # 〔讲解〕 最多试 3 次(attempt = 0,1,2)。
        for attempt in range(3):
            try:
                # 〔讲解〕 调裁判 LLM(基类提供的异步接口)。response_format 强制 json_object 输出,降低解析失败概率。
                text = await self.async_complete(
                    messages, llm, max_tokens=30000, response_format={"type": "json_object"}
                )
                data = self._parse_json(text)
                # 〔讲解〕 解析出空 dict 视为失败 → 抛错进入重试分支。
                if not data:
                    raise ValueError("Empty JSON response")
                # 〔讲解〕 成功:把裁判 JSON 各字段塞进 RubricResult。score/confidence/weight 都 float() 强转防类型不一致。
                #         weight 取自 rubric 自身(裁判不决定权重,权重是细则预设的,可为负)。
                return RubricResult(
                    criterion=rubric.get("criterion", ""),
                    verdict=data.get("verdict", "Not Satisfied"),
                    score=float(data.get("score", 0.0)),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=data.get("reasoning", ""),
                    weight=float(rubric.get("weight", 0)),
                    axis=rubric.get("axis", ""),
                    success=True,
                )
            except Exception as e:
                # 〔讲解〕 最后一次(attempt==2)仍失败:不再重试,返回一个标记 success=False 的「错误结果」,
                #         score=0、verdict="Error",让整体评估能继续而不是崩掉。
                if attempt == 2:
                    return RubricResult(
                        criterion=rubric.get("criterion", ""),
                        verdict="Error",
                        score=0.0,
                        confidence=0.0,
                        reasoning=str(e),
                        weight=float(rubric.get("weight", 0)),
                        axis=rubric.get("axis", ""),
                        success=False,
                        error=str(e),
                    )
                # 〔讲解〕 非最后一次:指数退避后重试(等 2^0=1、2^1=2 秒),给 API 喘息、缓解限流。
                await asyncio.sleep(2 ** attempt)

    # 〔讲解〕 _evaluate_with_chunks:文档超长时的判定路径 —— 先逐块抽证据,再综合成一条最终判定。
    #         流程:N 块 × CHUNK_PROMPT(抽证据)→ 汇总所有证据 → 1 次 SYNTHESIS_PROMPT(出结论)。
    #         注意:此路径没有像 _evaluate_single 那样的重试包裹。
    async def _evaluate_with_chunks(self, rubric: Dict, content: str, llm: str) -> RubricResult:
        """Evaluate one rubric item using chunked content + synthesis."""
        chunks = self._chunk_content(content)
        all_evidence = []

        # 〔讲解〕 逐块抽证据。enumerate(chunks, 1) 让块号从 1 开始(给 prompt 里的 "chunk i of N" 用)。
        for i, chunk in enumerate(chunks, 1):
            chunk_prompt = CHUNK_PROMPT_TEMPLATE.format(
                chunk_num=i,
                total_chunks=len(chunks),
                # 〔讲解〕 简易「上下文摘要」:第一块写 First chunk,之后只写一句占位提示(此实现没做真正的跨块记忆)。
                context_summary="Previous chunks evaluated" if i > 1 else "First chunk",
                chunk_content=chunk,
                rubric_title=rubric.get("criterion", ""),
                rubric_category=rubric.get("axis", ""),
            )
            text = await self.async_complete(
                [
                    {"role": "system", "content": "You are evaluating document chunks for rubric criteria."},
                    {"role": "user", "content": chunk_prompt},
                ],
                llm, max_tokens=30000, response_format={"type": "json_object"},
            )
            chunk_data = self._parse_json(text)
            # 〔讲解〕 把本块抽到的证据列表追加进总证据池(extend 而非 append,摊平成一维)。
            all_evidence.extend(chunk_data.get("relevant_evidence", []))

        # 〔讲解〕 综合阶段:把全部证据(转成带缩进的 JSON 字符串塞进 prompt)交给裁判,产出最终 0/1 判定。
        synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            all_evidence=json.dumps(all_evidence, indent=2),
            rubric_title=rubric.get("criterion", ""),
            rubric_category=rubric.get("axis", ""),
        )
        text = await self.async_complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": synthesis_prompt},
            ],
            llm, max_tokens=30000, response_format={"type": "json_object"},
        )
        data = self._parse_json(text)
        # 〔讲解〕 同 _evaluate_single 的成功分支,组装 RubricResult(这里固定 success=True,未对空 data 做失败标记)。
        return RubricResult(
            criterion=rubric.get("criterion", ""),
            verdict=data.get("verdict", "Not Satisfied"),
            score=float(data.get("score", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
            weight=float(rubric.get("weight", 0)),
            axis=rubric.get("axis", ""),
            success=True,
        )

    # 〔讲解〕 _evaluate_rubric:单条 rubric 的「调度入口」—— 决定走分块还是不分块,并受信号量限流。
    #         semaphore 在外层创建,这里 async with 占一个名额,实现「最多 max_concurrent 条同时跑」。
    async def _evaluate_rubric(self, rubric: Dict, content: str, llm: str, semaphore: asyncio.Semaphore) -> RubricResult:
        async with semaphore:
            # 〔讲解〕 粗估文档 token = 字符数 // 4;超过 (模型上限 - 预留) 就分块,否则常规单次评估。
            estimated_tokens = len(content) // 4
            if estimated_tokens > MODEL_TOKEN_LIMIT - CONTEXT_LIMIT_RESERVE:
                return await self._evaluate_with_chunks(rubric, content, llm)
            return await self._evaluate_single(rubric, content, llm)

    # ------------------------------------------------------------------
    # Scoring (mirrors researchrubrics calculate_compliance_score)
    # ------------------------------------------------------------------

    # 〔讲解〕 _calculate_score:加权聚合出「总分」。这是本评分体系的核心公式。
    #         分母 = 所有正权重之和(负权重不计入分母);分子 = Σ(score × weight)(负权重条目若被判 1 会拉低分子)。
    #         返回 numerator/denominator ∈ 大致 0~1;分母为 0(没有正权重项)则返回 None。
    #         入参 rubric_items 在此函数体内未实际使用,只是签名保留(与原项目对齐)。
    def _calculate_score(self, rubric_items: List[Dict], results: List[RubricResult]) -> Optional[float]:
        denominator = sum(r.weight for r in results if r.weight > 0)
        if denominator == 0:
            return None
        numerator = sum(r.score * r.weight for r in results)
        return numerator / denominator

    # 〔讲解〕 _axis_scores:按「评估轴(axis)」分组各算一个加权分,用于看答案在不同维度上的强弱。
    #         逻辑与 _calculate_score 相同,只是先按 axis 把结果归桶再分别计算。
    def _axis_scores(self, results: List[RubricResult]) -> Dict[str, float]:
        # 〔讲解〕 defaultdict(list):访问不存在的 axis 键时自动建空列表,省去手动初始化。
        axis_groups: Dict[str, List[RubricResult]] = defaultdict(list)
        for r in results:
            # 〔讲解〕 只把有 axis 标签的结果归桶(空 axis 的跳过)。
            if r.axis:
                axis_groups[r.axis].append(r)
        scores = {}
        for axis, group in axis_groups.items():
            denom = sum(r.weight for r in group if r.weight > 0)
            # 〔讲解〕 该轴有正权重才算分(避免除零),否则这个轴不出现在结果里。
            if denom > 0:
                scores[axis] = sum(r.score * r.weight for r in group) / denom
        return scores

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    # 〔讲解〕 compute_score:本评分器的主入口(异步)。给一份 prediction + 含 rubric 的 item,产出完整评分结果字典。
    #         返回字段:metrics(总分+各轴分)、readable_explanation_str(给人看的逐条说明)、
    #         rubric_items_with_grades(逐条结构化结果)、judge_text、confidence。
    #         入参 error/err_msg:上游若已知此条出错,可直接短路返回兜底结果。
    async def compute_score(
        self, prediction: str, item: Dict, llm: str = "gpt-4.1-mini", error: bool = False, err_msg: str = ""
    ) -> Dict:
        # 〔讲解〕 上游已标记出错:直接返回兜底空结果,不浪费 LLM 调用。
        if error:
            return self.default_response(err_msg=err_msg)

        # 〔讲解〕 取本题的 rubric 列表。优先用 item["rubrics"]。
        rubric_items = item.get("rubrics", [])
        if not rubric_items:
            # 〔讲解〕 没有现成 rubrics 时的兼容回退:从 combo_judgements(历史评判记录)里「反推」出 rubric 定义。
            #         即:拿第一条已评判记录里的 rubric_items_with_grades,只抽出 criterion/weight/axis 三个字段重建 rubric。
            combo_judgements = item.get("combo_judgements", [])
            if not combo_judgements:
                return self.default_response(err_msg="No rubric found in item")
            rubric_items = [
                {k: r[k] for k in ("criterion", "weight", "axis") if k in r}
                for r in combo_judgements[0].get("rubric_items_with_grades", [])
            ]
            # 〔讲解〕 回退后仍为空 → 确实无 rubric,返回兜底。
            if not rubric_items:
                return self.default_response(err_msg="No rubric found in item")

        # 〔讲解〕 创建并发信号量,容量 = max_concurrent,传给每个 _evaluate_rubric 共享,统一限流。
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Evaluate all rubric items in parallel (mirrors evaluate_all_rubrics)
        # 〔讲解〕 给每条 rubric 建一个协程任务(此刻只是创建,尚未真正发起 LLM 调用)。
        tasks = [
            self._evaluate_rubric(rubric, prediction, llm, semaphore)
            for rubric in rubric_items
        ]
        # 〔讲解〕 asyncio.gather 并发执行所有任务并按原顺序收集结果(实际并发度由上面的信号量约束)。
        results: List[RubricResult] = await asyncio.gather(*tasks)

        # 〔讲解〕 算总分;assert 防御 —— 走到这里 rubric 非空,理论上总有正权重,不该是 None。
        overall_score = self._calculate_score(rubric_items, results)
        assert overall_score is not None
        # 〔讲解〕 metrics 先放总分,再 update 合入各轴分,组成完整指标字典。
        metrics = {"overall_score": overall_score}
        metrics.update(self._axis_scores(results))

        # Build readable explanations
        # 〔讲解〕 下面这段把结构化结果转成「人类可读」和「机器可读」两份输出,供报告/分析用。
        rubric_items_with_grades = []
        readable_explanation_list = []
        judge_text = ""
        for result in results:
            # 〔讲解〕 人读版每条:形如 "[Satisfied] 细则文本\n\tReasoning: 理由"。
            readable = f"[{result.verdict}] {result.criterion}\n\tReasoning: {result.reasoning}"
            readable_explanation_list.append(readable)
            # 〔讲解〕 机器读版每条:逐字段落进 dict,后续可入库/做统计。
            rubric_items_with_grades.append({
                "criterion": result.criterion,
                "weight": result.weight,
                "axis": result.axis,
                "verdict": result.verdict,
                "score": result.score,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "success": result.success,
            })
            # 〔讲解〕 judge_text 在循环里被反复覆盖,最终只留「最后一条」的 reasoning(完整逐条文本在上面的 grades 里)。
            judge_text = result.reasoning  # last one; full text per-rubric is in rubric_items_with_grades

        # 〔讲解〕 排序让 [Not Satisfied] 的条目排在前面(reverse=True + 布尔 key:startswith 为 True 排前),
        #         便于人一眼先看到「没满足」的短板。
        readable_explanation_list.sort(
            key=lambda x: x.startswith("[Not Satisfied]"), reverse=True
        )
        readable_explanation_str = "\n\n".join(readable_explanation_list)
        # 〔讲解〕 整体前置两个换行,纯排版(让这段说明在拼接到更大文本时上方留空)。
        readable_explanation_str = f"\n\n{readable_explanation_str}"

        # 〔讲解〕 另调基类的 extract_confidence,对整篇答案再要一个整体置信度(与逐条 confidence 不同维度)。
        confidence = await self.extract_confidence(prediction, llm=llm)

        # 〔讲解〕 返回标准化结果字典,字段结构与 default_response 完全对齐,保证调用方处理一致。
        return {
            "metrics": metrics,
            "readable_explanation_str": readable_explanation_str,
            "rubric_items_with_grades": rubric_items_with_grades,
            "judge_text": judge_text,
            "confidence": confidence,
        }
