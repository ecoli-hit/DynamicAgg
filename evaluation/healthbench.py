# 〔讲解〕本文件 = HealthBench 数据集的「LLM 评分器」,处于 pipeline 的 evaluation/ 这一环。
# 〔讲解〕上游:rollout/ 跑出搜索轨迹、aggregation+aggagent 聚合出一份最终答案(prediction);
# 〔讲解〕本文件下游:把这份 prediction 连同题目(item)喂给裁判大模型(默认 gpt-4.1),逐条 rubric 打分,
# 〔讲解〕汇总成 overall_score 与各 tag 维度分。HealthBench 是 OpenAI 出的医疗问答 benchmark,
# 〔讲解〕评分方式不是对答案,而是用「评分细则(rubric)」逐条判定「答案是否满足该条」,满足就拿对应分值。
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import os
import re
from typing import Dict, List

from dotenv import load_dotenv

# 〔讲解〕base.Evaluator 是抽象基类,提供 async_complete(调裁判 LLM)、extract_confidence(抽置信度)等通用能力。
from .base import Evaluator
# 〔讲解〕复用 rollout 阶段的系统提示;当 item 没存「实际问过模型的消息」时,用它来还原裁判看到的对话上下文。
from rollout.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_DR


# 〔讲解〕读取 .env,把 OPENAI_API_KEY 等密钥注入环境变量(裁判模型调用要用)。
load_dotenv()

# 〔讲解〕HEALTHBENCH_GRADER_TEMPLATE = 喂给裁判 LLM 的「评分指令模板」。
# 〔讲解〕<<conversation>> 和 <<rubric_item>> 是两个占位符,后面 build_prompt 会用真实内容替换。
# 〔讲解〕模板核心要求:让裁判只看最后一轮 assistant 回复,判断它是否满足这条 rubric,返回 {explanation, criteria_met} 的 JSON。
# 〔讲解〕注意 rubric 可能是「负向条件」(如"给了错误信息"),裁判要如实判定是否命中,而非判断答案好坏。
HEALTHBENCH_GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


# 〔讲解〕HealthBenchJudgeResponse = 一道题评完后的「结构化评分结果容器」(dataclass)。
# 〔讲解〕metrics: 各项分数(overall_score + 各 tag 分);readable_explanation_str: 人读的逐条解释拼串;
# 〔讲解〕rubric_items_with_grades: 每条 rubric 带上判定结果;judge_text: 裁判最后一次原始回复;confidence: 答案自报置信度。
@dataclass
class HealthBenchJudgeResponse:
    metrics: Dict
    readable_explanation_str: str
    rubric_items_with_grades: List[Dict]
    judge_text: str
    confidence: int = 0

    # 〔讲解〕把 dataclass 转成普通 dict,方便序列化成 JSON 落盘 / 跨模块传递。
    def to_dict(self) -> Dict:
        return {
            "metrics": self.metrics,
            "readable_explanation_str": self.readable_explanation_str,
            "rubric_items_with_grades": self.rubric_items_with_grades,
            "judge_text": self.judge_text,
            "confidence": self.confidence,
        }


# 〔讲解〕HealthBenchEvaluator = HealthBench 的具体评分器,实现基类 Evaluator 要求的四个抽象方法。
# 〔讲解〕角色:把「答案 + 题目 + 单条 rubric」拼成裁判提示,调裁判 LLM,解析 JSON,汇总打分。
class HealthBenchEvaluator(Evaluator):
    # 〔讲解〕build_prompt:为「单条 rubric」拼出完整裁判提示词。
    # 〔讲解〕入参 prediction=待评答案, item=整道题数据, rubric_item=已带分值的单条细则文本; 返回填好占位符的提示串。
    def build_prompt(self, prediction: str, item: Dict, rubric_item: str) -> str:
        """Build grading prompt for a single rubric item."""
        # Strip control characters that break JSON serialization (keeps \t, \n, \r)
        # 〔讲解〕_clean = 一个小工具:删掉会破坏 JSON 的不可见控制字符(保留制表/换行/回车);s 为空则原样返回。
        _clean = lambda s: re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s) if s else s
        if prediction:
            prediction = _clean(prediction)
        # 〔讲解〕优先用 item 里「实际问过模型的消息」还原真实对话;没有就退而临时拼一个 system+用户 prompt。
        if "actual_queried_prompt_messages" in item:
            messages = item["actual_queried_prompt_messages"]
        else:
            # 〔讲解〕system 消息 = rollout 用的通用提示 + Deep-Research 专用提示拼接,目的是让裁判看到的语境与生成时一致。
            system_msg = {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + SYSTEM_PROMPT_DR}
            messages = [system_msg] + item.get("prompt", [])
        # 〔讲解〕对每条消息的 content 也做控制字符清洗(用 **m 复制原字段再覆盖 content,不破坏其它键)。
        messages = [{**m, "content": _clean(m.get("content", "") or "")} for m in messages]
        rubric_item = _clean(str(rubric_item))
        # 〔讲解〕把待评答案作为「最后一轮 assistant 回复」拼到对话末尾 —— 这正是裁判要打分的目标轮次。
        convo_with_response = messages + [
            dict(content=prediction, role="assistant")
        ]
        # 〔讲解〕把多轮消息拍平成「role: content」的纯文本对话,供模板里的 <<conversation>> 占位符使用。
        convo_str = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in convo_with_response]
        )
        # 〔讲解〕用真实对话和真实 rubric 替换模板两个占位符,得到最终裁判提示词。
        grader_prompt = HEALTHBENCH_GRADER_TEMPLATE.replace(
            "<<conversation>>", convo_str
        ).replace("<<rubric_item>>", rubric_item)
        return grader_prompt

    # 〔讲解〕default_response:出错或无 rubric 时返回的「空结果」,保证下游拿到的结构始终完整不报错。
    def default_response(self, err_msg: str = "") -> Dict:
        return HealthBenchJudgeResponse(
            metrics={},
            readable_explanation_str="",
            rubric_items_with_grades=[],
            judge_text=err_msg or "Error",
            confidence=0,
        ).to_dict()

    # 〔讲解〕parse_response:基类要求实现的抽象方法,这里直接委托给下面的 JSON 解析函数。
    def parse_response(self, judge_text: str) -> Dict:
        """Parse judge response text into structured format."""
        return self.parse_json_to_dict(judge_text)

    # 〔讲解〕parse_json_to_dict:把裁判返回的文本解析成 dict;裁判常用 ```json ... ``` 包裹,需先剥壳。
    def parse_json_to_dict(self, json_string: str) -> dict:
        # Remove markdown-style ```json``` markers if present
        # 〔讲解〕正则去掉首部 ```json 和尾部 ```(markdown 代码块标记),只留中间纯 JSON。
        json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

        try:
            return json.loads(json_cleaned)
        # 〔讲解〕裁判偶尔吐出非法 JSON,这里兜底返回空 dict(让上层 while 循环重试,见 compute_score)。
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
            return {}


    # 〔讲解〕calculate_score:HealthBench 核心计分公式 = 命中的正分之和 / 所有正分满分之和。
    # 〔讲解〕只把 points>0 的条目计入「满分分母」;负分条目命中会扣分(进分子),但不进分母。
    # 〔讲解〕返回 0~1 的比例;若满分为 0(某个 tag 下无正分项)返回 None,表示「该维度无意义,跳过」。
    def calculate_score(self, rubric_items, grading_response_list) -> float | None:
        # 〔讲解〕分母:把所有「正分」rubric 的分值加总,作为这道题(或这个 tag)的理论最高分。
        total_possible_points = sum(
            rubric_item.get("points", 0)
            for rubric_item in rubric_items
            if rubric_item.get("points", 0) > 0
        )
        if total_possible_points == 0:
            # should not happen for overall score, but may happen for tags
            return None

        # 〔讲解〕分子:遍历(rubric, 裁判结果)配对,只要裁判判定 criteria_met 为真,就把该条分值累加(含负分)。
        # 〔讲解〕strict=True 强制两个列表等长,长度不一致会直接报错,防止错位计分。
        achieved_points = sum(
            rubric_item.get("points", 0)
            for rubric_item, grading_response in zip(
                rubric_items, grading_response_list, strict=True
            )
            if grading_response["criteria_met"]
        )
        overall_score = achieved_points / total_possible_points
        return overall_score

    # 〔讲解〕compute_score:本文件主入口(async),评完一道题给出完整结果。
    # 〔讲解〕入参 prediction=答案, item=题目, llm=裁判模型名(默认 gpt-4.1), error/err_msg=上游若已出错则直接走默认结果。
    async def compute_score(
        self, prediction: str, item: Dict, llm: str = "gpt-4.1", error: bool = False, err_msg: str = ""
    ) -> Dict:
        """Compute evaluation score using rubric-based evaluation."""
        # 〔讲解〕上游(如 rollout)已标记出错,就不浪费裁判调用,直接返回空结果。
        if error:
            return self.default_response(err_msg=err_msg)

        # Get rubric from item
        # 〔讲解〕取出这道题的评分细则列表;没有 rubric 就无法评分,返回默认结果。
        rubric_items = item.get("rubrics", [])
        if not rubric_items:
            return self.default_response(err_msg="No rubric found in item")

        grading_response_list = []

        # Evaluate each rubric item
        # 〔讲解〕逐条 rubric 调一次裁判:一道题有 N 条细则就调 N 次 LLM(成本主要来源)。
        for rubric_item in rubric_items:
            criterion = rubric_item.get("criterion", "")
            points = float(rubric_item.get("points", 0))

            # 〔讲解〕把分值前置写进 rubric 文本(如 "[5.0] 应建议就医"),让裁判同时看到判据和它的权重。
            judge_prompt = self.build_prompt(prediction, item, f"[{points}] {criterion}")
            messages = [{"role": "user", "content": judge_prompt}]

            # 〔讲解〕重试循环:反复调裁判,直到拿到包含合法布尔 criteria_met 的 JSON 才跳出(防止裁判偶发坏输出污染计分)。
            while True:
                # 〔讲解〕async_complete 来自基类,实际调 litellm/tinker 访问裁判模型,max_tokens=1024 限制单条评语长度。
                judge_text = await self.async_complete(messages, llm, max_tokens=1024)
                grading_response_dict = self.parse_json_to_dict(judge_text)
                if "criteria_met" in grading_response_dict:
                    label = grading_response_dict["criteria_met"]
                    # 〔讲解〕严格校验必须是 Python 的 True/False(用 is 比较),排除 "true" 字符串、1/0 等歧义值。
                    if label is True or label is False:
                        break

            grading_response_list.append(grading_response_dict)

        # 〔讲解〕全部 rubric 评完后,算总分;assert 兜底:整题满分理论上不会为 0,若为 None 说明数据异常。
        overall_score = self.calculate_score(rubric_items, grading_response_list)
        assert overall_score is not None
        metrics = {"overall_score": overall_score}

        # 〔讲解〕example_tags = 这道题本身的分类标签(如主题/难度);把整题总分复制到每个标签,便于按题型聚合。
        example_tag_scores = {tag: overall_score for tag in item["example_tags"]}
        assert len(example_tag_scores) == len(item["example_tags"])  # No duplicates.
        metrics.update(example_tag_scores)

        # 〔讲解〕下面按「rubric 级标签」分组:同一 tag(如"安全性""准确性")下的若干 rubric 单独算一个维度分。
        # 〔讲解〕defaultdict(list) 让缺失的 tag 键自动初始化为空列表,省去手动判断。
        rubric_tag_items_grades = defaultdict(list)
        for rubric_item, grading_response in zip(rubric_items, grading_response_list):
            curr_item_tags = set()  # Ensure no duplicates in a rubric item.
            for tag in rubric_item.get("tags", []):
                # 〔讲解〕把(该 rubric, 它的判定)塞进对应 tag 的桶里,后面对每个桶单独调 calculate_score。
                rubric_tag_items_grades[tag].append((rubric_item, grading_response))
                assert tag not in curr_item_tags
                curr_item_tags.add(tag)

        rubric_tag_scores = {}
        for tag, items_grades in rubric_tag_items_grades.items():
            # 〔讲解〕zip(*...) 把 [(rubric,grade), ...] 拆回两条平行列表,再算这个 tag 的分。
            items, grades = zip(*items_grades)
            score = self.calculate_score(items, grades)
            # 〔讲解〕只有当该 tag 下至少有一条正分 rubric(score 非 None)才记录,否则跳过这个无意义维度。
            if score is not None:  # implies at least one positive criterion
                rubric_tag_scores[tag] = score
        metrics.update(rubric_tag_scores)

        # construct the list of explanations and grades
        # 〔讲解〕下面构造两份「人读 / 机读」的逐条结果:一份纯文本解释串,一份带判定的 rubric 字典列表。
        rubric_items_with_grades = []
        readable_explanation_list = []
        for rubric_item, grading_response in zip(rubric_items, grading_response_list):
            explanation = grading_response.get("explanation", "No explanation provided")
            criteria_met = grading_response["criteria_met"]
            # 〔讲解〕拼一行人读解释:"[True/False] 原始rubric\n\t解释",方便事后人工核查裁判判得对不对。
            readable_explanation = (
                f"[{criteria_met}] {rubric_item}\n\tExplanation: {explanation}"
            )
            readable_explanation_list.append(readable_explanation)
            rubric_items_with_grades.append(
                {
                    **rubric_item,
                    "criteria_met": criteria_met,
                    "explanation": explanation,
                }
            )

        # 〔讲解〕排序把「未命中([False])」的条目排到前面(reverse=True 让 True 键排后),方便先看哪些细则没满足。
        readable_explanation_list.sort(
            key=lambda x: x.startswith("[False]"), reverse=True
        )
        readable_explanation_str = "\n\n".join(readable_explanation_list)
        readable_explanation_str = f"\n\n{readable_explanation_str}"

        # 〔讲解〕额外再调一次裁判,从答案文本里抽出模型自报的「Confidence: X%」置信度(基类实现),用于后续校准分析。
        confidence = await self.extract_confidence(prediction, llm=llm)

        # 〔讲解〕组装最终结构化结果并转 dict 返回;judge_text 取的是最后一条 rubric 的裁判原文(仅作样例留痕)。
        result = HealthBenchJudgeResponse(
            metrics=metrics,
            readable_explanation_str=readable_explanation_str,
            rubric_items_with_grades=rubric_items_with_grades,
            judge_text=judge_text,
            confidence=confidence,
        ).to_dict()

        return result
