#〔讲解〕本文件属于 evaluation/ 环节,是 DeepSearchQA benchmark 的"裁判(LLM-as-judge)"。
#〔讲解〕上游:rollout/ + aggregation/ 把模型最终答案(prediction)交过来;本文件用一个大模型(默认 gpt-4.1)
#〔讲解〕去判这份答案对不对,并算出 precision/recall/F1 等指标。它继承 base.py 里的抽象类 Evaluator。
#〔讲解〕核心难点:DeepSearchQA 的题分"单答案/集合答案"两种,判分要逐个 ground-truth 命中 + 统计多余答案。
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import textwrap
from typing import Dict, List, Tuple

#〔讲解〕litellm 是统一多家大模型 API 的封装库(OpenAI/Gemini 等用同一套调用),这里用它来调"裁判模型"。
import litellm
#〔讲解〕load_dotenv 把 .env 文件里的 API key(如 OPENAI_API_KEY)读进环境变量,供下面 os.getenv 取用。
from dotenv import load_dotenv

#〔讲解〕从同目录 base.py 引入抽象基类 Evaluator(定义了 extract_confidence / async_complete 等通用能力)。
from .base import Evaluator

load_dotenv()

#〔讲解〕这是给"被评测模型"看的输出格式说明:要求它按 Explanation/Exact Answer/Confidence 三段式作答。
#〔讲解〕注意它定义在这里供别处引用;Confidence 那一行就是后面 extract_confidence 要抓的置信度来源。
DEEPSEARCHQA_INSTRUCTION = """Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

#〔讲解〕这是给"裁判模型"的主体指令(prompt 模板上半段):说明判分任务、两种 Prompt Type 的判法、输出要求。
#〔讲解〕textwrap.dedent 只是去掉源码缩进让多行字符串左对齐,不改内容。关键约定:输出必须是嵌套 JSON,
#〔讲解〕含 Answer Correctness → {Explanation, Correctness Details(逐项 true/false), Excessive Answers, confidence}。
DEEPSEARCH_QA_PROMPT = textwrap.dedent("""\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), `"Excessive Answers"` (a list of strings indicating the excessive answers), and `"confidence"` (an integer 0-100, extracted from the AI Response if provided, otherwise 0).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


""")

#〔讲解〕prompt 模板的下半段:先给裁判一个"标准输出范例(few-shot)"教它 JSON 长什么样,
#〔讲解〕再用 {prompt}/{prompt_type}/{answer}/{response} 占位符把真实题目、答案类型、ground-truth、被评答案填进去。
#〔讲解〕r"""..."""(raw 字符串)是为了让 ```json 代码块里的反斜杠不被转义。下方 build_prompt 会 .format() 它。
GRADER_RATING_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ],
    "confidence": 85
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


#〔讲解〕一个结构化容器(dataclass),装一道题判完后的全部结果字段。
#〔讲解〕all_correct=完全答对;correct_with_excessive_answers=答全但多答了;fully_incorrect=全错;
#〔讲解〕precision/recall/f1_score=集合答案的查准/查全/调和均值;judge_text=裁判原文;confidence=置信度。
@dataclass
class DeepSearchQAJudgeResponse:
    all_correct: bool
    correct_with_excessive_answers: int
    fully_incorrect: int
    precision: float
    recall: float
    f1_score: float
    judge_text: str
    confidence: int = 0

    #〔讲解〕把 dataclass 摊平成普通 dict,方便落盘成 JSON / 汇总统计。流程里最终对外都用 dict 形态。
    def to_dict(self) -> Dict:
        return {
            "all_correct": self.all_correct,
            "correct_with_excessive_answers": self.correct_with_excessive_answers,
            "fully_incorrect": self.fully_incorrect,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "judge_text": self.judge_text,
            "confidence": self.confidence,
        }


#〔讲解〕本文件主角:DeepSearchQA 专用评测器,继承抽象基类 Evaluator。
#〔讲解〕职责四步——build_prompt 拼裁判提示 → 调大模型 → parse_response 解析 JSON → _calculate_metrics 算指标。
class DeepSearchQAEvaluator(Evaluator):
    #〔讲解〕组装递给裁判模型的完整 prompt。入参:prediction=被评模型的答案;item=数据集里这道题(含答案/题干/答案类型)。
    #〔讲解〕用 .get(默认值) 防止字段缺失报错;返回"上半段说明 + 下半段填好占位符的范例与题面"拼接而成的完整提示。
    def build_prompt(self, prediction: str, item: Dict) -> str:
        answer = item.get("answer", "")
        question = item.get("question", "")
        answer_type = item.get("answer_type", "")
        return DEEPSEARCH_QA_PROMPT + GRADER_RATING_OUTPUT_EXAMPLE.format(
            prompt=question,
            prompt_type=answer_type,
            answer=answer,
            response=prediction,
        )

    #〔讲解〕兜底结果:当出错或无法判分时返回"全错+全 0"的标准结构,保证下游永远拿到完整字段不会 KeyError。
    def default_response(self, err_msg: str = "") -> Dict:
        return DeepSearchQAJudgeResponse(
            all_correct=False,
            correct_with_excessive_answers=0,
            fully_incorrect=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            judge_text=err_msg or "Error",
            confidence=0,
        ).to_dict()

    #〔讲解〕从裁判模型的自由文本里"抠出"那段 JSON。大模型常把 JSON 包在 ```json ... ``` 代码块里,或夹在解释文字中。
    def _extract_json_blob(self, raw_text: str) -> str:
        #〔讲解〕优先用正则匹配 ```json ... ``` 围栏里的 {...}。re.DOTALL 让 . 能跨行匹配,IGNORECASE 容忍大小写。
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1)
        #〔讲解〕没有围栏就退而求其次:取第一个 { 到最后一个 } 之间的整段,当作 JSON 主体。
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw_text[start : end + 1]
        #〔讲解〕实在找不到大括号则返回空串,交给上游走 default_response 兜底。
        return ""

    #〔讲解〕把裁判给的 "Correctness Details" 归一化成两个并行列表:ratings(每项对/错的布尔)与 expected_answers(对应答案文字)。
    #〔讲解〕兼容两种格式:① 字典 {答案:true/false};② 列表 [{"key":答案,"value":true}, ...](simple-evals 风格)。
    def _normalize_correctness_details(self, details_raw) -> Tuple[List[bool], List[str]]:
        # Supports:
        # 1) {"A": true, "B": false}
        # 2) [{"key": "A", "value": true}, ...] (simple-evals style)
        ratings: List[bool] = []
        expected_answers: List[str] = []

        #〔讲解〕格式①:字典形态,逐对 (答案文字, 命中布尔) 收进两个列表。bool(v) 兜住裁判偶尔给 "true" 字符串等情况。
        if isinstance(details_raw, dict):
            for k, v in details_raw.items():
                expected_answers.append(str(k))
                ratings.append(bool(v))
            return ratings, expected_answers

        #〔讲解〕格式②:列表形态,每个元素是 {"key":..., "value":...},取出 key/value 收进列表(缺字段给默认值防崩)。
        if isinstance(details_raw, list):
            for item in details_raw:
                if isinstance(item, dict):
                    key = item.get("key", "")
                    value = item.get("value", False)
                    expected_answers.append(str(key))
                    ratings.append(bool(value))
            return ratings, expected_answers

        #〔讲解〕两种都不是(如 None / 字符串)就返回空列表,后续指标会按"无 ground-truth"处理。
        return ratings, expected_answers

    #〔讲解〕本文件的"打分核心":把裁判给的"逐项命中布尔列表 ratings + 多余答案个数"换算成 P/R/F1 及三个对错标签。
    #〔讲解〕信息检索术语映射:TP=答对的 ground-truth 个数;FN=漏掉的 ground-truth;FP=多答的(excessive)个数。
    def _calculate_metrics(self, ratings: List[bool], excessive_answers_count: int) -> Dict:
        num_correct = sum(1 for r in ratings if r)
        true_positive = num_correct
        #〔讲解〕漏答 = ground-truth 总数 − 答对数(应给却没给的)。
        false_negative = len(ratings) - num_correct
        #〔讲解〕误答 = 裁判判定的多余答案个数(给了却不该给的)。
        false_positive = excessive_answers_count

        #〔讲解〕这道题是否存在 ground-truth(ratings 非空)。空列表代表裁判没给逐项明细,需特殊处理。
        has_expected_answers = bool(ratings)
        #〔讲解〕"全部 ground-truth 都答对":有明细 且 答对数==总数。
        all_expected_answers_correct = has_expected_answers and (num_correct == len(ratings))
        #〔讲解〕"全错":有明细 且 一个都没答对。
        fully_incorrect = 1 if (has_expected_answers and num_correct == 0) else 0
        #〔讲解〕"答全但多答":有多余答案(FP>0) 且 (该答的全答对 或 本就没明细) → 内容对但有冗余。
        correct_with_excessive_answers = 1 if (false_positive > 0 and (all_expected_answers_correct or not has_expected_answers)) else 0
        #〔讲解〕"完全正确":该答的全答对(或无明细) 且 没有任何多余答案。这是最严格的"全对"判定。
        all_correct = (all_expected_answers_correct or not has_expected_answers) and false_positive == 0

        #〔讲解〕标准 IR 公式,分母为 0 时退回 0.0 防除零。precision=TP/(TP+FP),衡量"答出来的有多少是对的"。
        precision = float(true_positive / (true_positive + false_positive)) if (true_positive + false_positive) > 0 else 0.0
        #〔讲解〕recall=TP/(TP+FN),衡量"该答的覆盖了多少"。
        recall = float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) > 0 else 0.0
        #〔讲解〕f1=查准查全的调和平均,综合一个数,(precision+recall)=0 时退回 0.0。
        f1_score = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "all_correct": all_correct,
            "correct_with_excessive_answers": correct_with_excessive_answers,
            "fully_incorrect": fully_incorrect,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    #〔讲解〕把裁判模型的整段输出文本,解析成最终的结果 dict。串起前面三个辅助函数:抠 JSON → 解析 → 归一化 → 算指标。
    def parse_response(self, judge_text: str) -> Dict:
        json_blob = self._extract_json_blob(judge_text)
        #〔讲解〕没抠到 JSON 就直接兜底,并把裁判原文塞进 judge_text 便于事后排查。
        if not json_blob:
            return self.default_response(err_msg=judge_text)

        #〔讲解〕尝试解析 JSON;裁判常生成不合法 JSON(多逗号/漏引号),解析失败同样兜底,避免整批评测中断。
        try:
            parsed = json.loads(json_blob)
        except json.JSONDecodeError:
            return self.default_response(err_msg=judge_text)

        # Supports both prompt formats:
        # - nested {"Answer Correctness": {...}}
        # - flat {"correctness_details": ..., "excessive_answers": ...}
        #〔讲解〕兼容两种 JSON 形态:有 "Answer Correctness" 外层就取它,否则把整个 parsed 当内容(扁平格式)。
        correctness = parsed.get("Answer Correctness", parsed)
        #〔讲解〕取逐项明细,大小写/命名两种 key 都试一遍,都没有则给空字典。
        details = correctness.get("Correctness Details", correctness.get("correctness_details", {}))
        #〔讲解〕取多余答案列表,同样两种命名都兜。
        excessive_answers = correctness.get("Excessive Answers", correctness.get("excessive_answers", []))
        #〔讲解〕取置信度:先在 correctness 里找,再退到顶层 parsed,最后默认 0。
        confidence = correctness.get("confidence", parsed.get("confidence", 0))

        #〔讲解〕把明细归一化成布尔列表(此处只要 ratings,答案文字用 _ 丢弃)。
        ratings, _ = self._normalize_correctness_details(details)
        #〔讲解〕防御:裁判若没把 excessive 给成 list(给了 None/字符串等),强制改回空 list 以免 len() 出错。
        if not isinstance(excessive_answers, list):
            excessive_answers = []

        #〔讲解〕用归一化后的命中列表 + 多余答案个数,算出全套指标。
        metrics = self._calculate_metrics(ratings, excessive_answers_count=len(excessive_answers))

        #〔讲解〕把指标 + 裁判原文 + 置信度打包成统一结构返回(再转 dict)。
        return DeepSearchQAJudgeResponse(
            all_correct=metrics["all_correct"],
            correct_with_excessive_answers=metrics["correct_with_excessive_answers"],
            fully_incorrect=metrics["fully_incorrect"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            judge_text=judge_text,
            confidence=confidence,
        ).to_dict()

    #〔讲解〕对外主入口(异步):给一道题打分。async 是因为要 await 网络请求(调裁判模型),便于大批量并发评测。
    #〔讲解〕入参:prediction=被评答案;item=题目;llm=裁判模型名;error/err_msg=上游 rollout 是否已出错及其信息。
    async def compute_score(self, prediction: str, item: Dict, llm: str = "gpt-4.1", error: bool = False, err_msg: str = "") -> Dict:
        """Compute evaluation score using LLM judge."""
        #〔讲解〕若上游生成阶段就报错了,不浪费裁判调用,直接兜底;但仍补抓一次置信度(从被评答案里抽)。
        if error:
            result = self.default_response(err_msg=err_msg)
            if "confidence" not in result:
                result["confidence"] = await self.extract_confidence(prediction, llm=llm)
            return result

        judge_prompt = self.build_prompt(prediction, item)

        #〔讲解〕构造调用裁判模型的请求体。注意 model 先写死 "gpt-4.1",下面再按 llm 名覆盖成带 provider 前缀的正式名。
        body = {
            "model": "gpt-4.1",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": judge_prompt}]
        }

        #〔讲解〕按裁判模型名路由到不同 provider,并塞对应 API key(litellm 用 "provider/model" 格式区分厂商)。
        if "gemini" in llm:
            body["model"] = f"gemini/{llm}"
            body["api_key"] = os.getenv("GEMINI_API_KEY")
        elif "gpt" in llm:
            body["model"] = f"openai/{llm}"
            body["api_key"] = os.getenv("OPENAI_API_KEY")
        #〔讲解〕其它(如本地自托管模型)走默认地址,key 填占位 "EMPTY"(本地推理服务通常不校验)。
        else:
            body["api_key"] = "EMPTY"

        #〔讲解〕真正发起异步调用并解析。整段包在 try 里:任何网络/解析异常都不该让整批评测崩,统一走兜底。
        try:
            judge_response = await litellm.acompletion(**body)
            #〔讲解〕取裁判返回的第一条 message 文本内容(litellm 兼容 OpenAI 的 choices[0].message.content 结构)。
            judge_text = judge_response.choices[0].message.content

            result = self.parse_response(judge_text)
            #〔讲解〕若解析结果里没有 confidence,再单独发一次小请求从被评答案里抽置信度补上(逻辑在 base.py)。
            if "confidence" not in result:
                result["confidence"] = await self.extract_confidence(prediction, llm=llm)

            return result
        except Exception as e:
            #〔讲解〕兜底:把异常信息写进 judge_text,并照样补抓置信度,保证返回结构完整可统计。
            result = self.default_response(err_msg=f"Error: {str(e)}")
            if "confidence" not in result:
                result["confidence"] = await self.extract_confidence(prediction, llm=llm)
            return result
