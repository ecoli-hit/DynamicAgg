#〔讲解〕本文件 = AggAgent 的 evaluation/ 判分层,专门给 BrowseComp 这个"难网页检索问答"基准打分。
#〔讲解〕在 pipeline 里位于最末端:rollout/ 跑出搜索轨迹 → aggregation/+aggagent/ 聚合成最终答案 → 本文件用一个"大模型评委(LLM-as-judge)"判断这份答案对不对。
#〔讲解〕核心做法:把(题目+模型预测+标准答案)塞进一段评委 prompt,调 LLM 评委,要求它只输出 JSON,再解析成 correct/incorrect 的结构化结论。
#〔讲解〕上游:聚合层产出的 prediction 文本;下游:统计准确率/置信度的脚本。本文件不改答案,只判分。
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Dict

#〔讲解〕litellm = 统一封装各家 LLM API 的库,用它一套接口就能调 OpenAI/Gemini/本地模型;此处用来调"评委模型"。
import litellm
#〔讲解〕load_dotenv 把 .env 文件里的密钥(如 OPENAI_API_KEY)读进环境变量,后面 os.getenv 才取得到。
from dotenv import load_dotenv

#〔讲解〕Evaluator 是评测器基类(在同目录 base.py),本文件的 BrowseCompEvaluator 继承它并填充 BrowseComp 专属逻辑。
from .base import Evaluator

#〔讲解〕模块导入时立刻执行一次,确保 .env 里的密钥在任何评委调用前就已注入环境。
load_dotenv()

#〔讲解〕被测模型应遵循的回答格式(Explanation/Exact Answer/Confidence 三段)。评委后续就靠这个格式从回答里抽"最终答案"和"置信度"。
BROWSECOMP_INSTRUCTION = """Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

#〔讲解〕评委 prompt 模板:命令评委"只依据标准答案"判对错,抽取最终答案、给理由、定 correct/incorrect、取置信度,并严格只输出指定 JSON。
#〔讲解〕注意 {question}/{response}/{ground_truth} 是 .format 占位符;而 {{...}} 是转义后的真实花括号(输出 JSON 用),不会被 format 替换。
#〔讲解〕correct_answer 可含多个用 "OR" 分隔的答案,命中任一即算对——这是 BrowseComp 标注的特点。
GRADER_TEMPLATE = """
You are an evaluator. Based ONLY on the [correct_answer], judge whether the [response] to the [question] is correct.

=== INPUTS ===
[question]: {question}
[response]: {response}
[correct_answer]: {ground_truth}

=== TASK ===
1. Extract the single final answer from the [response]. If no clear final answer exists, write "None".
2. Give a concise explanation (reasoning) that ONLY compares the extracted answer with the [correct_answer]. Do not solve the problem again or add extra background.
3. Decide correctness: set correctness = correct if they are equivalent / within a tiny numeric tolerance and acceptable difference of expression style; otherwise incorrect. [correct_answer] may contain multiple answers separated by "OR", the response is correct if it matches any of the answers.
4. Extract a confidence score (0-100). If the [response] provides none, use 0.

=== OUTPUT FORMAT (STRICT) ===
Return a valid JSON object with exactly these keys:
{{
  "extracted final answer": <string>,
  "reasoning": <string>,
  "correctness": <string "correct" or "incorrect">,
  "confidence": <integer 0-100>
}}

Do NOT output anything else|no comments, no code fences.
""".strip()


#〔讲解〕用 dataclass 装一条评委判分结果的结构化容器:抽取的最终答案、对错结论、评委原文、理由、置信度。
#〔讲解〕dataclass 自动生成 __init__ 等样板;confidence 给了默认值 0,故为可选字段。
@dataclass
class BrowseCompJudgeResponse:
    extracted_final_answer: str
    correctness: str
    judge_text: str
    reasoning: str
    confidence: int = 0

    #〔讲解〕把这条结果转成普通 dict,方便序列化进 json / 写结果文件。注意键名顺序与字段名是手写指定的(非自动)。
    def to_dict(self) -> Dict:
        return {
            "extracted_final_answer": self.extracted_final_answer,
            "correctness": self.correctness,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "judge_text": self.judge_text,
        }


#〔讲解〕BrowseComp 专属评测器:继承基类 Evaluator,实现"如何拼评委 prompt / 出错时的兜底结果 / 解析评委输出 / 主流程打分"四件事。
class BrowseCompEvaluator(Evaluator):
    #〔讲解〕构造评委 prompt:从数据样本 item 取标准答案与题目,连同模型预测 prediction 一起填进 GRADER_TEMPLATE。
    #〔讲解〕用 .get 取值并给默认空串,样本缺字段也不会抛 KeyError。
    def build_prompt(self, prediction: str, item: Dict) -> str:
        answer = item.get("answer", "")
        question = item.get("question", "")
        return GRADER_TEMPLATE.format(question=question, response=prediction, ground_truth=answer)

    #〔讲解〕兜底/默认结果:当上游报错或评委调用失败时,返回一条"判为错(no)、置信度 0"的占位结果,保证 pipeline 不中断。
    #〔讲解〕err_msg 会塞进 judge_text 充当错误说明,便于事后排查。
    def default_response(self, err_msg: str = "") -> Dict:
        return BrowseCompJudgeResponse(
            extracted_final_answer="None",
            correctness="no",
            confidence=0,
            reasoning="None",
            judge_text=err_msg or "Error",
        ).to_dict()

    #〔讲解〕解析评委返回的文本:按要求评委应只输出纯 JSON,这里直接 json.loads 解析。
    #〔讲解〕注意 JSON 里的键是带空格的 "extracted final answer",取出后映射到字段名 extracted_final_answer。若评委没给某键则用默认值兜底。
    def parse_response(self, judge_text: str) -> Dict:
        parsed = json.loads(judge_text)
        extracted_final_answer = parsed.get("extracted final answer", "None")
        reasoning = parsed.get("reasoning", "None")
        correctness = parsed.get("correctness", "incorrect")
        confidence = parsed.get("confidence", 0)

        return BrowseCompJudgeResponse(
            extracted_final_answer=extracted_final_answer,
            correctness=correctness,
            confidence=confidence,
            reasoning=reasoning,
            judge_text=judge_text,
        ).to_dict()

    #〔讲解〕评测主入口(异步):给一条预测 prediction 打分。llm 指定评委模型(默认 gpt-4.1);error/err_msg 表示上游本身已出错。
    #〔讲解〕async 是因为要 await 网络调用(评委 API),这样多条样本能并发评测、不互相阻塞。
    async def compute_score(self, prediction: str, item: Dict, llm: str = "gpt-4.1", error: bool = False, err_msg: str = "") -> Dict:
        """Compute evaluation score using LLM judge."""
        #〔讲解〕短路分支:若上游(如 rollout/聚合)已失败,根本不调评委,直接返回兜底结果。
        if error:
            result = self.default_response(err_msg=err_msg)
            #〔讲解〕万一兜底结果缺 confidence,再单独调 extract_confidence(基类方法)从预测文本里抠一个置信度补上。
            if "confidence" not in result:
                result["confidence"] = await self.extract_confidence(prediction, llm=llm)
            return result

        judge_prompt = self.build_prompt(prediction, item)

        #〔讲解〕litellm 调用的请求体:固定评委模型、限制输出长度、把评委 prompt 作为单条 user 消息。下面会按 llm 名覆盖 model/api_key。
        body = {
            "model": "gpt-4.1",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": judge_prompt}]
        }

        #〔讲解〕按评委模型名分流路由:litellm 要求 "provider/model" 前缀来识别厂商,并配对应密钥。
        if "gemini" in llm:
            #〔讲解〕Gemini 系:加 "gemini/" 前缀,取 Google 的密钥。
            body["model"] = f"gemini/{llm}"
            body["api_key"] = os.getenv("GEMINI_API_KEY")
        elif "gpt" in llm:
            #〔讲解〕GPT 系:加 "openai/" 前缀,取 OpenAI 的密钥。
            body["model"] = f"openai/{llm}"
            body["api_key"] = os.getenv("OPENAI_API_KEY")
        else:
            #〔讲解〕其余情况(如本地/自建 OpenAI 兼容服务):密钥占位为 "EMPTY",依赖 litellm 的其它配置(如 base_url)指向本地端点。
            body["api_key"] = "EMPTY"

        #〔讲解〕真正调评委并解析。整个网络调用包在 try 里,任何异常都转成兜底结果,绝不让单条样本异常拖垮整轮评测。
        try:
            #〔讲解〕await 异步发起评委补全请求;**body 把上面拼好的字典展开成关键字参数传入。
            judge_response = await litellm.acompletion(**body)
            #〔讲解〕取出评委回复的正文文本(OpenAI 风格响应结构:choices[0].message.content)。
            judge_text = judge_response.choices[0].message.content

            result = self.parse_response(judge_text)
            #〔讲解〕同样的置信度兜底:解析结果若缺 confidence,就从预测文本另行抽取补齐。
            if "confidence" not in result:
                result["confidence"] = await self.extract_confidence(prediction, llm=llm)

            return result
        except Exception as e:
            #〔讲解〕捕获一切异常(网络失败 / JSON 解析失败等),带上异常信息走兜底,保证返回结构与正常路径一致。
            result = self.default_response(err_msg=f"Error: {str(e)}")
            if "confidence" not in result:
                result["confidence"] = await self.extract_confidence(prediction, llm=llm)
            return result
