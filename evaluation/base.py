# 〔讲解〕本文件是 evaluation/ 这一环的「评估器抽象基类 + LLM 调用底座」。
# 〔讲解〕在 pipeline 里:rollout/ 跑出搜索轨迹、aggregation/+aggagent/ 把多条轨迹聚合成一份最终答案(prediction),
# 〔讲解〕然后由 evaluation/ 用「大模型当裁判(LLM-as-judge)」给这份答案判分。
# 〔讲解〕本文件不绑定任何具体任务,只提供:① 统一的 litellm 调用入口(支持多家模型 + 本地 vLLM)
# 〔讲解〕② 抽取置信度的小工具 ③ Evaluator 抽象类(定义所有任务评估器必须实现的 4 个接口)。
# 〔讲解〕上游 = 聚合后的 prediction + 数据集 item;下游 = 各任务子类(browsecomp/deepsearchqa/healthbench/researchrubrics)继承本类落地。
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import re
from typing import Any, Dict, Optional

# 〔讲解〕litellm 是「一个 SDK 调所有家大模型」的统一封装库;下面 _build_litellm_body 就是给它拼调用参数。
import litellm


# 〔讲解〕下面这个三引号字符串是「让裁判模型从一段回答里抠出 Confidence 数字」的 prompt 模板。
# 〔讲解〕{prediction} 是占位符,运行时用 .format 填入被评估的回答;末尾要求模型只回 JSON,便于程序解析。
CONFIDENCE_EXTRACTION_PROMPT = """Extract the confidence score from the AI response below.
Find the line starting with "Confidence:" and return its integer value (0-100).

AI Response:
{prediction}

Return ONLY a JSON object: {{"confidence": <integer>}}
If no confidence line is found, return {{"confidence": 0}}."""


# 〔讲解〕_build_litellm_body:根据模型名 llm 自动判断「该走哪家厂商、该填什么 key/地址」,拼出 litellm 调用所需的 body 字典。
# 〔讲解〕这是「provider routing(供应商路由)」——同一份代码靠模型名里的关键字(gemini/gpt/Qwen…)切到不同后端。
# 〔讲解〕入参:llm=模型名;messages=对话消息列表;max_tokens=最大输出长度;**extra=额外参数(如 temperature)。返回拼好的 body dict。
def _build_litellm_body(llm: str, messages: list, max_tokens: int, **extra) -> Dict[str, Any]:
    """Build litellm body with provider routing."""
    # 〔讲解〕先放公共字段,再把调用方传的 extra 合并进来(extra 可覆盖默认)。
    body: Dict[str, Any] = {"max_tokens": max_tokens, "messages": messages}
    body.update(extra)
    # 〔讲解〕分支 1:模型名含 "gemini" → 走 Google Gemini,模型名加 "gemini/" 前缀,key 从环境变量 GEMINI_API_KEY 取。
    if "gemini" in llm:
        body["model"] = f"gemini/{llm}"
        body["api_key"] = os.getenv("GEMINI_API_KEY")
    # 〔讲解〕分支 2:模型名含 "gpt"/"openai" → 走 OpenAI;若已带 "openai/" 前缀则不重复加,否则补上。
    elif "gpt" in llm or "openai" in llm:
        body["model"] = f"openai/{llm}" if not llm.startswith("openai/") else llm
        body["api_key"] = os.getenv("OPENAI_API_KEY")
    # 〔讲解〕分支 3:模型名含 "Qwen" → 走本地自建的 vLLM 服务(hosted_vllm 前缀)。
    elif "Qwen" in llm:
        body["model"] = f"hosted_vllm/{llm}"
        # 〔讲解〕api_base 指向本地推理服务地址,默认 localhost:7000,可用环境变量 EVAL_API_BASE 覆盖。
        body["api_base"] = os.getenv("EVAL_API_BASE", "http://localhost:7000/v1")
        # 〔讲解〕setdefault = 只在调用方没传时才设默认值,不覆盖 extra 里已给的采样参数。
        body.setdefault("temperature", 0.7)
        body.setdefault("top_p", 0.8)
        # 〔讲解〕给 Qwen 关掉「思考模式(thinking)」——评分场景要直接出结论,不需要长链推理拖慢/扰乱输出。
        body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    # 〔讲解〕兜底分支:未知模型名 → 原样当 model,key 填 "EMPTY"(典型用于本地无鉴权的 OpenAI 兼容服务)。
    else:
        body["model"] = llm
        body["api_key"] = "EMPTY"
    return body


# 〔讲解〕Evaluator:所有任务评估器的抽象基类(ABC = Abstract Base Class,不能直接实例化)。
# 〔讲解〕它把「怎么调 LLM」「怎么抽置信度」这些公共能力实现好,把「怎么搭判分 prompt、怎么解析裁判输出、怎么算分」
# 〔讲解〕留成抽象方法交给各任务子类。角色 = 评估这一环的统一骨架/模板。
class Evaluator(ABC):
    # 〔讲解〕_completer:可选的「替身调用器」。默认 None 走 litellm;若被注入,则用它(而非 litellm)来完成 LLM 调用。
    # 〔讲解〕注释里的 TinkerMessageCompleter 是一种用本地/训练框架自带的补全器,便于在特定环境下复用同一套评估逻辑。
    _completer: Optional[Any] = None  # TinkerMessageCompleter when set

    # 〔讲解〕set_completer:注入替身调用器的入口。调用后,本评估器的 LLM 请求改走该 completer 而不走 litellm。
    def set_completer(self, completer: Any) -> None:
        """Inject a TinkerMessageCompleter to use instead of litellm."""
        self._completer = completer

    # 〔讲解〕async_complete:统一的「发一次 LLM 请求拿回文本」的异步方法。是本类对外的核心调用入口。
    # 〔讲解〕入参:messages=对话;llm=模型名;max_tokens/extra=透传给后端。返回:模型回复的纯文本字符串。
    async def async_complete(self, messages: list, llm: str, max_tokens: int = 1024, **extra) -> str:
        """Call either the tinker completer or litellm, returning the response text."""
        # 〔讲解〕优先用注入的 completer(它本身是 async 可调用对象),从返回 dict 里取 content 字段。
        if self._completer is not None:
            result = await self._completer(messages)
            return result["content"]
        # 〔讲解〕否则走标准路径:拼 body → 异步调 litellm。
        body = _build_litellm_body(llm, messages, max_tokens, **extra)
        # 〔讲解〕acompletion = litellm 的异步补全接口;**body 把上面拼好的参数全部展开传入。
        resp = await litellm.acompletion(**body)
        # 〔讲解〕从标准 OpenAI 风格响应结构里取出第一个候选的文本内容。
        return resp.choices[0].message.content

    # 〔讲解〕extract_confidence:用一次「短小的 LLM 调用」从回答里抠出 0~100 的置信度数字(给 BON 等按自信度选答案的策略用)。
    # 〔讲解〕入参:prediction=被评估的回答;llm=用哪个小模型来抽(默认 gpt-4.1-mini,便宜)。返回:0~100 的整数。
    async def extract_confidence(self, prediction: str, llm: str = "gpt-4.1-mini") -> int:
        """Short eval call to extract a structured 'Confidence: X%' from the prediction."""
        # 〔讲解〕空回答直接判 0,省掉一次无意义的 LLM 调用。
        if not prediction or not prediction.strip():
            return 0
        # 〔讲解〕把回答填进上面的抽取模板,得到完整 prompt。
        prompt = CONFIDENCE_EXTRACTION_PROMPT.format(prediction=prediction)
        try:
            # 〔讲解〕只要 32 token、temperature=0(确定性输出),让模型回一个 {"confidence": N} 的 JSON。
            text = await self.async_complete(
                [{"role": "user", "content": prompt}], llm, max_tokens=32, temperature=0.0
            )
            # 〔讲解〕模型有时会用 ```json ... ``` 把 JSON 包起来,这里用正则把首尾的代码块围栏去掉,只留纯 JSON。
            json_cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
            parsed = json.loads(json_cleaned)
            # 〔讲解〕取 confidence 字段,缺失则默认 0。
            confidence = parsed.get("confidence", 0)
            # 〔讲解〕是数值才接受;clamp(钳制)到 0~100 区间并转成 int,防止模型给出越界或浮点值。
            if isinstance(confidence, (int, float)):
                return max(0, min(100, int(confidence)))
        # 〔讲解〕任何异常(网络/JSON 解析失败等)都不抛出,只打印并落到下面 return 0——评估流程容错,不因单题崩掉整批。
        except Exception as e:
            print(f"Failed to extract confidence: {e}")
        return 0

    # 〔讲解〕以下 4 个 @abstractmethod 是「契约」:子类必须各自实现,否则无法实例化。它们刻画了一个评估器的完整职责。
    # 〔讲解〕build_prompt:把「被评估的回答 prediction + 数据集条目 item(含题目/参考答案/rubric 等)」组装成给裁判模型的判分 prompt。
    @abstractmethod
    def build_prompt(self, prediction: str, item: Dict) -> str:
        raise NotImplementedError

    # 〔讲解〕parse_response:把裁判模型回的原始文本 judge_text 解析成结构化结果 dict(如 {correct: bool, score: ..., reasoning: ...})。
    @abstractmethod
    def parse_response(self, judge_text: str) -> Dict:
        raise NotImplementedError

    # 〔讲解〕default_response:出错/无法判分时返回的「默认结果」dict;err_msg 记录原因。保证结果结构始终统一,下游不必特判。
    @abstractmethod
    def default_response(self, err_msg: str = "") -> Dict:
        raise NotImplementedError

    # 〔讲解〕compute_score:对外的主入口,串起「搭 prompt → 调裁判 → 解析」全流程,产出该条样本的最终评估结果。
    # 〔讲解〕入参:prediction=回答;item=数据条目;error/err_msg=上游若已出错可直接走默认结果。返回:评估结果 dict。
    @abstractmethod
    async def compute_score(self, prediction: str, item: Dict, error: bool = False, err_msg: str = "") -> Dict:
        """Compute the evaluation score for a prediction."""
        raise NotImplementedError
