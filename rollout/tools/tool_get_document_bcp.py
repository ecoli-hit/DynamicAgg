# 〔讲解〕本文件定义 ReAct 轨迹里的 get_document 工具:输入一个文档 id(docid),返回该文档的【完整正文】。
# 〔讲解〕在 pipeline 中属于 rollout/tools 这一环——给 ReAct 搜索智能体提供的"动作"之一。
# 〔讲解〕上游:同目录的 search 工具(tool_search_bcp.py)返回 top-5 命中,每条带 docid + 截断后的 snippet(片段)。
# 〔讲解〕本工具的存在意义:snippet 受 token 上限截断,智能体看片段觉得相关时,可凭 docid 调本工具把整篇文档拉全。
# 〔讲解〕下游:返回的整篇文本进入 ReAct 的 observation,供智能体继续推理 / 决定下一步动作。
import asyncio
from typing import Optional, Union

# 〔讲解〕qwen_agent 是这套 ReAct 框架的底座;BaseTool 是工具基类,register_tool 把工具登记进框架的工具注册表。
from qwen_agent.tools.base import BaseTool, register_tool

# 〔讲解〕复用 search 工具里的 SearchBCP——它持有一个全局唯一的 searcher(检索器)单例。get_document 不另建检索器,直接借用同一个。
from tools.tool_search_bcp import SearchBCP

# 〔讲解〕装饰器:把本类以工具名 "get_document" 注册;allow_overwrite=True 允许覆盖同名旧注册(重复 import 时不报错)。
@register_tool("get_document", allow_overwrite=True)
# 〔讲解〕GetDocumentBCP:get_document 工具的实现类。职责=按 docid 取整篇文档。BCP 是这套数据/检索方案的内部代号。
class GetDocumentBCP(BaseTool):
    # 〔讲解〕name/description/parameters 是给 LLM 看的工具"说明书":模型据此决定何时调用、传什么参数。
    name = "get_document"
    description = "Retrieve a full document by its docid."
    # 〔讲解〕parameters 是 JSON Schema:声明本工具只收一个必填字符串参数 docid。框架据此校验模型生成的入参。
    parameters = {
        "type": "object",
        "properties": {
            "docid": {
                "type": "string",
                "description": "Document ID to retrieve",
            }
        },
        "required": ["docid"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    # 〔讲解〕call:工具被调用时的实际执行入口(异步)。params=模型生成的入参,返回值=字符串形式的 observation。
    async def call(self, params: Union[str, dict], **kwargs) -> str:
        # 〔讲解〕容错:模型给的入参可能格式不对(缺 docid / 不是 dict),取不到就返回友好报错而非抛异常中断轨迹。
        try:
            docid = params["docid"]
        except:
            return "[get_document] Invalid request format: Input must be a JSON object containing 'docid' field"

        try:
            # 〔讲解〕拿到全局共享的 searcher 单例(懒加载,首次调用时才真正构建检索器,后续复用)。
            searcher = await SearchBCP.get_searcher()
            # 〔讲解〕searcher.get_document 是阻塞式调用;用 asyncio.to_thread 丢到线程池跑,避免卡住事件循环(让并发轨迹不被堵塞)。
            result = await asyncio.to_thread(searcher.get_document, docid)
            # 〔讲解〕检索器返回 None = 没有这个 docid(可能是模型编造或写错的 id),返回 not found 提示让模型纠正。
            if result is None:
                return f"[get_document] Document with docid '{docid}' not found"
        # 〔讲解〕兜底:检索器内部出错时统一返回 server error,不把堆栈泄露给模型,保证轨迹能继续往下走。
        except Exception:
            return "[get_document] server error"

        # 〔讲解〕正常路径:把整篇文档原文返回,作为这一步的 observation。
        return result

    # 〔讲解〕get_tool_definitions:把工具导出成 OpenAI function-calling 格式的定义,供注入到 LLM 请求里。
    def get_tool_definitions(self):
        # 〔讲解〕copy 一份再改,避免污染类属性 parameters(共享可变对象被改会影响所有实例)。
        parameters = self.parameters.copy()
        # 〔讲解〕additionalProperties=False:禁止模型在入参里塞 schema 之外的多余字段,收紧约束、减少幻觉参数。
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }
