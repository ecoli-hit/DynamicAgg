"""
Placeholder for custom searcher. Implement your own searcher here.
"""

#〔讲解〕本文件是「自定义检索器」的模板/占位实现，属于 pipeline 的 rollout 环(rollout/=跑 ReAct 搜索轨迹)。
#〔讲解〕rollout 阶段里的 Agent 要靠「searcher(检索器)」去查资料；不同知识源(网页/本地索引/向量库)各写一个 searcher。
#〔讲解〕本文件不是真正能用的检索器，而是给开发者照着抄的脚手架：继承 BaseSearcher 后，把每个方法的占位逻辑换成你自己的实现即可。
#〔讲解〕上游=调度 ReAct 轨迹的 rollout 代码会按 search_type 选用某个 searcher；下游=被聚合环(aggregation/aggagent)消费的轨迹里就包含这里 search() 返回的检索结果。

import logging
from typing import Any, Dict, Optional

#〔讲解〕从同目录 base 模块导入抽象基类 BaseSearcher：所有检索器(网页/本地/自定义)都继承它，从而拥有统一的对外接口(search / get_document / search_type 等)。
from .base import BaseSearcher

logger = logging.getLogger(__name__)


#〔讲解〕CustomSearcher：自定义检索器的样板类。职责=把"接收 query → 返回 top-k 命中文档"这套接口实现出来，让 ReAct Agent 能像调工具一样调用它。
#〔讲解〕角色=可插拔的检索后端之一；框架通过 search_type=="custom" 把它和其它检索器区分开、按需实例化。
class CustomSearcher(BaseSearcher):
    #〔讲解〕parse_args：类方法，约定在这里往 CLI 的 argparse parser 上挂本检索器需要的命令行参数(如索引路径、API key 等)。
    #〔讲解〕用 @classmethod 是因为参数声明发生在「实例化之前」——框架先解析 CLI，再用解析结果 args 去 __init__ 造对象。此处为占位，pass 表示模板没有参数要声明。
    @classmethod
    def parse_args(cls, parser):
        # Parse arguments from CLI that you will need to instantiate the searcher
        pass

    #〔讲解〕__init__：构造函数。入参 args 即上面 parse_args 解析出的命令行参数集合；把它存到 self.args 供后续 search/get_document 取用(如读取索引路径)。
    def __init__(self, args):
        # Initialize the searcher with the arguments
        self.args = args
        # ...

    #〔讲解〕search：检索器的核心方法，也是 ReAct Agent 调用最频繁的工具。入参 query=查询串，k=要返回的命中条数(默认 10)；返回值=按相关度排序的命中列表。
    #〔讲解〕返回的每条命中是一个 dict，约定字段：docid(文档唯一 id，必填，后续可用它去 get_document 取全文)、score(相关度分，可选)、text(片段/正文，必填，给 Agent 阅读)。
    #〔讲解〕模板里返回的是写死的占位结果，真实实现要把这段换成「调用你的搜索引擎/索引并拼成此格式」的逻辑。
    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        # Implement your search logic here
        # ...
        return [
            {
                "docid": "place-holder-docid",  # Required
                "score": 0.0,  # Optional
                "text": "place-holder-text",  # Required
            },
            # ...
        ]

    #〔讲解〕get_document：按 docid 取「整篇文档」。search 返回的片段(snippet)可能被截断，Agent 想看全文时就用这个方法按 id 回查。
    #〔讲解〕入参 docid=文档 id；返回 dict(含 docid 与完整 text)，取不到时按约定返回 None。此处为占位，直接回填一个假文档。
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        # Implement your logic for getting a full document by id
        # ...
        return {
            "docid": docid,
            "text": "place-holder-text",
        }

    #〔讲解〕search_type：只读属性，返回本检索器的类型标识字符串。框架据此名字(此处 "custom")来识别/选择该使用哪一个检索器。
    @property
    def search_type(self) -> str:
        return "custom"

    # Optionally, you can also override the description of your tools to be used in the LLM:
    #〔讲解〕search_description：返回一段「给 LLM 看的工具说明」。在 ReAct 里，工具的自然语言描述会写进 prompt，决定 Agent 是否/如何调用本工具，所以这段话很重要。
    #〔讲解〕可选覆写：不写则用基类默认描述；想让模型更懂这个检索源的特性(返回 top-k、含 docid/score/snippet、片段可能被 token 上限截断)时就在此自定义。
    def search_description(self, k: int = 10) -> str:
        return "Perform a search on a custom knowledge source. Returns top-{k} hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits)."

    #〔讲解〕get_document_description：同理，是「按 id 取全文」这个工具给 LLM 的自然语言说明，告诉模型何时该用 docid 去回查完整文档。
    def get_document_description(self) -> str:
        return "Retrieve a full document by its docid."
