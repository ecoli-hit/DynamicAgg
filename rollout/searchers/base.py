# 〔讲解〕模块说明:
# 〔讲解〕本文件定义了所有「检索器(Searcher)」的抽象基类 BaseSearcher,位于 rollout/searchers/ 检索层。
# 〔讲解〕在 pipeline 里属于 rollout 环节的底层工具:rollout 跑 ReAct 搜索轨迹时,Agent 需要「查知识源」的能力,
# 〔讲解〕这个能力就由各种具体检索器(如 BM25、FAISS 向量检索)提供。本文件只规定「检索器必须长什么样」(接口契约),
# 〔讲解〕不含具体实现。上游:rollout 的 ReAct agent / tools 会调用检索器的 search/get_document;下游:子类(bm25.py、faiss.py 等)继承本类填实现。
"""
Abstract base class for search implementations.
"""

import argparse
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


# 〔讲解〕BaseSearcher:所有检索器的抽象基类。
# 〔讲解〕职责 = 用「抽象方法」钉死一套统一接口,逼着每个具体检索器都实现同样的方法名/签名/返回结构,
# 〔讲解〕这样上层 ReAct agent 不必关心底下是 BM25 还是向量检索,换检索器无需改调用代码(多态)。
# 〔讲解〕继承 ABC(Abstract Base Class):被标了 @abstractmethod 的方法若子类没实现,该子类就无法被实例化,启动即报错。
class BaseSearcher(ABC):
    """Abstract base class for all search implementations."""

    # 〔讲解〕parse_args:声明这个检索器需要哪些命令行参数(如索引路径、模型名等),由子类往 parser 里 add_argument。
    # 〔讲解〕用 @classmethod 是因为「要解析哪些参数」是类级别的事,在还没造出实例(实例要靠参数才能造)前就得先调用它;返回 None,只产生副作用(改 parser)。
    @classmethod
    @abstractmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add searcher-specific arguments to the argument parser."""
        pass

    # 〔讲解〕__init__:用上一步解析好的 args 真正初始化检索器(如加载索引、连接服务、读模型)。
    # 〔讲解〕标 @abstractmethod 是强制子类自己写构造逻辑——不同检索器初始化方式天差地别,基类不给默认实现。
    @abstractmethod
    def __init__(self, args):
        """Initialize the searcher with parsed arguments."""
        pass

    # 〔讲解〕search:核心检索方法。给定查询串 query,返回 top-k 条命中结果。
    # 〔讲解〕关键约定(返回结构):每条结果是 dict,固定字段 {"docid": 文档ID, "score": 相关度分, "snippet": 文档片段}。
    # 〔讲解〕这是 agent「搜一下」时拿到的东西,字段统一才能让上层和聚合层稳定解析。
    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform search and return results.

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            List of search results with format: {"docid": str, "score": float, "snippet": str}
        """
        pass

    # 〔讲解〕get_document:按 docid 取「整篇」文档全文。search 只给截断的 snippet,agent 想看全文就用这个深读。
    # 〔讲解〕返回 {"docid", "snippet"} 或 None(查不到该 ID 时返回 None,让调用方能区分「空文档」和「没这篇」)。
    @abstractmethod
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full document by ID.

        Args:
            docid: Document ID to retrieve

        Returns:
            Document dictionary with format: {"docid": str, "snippet": str} or None if not found
        """
        pass

    # 〔讲解〕search_type:只读属性,返回检索器类型字符串(如 'BM25'、'FAISS')。
    # 〔讲解〕@property + @abstractmethod 组合 = 强制子类提供一个「像属性一样访问」的标识(self.search_type 而非 self.search_type()),
    # 〔讲解〕常用于日志/记录这条轨迹用了哪种检索方式,便于实验复现与归因。
    @property
    @abstractmethod
    def search_type(self) -> str:
        """Return the type of search (e.g., 'BM25', 'FAISS')."""
        pass

    # 〔讲解〕search_description:给 LLM 看的「search 工具说明书」。注意它【不是】抽象方法——基类直接给了默认实现,子类可不重写。
    # 〔讲解〕作用:在把 search 注册成 ReAct 可调用工具时,这段文字告诉模型这个工具干啥、返回啥字段,模型据此决定何时调用、怎么用返回值。
    # 〔讲解〕k 会被插进描述里(top-{k}),让模型知道一次能拿几条。
    def search_description(self, k: int = 10) -> str:
        """
        Description of the search tool to be passed to the LLM.
        """
        return f"Perform a search on a knowledge source. Returns top-{k} hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits)."

    # 〔讲解〕get_document_description:同理,给 LLM 看的「get_document 工具说明书」,非抽象、有默认实现。
    # 〔讲解〕告诉模型「可以用 docid 取整篇文档」,配合上面的 search 形成「先搜片段→再按需取全文」的检索范式。
    def get_document_description(self) -> str:
        """
        Description of the get_document tool to be passed to the LLM.
        """
        return "Retrieve a full document by its docid."