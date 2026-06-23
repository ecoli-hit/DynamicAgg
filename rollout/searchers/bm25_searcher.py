#〔讲解〕本文件 = rollout 检索层(rollout/searchers/)里的「BM25 本地检索器」实现。
#〔讲解〕BM25 是经典的关键词匹配打分算法(词频/逆文档频率那一类),不依赖联网,靠预先建好的 Lucene 倒排索引在本地语料里查文档。
#〔讲解〕在 pipeline 里:rollout 阶段的 ReAct 智能体每走一步,会调用某个 searcher 的 search() 去查资料;本文件就是其中一种可选后端(对应离线/本地语料场景,区别于联网的网页搜索)。
#〔讲解〕上游:rollout 主循环 / search 工具按命令行选定的 searcher 类型实例化本类;下游:本类返回的命中文档喂回给 ReAct 智能体当观察(observation)。它继承统一基类 BaseSearcher,以便和其他检索器(如联网搜索)接口一致、可互换。
"""
BM25 searcher implementation using Pyserini.
"""

import json
import logging
from typing import Any, Dict, Optional

#〔讲解〕Pyserini 是把 Java 写的 Lucene 检索引擎封装给 Python 用的库;LuceneSearcher 就是实际执行 BM25 检索的对象。
from pyserini.search.lucene import LuceneSearcher

#〔讲解〕从同目录 base 模块导入抽象基类,本类必须实现它约定的接口(search / get_document / search_type 等),保证所有检索器「长得一样」可替换。
from .base import BaseSearcher

logger = logging.getLogger(__name__)


#〔讲解〕BM25Searcher:把 Pyserini 的本地 Lucene BM25 检索包装成框架统一的 searcher 接口。
#〔讲解〕职责 = ①初始化时加载本地 Lucene 索引 ②search() 按 query 返回 topk 命中 ③get_document() 按文档 id 取全文。
class BM25Searcher(BaseSearcher):
    #〔讲解〕parse_args:类方法,往命令行解析器里注册本检索器特有的参数。框架在启动时调用它,让用户能在命令行指定索引路径。
    @classmethod
    def parse_args(cls, parser):
        #〔讲解〕注册 --index-path 参数(必填):指向本地 Lucene 索引,可以是预建索引名(如 msmarco-v1-passage)或磁盘路径。
        parser.add_argument(
            "--index-path",
            required=True,
            help="Name or path of Lucene index (e.g. msmarco-v1-passage).",
        )

    #〔讲解〕构造函数:接收已解析好的命令行参数对象 args,完成「校验路径 → 加载索引 → 备好检索器」三步。
    def __init__(self, args):
        #〔讲解〕双重保险:即使 argparse 标了 required,这里再挡一次空值,缺索引路径直接报错而非后面莫名其妙崩。
        if not args.index_path:
            raise ValueError("index_path is required for BM25 searcher")

        self.args = args
        #〔讲解〕先把检索器置空;若下面加载失败,self.searcher 保持 None,search()/get_document() 里会据此拒绝服务。
        self.searcher = None

        logger.info(f"Initializing BM25 searcher with index: {args.index_path}")

        #〔讲解〕用 try 包住索引加载:Lucene 索引加载可能因路径不存在/格式不对而抛各种异常。
        try:
            #〔讲解〕真正加载索引、构建检索器对象,这一步成功后才能查;开销主要在这里(读磁盘上的倒排索引)。
            self.searcher = LuceneSearcher(args.index_path)
        #〔讲解〕捕获任意底层异常,统一转成语义清晰的 ValueError;`from exc` 保留原始异常链,方便排查根因。
        except Exception as exc:
            raise ValueError(
                f"Index '{args.index_path}' is not a valid local Lucene index path."
            ) from exc

        logger.info("BM25 searcher initialized successfully")

    #〔讲解〕search:检索主入口。输入查询字符串 query 和想要的条数 k,返回 topk 命中文档列表(每条含 docid/score/text)。这是 ReAct 智能体每步搜索实际调用的方法。
    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        #〔讲解〕防御:索引没加载成功就调用,直接抛错,避免对 None 调方法触发更隐晦的报错。
        if not self.searcher:
            raise RuntimeError("Searcher not initialized")

        #〔讲解〕调 Pyserini 执行 BM25 检索,返回按相关性排序的前 k 个命中对象(hits),每个 hit 带 docid、score 及原始文档。
        hits = self.searcher.search(query, k)
        results = []

        #〔讲解〕逐条把 Lucene 原生 hit 转成框架统一的字典格式。
        for hit in hits:
            #〔讲解〕Lucene 把整篇文档以 JSON 字符串存在 "raw" 字段;这里取出并反序列化成 dict 才能拿到正文。
            raw = json.loads(hit.lucene_document.get("raw"))
            #〔讲解〕组装统一三元组:docid(文档号)、score(BM25 相关性分,越高越相关)、text(取 JSON 里的 contents 即正文)。
            results.append(
                {"docid": hit.docid, "score": hit.score, "text": raw["contents"]}
            )

        return results

    #〔讲解〕get_document:按文档 id 精确取回单篇全文(不是检索,是直查)。用于智能体已知 docid、想读全文细节的场景。返回字典或 None(查无此文)。
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        #〔讲解〕同样的未初始化防御。
        if not self.searcher:
            raise RuntimeError("Searcher not initialized")

        #〔讲解〕按 docid 直接定位文档对象;Pyserini 查不到会返回 None。
        doc = self.searcher.doc(docid)
        if doc is None:
            return None

        #〔讲解〕返回统一格式:doc.raw() 是该文档的原始 JSON 字符串,反序列化后取 contents 当正文。
        return {
            "docid": docid,
            "text": json.loads(doc.raw())["contents"],
        }

    #〔讲解〕search_type:只读属性,返回本检索器的类型标识字符串。供框架/日志区分当前用的是哪种检索后端。
    @property
    def search_type(self) -> str:
        return "BM25"
