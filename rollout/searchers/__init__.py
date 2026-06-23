# 〔讲解〕本文件 = rollout/searchers 子包的入口(package __init__)。
# 〔讲解〕在 pipeline 里属于 rollout 这一环最底层的「检索后端」:ReAct 智能体跑搜索轨迹时,
#         它的 search 工具最终调用的就是这里注册的某个 Searcher(BM25/FAISS/ReasonIR/自定义)。
# 〔讲解〕核心职责两件:(1) 把各 searcher 实现做「可选导入」——某个依赖没装也不报错,只是该后端不可用;
#         (2) 用一个 Enum 把命令行名字(如 "bm25")映射到对应的 Searcher 类,供上层 argparse 选择和实例化。
"""
Searchers package for different search implementations.
"""

from enum import Enum

# 〔讲解〕BaseSearcher 是所有检索器的抽象基类(各实现都继承它),这里无条件导入并对外暴露,作为统一接口契约。
from .base import BaseSearcher

# Make all searcher imports optional
# 〔讲解〕下面这套「先置 None,再 try import 覆盖」的写法是关键设计:每个后端各有重依赖
#         (BM25 需 rank_bm25、FAISS/ReasonIR 需 faiss+torch 等)。
# 〔讲解〕思路:先把名字默认设为 None;若依赖装好则 import 成功、把 None 覆盖成真正的类;
#         若依赖缺失则 except 吞掉异常,名字保持 None。这样「只装了部分依赖」的环境也能正常 import 本包,
#         不会因为某个后端装不了就整个 crash——真正用到该后端时才在下面 get_searcher_class 里报错。
BM25Searcher = None
try:
    from .bm25_searcher import BM25Searcher
except Exception:
    # 〔讲解〕故意捕获最宽的 Exception 并 pass:导入失败(缺包/编译错)一律静默,让 BM25Searcher 维持 None。
    pass

CustomSearcher = None
try:
    from .custom_searcher import CustomSearcher
except Exception:
    pass

# 〔讲解〕FAISS 与 ReasonIR 两个稠密检索器同住一个模块、共用同一套重依赖,所以放在同一个 try 里一起导入。
FaissSearcher = None
ReasonIrSearcher = None
try:
    from .faiss_searcher import FaissSearcher, ReasonIrSearcher
except Exception:
    pass


# 〔讲解〕SearcherType:用枚举集中管理「有哪些检索后端」以及「CLI 名字 ↔ Searcher 类」的映射表。
# 〔讲解〕角色 = 上层入口与具体实现之间的注册中心/工厂索引。新增一种检索器,只需在此加一个枚举成员即可被 CLI 选中。
class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""

    # 〔讲解〕每个枚举成员的「值」是一个二元组 (CLI名字, 对应的Searcher类)。
    # 〔讲解〕注意第二个元素可能是 None——当上面那个后端依赖缺失、import 失败时,这里就绑定到 None,
    #         代表「这个后端理论上存在但当前环境用不了」,留待取用时再报错(见 get_searcher_class)。
    BM25 = ("bm25", BM25Searcher)
    FAISS = ("faiss", FaissSearcher)
    REASONIR = ("reasonir", ReasonIrSearcher)
    CUSTOM = ("custom", CustomSearcher)

    # 〔讲解〕Enum 的自定义构造:Python 会把成员值元组解包后传进 __init__,
    #         于是每个成员就额外挂上 .cli_name 和 .searcher_class 两个属性,方便后面按名字查、拿类。
    def __init__(self, cli_name, searcher_class):
        self.cli_name = cli_name
        self.searcher_class = searcher_class

    # 〔讲解〕返回所有可选的 CLI 名字列表(如 ["bm25","faiss","reasonir","custom"]),
    #         典型用途是直接喂给 argparse 的 choices=,限定命令行 --searcher 只能填这几个值。
    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        # 〔讲解〕`for searcher_type in cls` 是遍历枚举的所有成员;逐个取出其 .cli_name 拼成列表。
        return [searcher_type.cli_name for searcher_type in cls]

    # 〔讲解〕工厂方法:输入一个 CLI 名字,返回对应的 Searcher 类(注意返回的是「类」本身,不是实例,由调用方再去实例化)。
    # 〔讲解〕这是 CLI 名 → 具体实现的最终落地点,把「用户在命令行选了什么」翻译成「该 new 哪个检索器」。
    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name."""
        # 〔讲解〕线性扫描所有枚举成员,找名字匹配的那个。
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                # 〔讲解〕找到了但类是 None,说明该后端因缺依赖没导入成功——此刻才抛 ImportError,
                #         做到「能 import 本包、能列 choices,真正要用某后端却没装时」再精准报错。
                if searcher_type.searcher_class is None:
                    raise ImportError(f"Searcher '{cli_name}' is not available due to missing dependencies")
                return searcher_type.searcher_class
        # 〔讲解〕循环走完都没匹配上,说明传进来的名字根本不在枚举里(拼错/不支持),抛 ValueError。
        raise ValueError(f"Unknown searcher type: {cli_name}")


# 〔讲解〕__all__ 限定 `from searchers import *` 时只导出这两个名字:统一基类 BaseSearcher 与注册中心 SearcherType。
# 〔讲解〕各具体实现类(BM25Searcher 等)不在此列——它们是「可能为 None」的内部细节,对外统一通过 SearcherType 取用。
__all__ = ["BaseSearcher", "SearcherType"]