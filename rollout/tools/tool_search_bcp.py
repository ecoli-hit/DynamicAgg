#〔讲解〕本文件定义 ReAct rollout 阶段可调用的 "search" 工具(注册名 search)。
#〔讲解〕在 pipeline 里属于 rollout/tools 这一环:rollout 阶段的智能体跑 ReAct 轨迹时,
#〔讲解〕每一步可能发出一次工具调用,这里的 SearchBCP 就是被调到的"本地检索"工具——
#〔讲解〕它接收一个 query 字符串,去 rollout/searchers 下的本地向量索引(默认 FAISS)里查 top-K 文档,
#〔讲解〕再按 token 上限把每条文档正文截成 snippet 返回。下游就是 ReAct 智能体拿这些 snippet 继续推理。
#〔讲解〕"BCP" 可理解为本地语料(corpus)检索的实现变体,与走 serper 等联网搜索的工具相对。
import asyncio
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

#〔讲解〕transformers 的 AutoTokenizer 用来按 token(而非字符)数截断 snippet,保证截断口径和模型一致。
from transformers import AutoTokenizer
#〔讲解〕qwen_agent 的工具基类 + 注册装饰器:register_tool 把本类登记进工具表,智能体才能按名字调用。
from qwen_agent.tools.base import BaseTool, register_tool

#〔讲解〕SearcherType 是检索器工厂:按 SEARCHER_TYPE 字符串拿到对应的检索器类(如 faiss)。
from searchers import SearcherType

#〔讲解〕@register_tool("search", ...) 把这个类注册成名为 "search" 的工具;allow_overwrite=True 允许覆盖同名旧注册,
#〔讲解〕避免重复 import 时报"工具已存在"。注册后 ReAct 智能体在 prompt 里看到的工具名就是 "search"。
@register_tool("search", allow_overwrite=True)
class SearchBCP(BaseTool):
    #〔讲解〕本类是一个 qwen_agent 工具。下面三个类属性(name/description/parameters)就是暴露给 LLM 的"工具说明书":
    #〔讲解〕LLM 据此决定何时调用、传什么参数。这是 function-calling 的标准三件套。
    name = "search"
    description = "Perform a search on a knowledge source. Returns top-5 hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits)."
    #〔讲解〕parameters 是 JSON Schema:声明本工具只收一个必填字段 query(字符串)。LLM 输出的参数会按此 schema 校验。
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            }
        },
        "required": ["query"],
    }

    #〔讲解〕下面三个类级变量做"单例缓存":检索器和 tokenizer 加载很重(读索引、读模型),
    #〔讲解〕用类属性缓存使整个进程只初始化一次,所有调用共享。初始为 None 表示尚未加载。
    _searcher = None
    #〔讲解〕异步锁,防止并发首次调用时重复构建检索器(竞态)。同样懒加载,见 get_searcher。
    _searcher_lock = None
    _snippet_tokenizer = None

    def __init__(self, cfg: Optional[dict] = None):
        #〔讲解〕仅透传给基类构造,本工具的真正配置全部来自环境变量(见下面各 _get_env_* 方法)。
        super().__init__(cfg)

    #〔讲解〕从环境变量读一个 int;读不到或无法转成整数则回退默认值。@classmethod 因为它服务于类级配置。
    @classmethod
    def _get_env_int(cls, key: str, default: int) -> int:
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            #〔讲解〕环境变量写错(非数字)时不崩溃,静默回退默认值——保证 rollout 不因配置笔误中断。
            return default

    #〔讲解〕从环境变量读一个 bool;把常见真值字符串(1/true/yes/y/on)统一识别为 True,其余为 False。
    @classmethod
    def _get_env_bool(cls, key: str, default: bool = False) -> bool:
        value = os.getenv(key)
        if value is None:
            return default
        #〔讲解〕统一转小写后判断是否落在真值集合里;这样大小写、yes/Y/ON 等写法都能正确解析。
        return value.lower() in {"1", "true", "yes", "y", "on"}

    #〔讲解〕从一堆环境变量组装检索器:决定用哪种检索器、索引在哪、用哪个 embedding 模型及其推理设置。
    #〔讲解〕返回一个已实例化的检索器对象。这是把"外部配置"翻译成"检索器构造参数"的集中入口。
    @classmethod
    def _build_searcher_from_env(cls):
        #〔讲解〕SEARCHER_TYPE 选检索器种类(默认 faiss);工厂据此返回对应的检索器类。
        searcher_type = os.getenv("SEARCHER_TYPE", "faiss")
        searcher_cls = SearcherType.get_searcher_class(searcher_type)

        #〔讲解〕用 SimpleNamespace 攒一个"伪 args 对象"(可像 args.index_path 这样点取),
        #〔讲解〕模拟命令行参数命名空间,直接喂给检索器构造函数——这样检索器无需关心配置来自 env 还是 argparse。
        args = SimpleNamespace(
            index_path=os.getenv("INDEX_PATH"),
            model_name=os.getenv("SEARCH_MODEL_NAME"),
            #〔讲解〕normalize:是否对向量做 L2 归一化(影响相似度计算口径),默认 True。
            normalize=cls._get_env_bool("SEARCH_NORMALIZE", True),
            #〔讲解〕pooling:句向量的池化方式,默认 "eos"(取结束符 token 的隐状态,Qwen 系 embedding 常用)。
            pooling=os.getenv("SEARCH_POOLING", "eos"),
            torch_dtype=os.getenv("SEARCH_TORCH_DTYPE", "float16"),
            dataset_name=os.getenv("SEARCH_DATASET_NAME"),
            #〔讲解〕task_prefix:加在 query 前的指令前缀(instruction-tuned embedding 的标准用法),
            #〔讲解〕告诉模型"这是 web 搜索 query,去取能回答它的段落",显著影响检索质量。默认值即官方推荐前缀。
            task_prefix=os.getenv(
                "SEARCH_TASK_PREFIX",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
            ),
            #〔讲解〕max_length:embedding 模型对单条输入的最大 token 长度,默认 8192。
            max_length=cls._get_env_int("SEARCH_MAX_LENGTH", 8192),
        )

        #〔讲解〕索引路径是本地检索的硬前提,缺了直接报错并提示格式,避免后续静默拿空结果。
        if not args.index_path:
            raise ValueError("INDEX_PATH is required for local searcher (e.g. /path/to/corpus.shard*.pkl)")

        return searcher_cls(args)

    #〔讲解〕对外的检索器获取入口(异步):带"双重检查 + 异步锁"的单例懒加载,保证进程内只构建一次且并发安全。
    @classmethod
    async def get_searcher(cls):
        #〔讲解〕第一重检查:已加载就直接返回,连锁都不进,走最快路径。
        if cls._searcher is not None:
            return cls._searcher

        # Lazy initialization of lock (must be done in async context)
        #〔讲解〕锁本身也懒加载——asyncio.Lock() 需在事件循环运行时创建,所以放到这个异步方法里而非类定义处。
        if cls._searcher_lock is None:
            cls._searcher_lock = asyncio.Lock()

        #〔讲解〕拿锁后做第二重检查:防止多个协程同时通过第一重检查后重复构建(经典 double-checked locking)。
        async with cls._searcher_lock:
            if cls._searcher is None:
                #〔讲解〕构建检索器是 CPU/IO 密集的同步操作,用 to_thread 丢到线程池跑,避免阻塞事件循环。
                cls._searcher = await asyncio.to_thread(cls._build_searcher_from_env)
        return cls._searcher

    #〔讲解〕获取用于截断 snippet 的 tokenizer(异步单例)。若上限<=0 表示不截断,直接返回 None。
    @classmethod
    async def get_snippet_tokenizer(cls):
        #〔讲解〕SNIPPET_MAX_TOKENS:每条 snippet 保留的最大 token 数,默认 512;<=0 视为关闭截断。
        snippet_max_tokens = cls._get_env_int("SNIPPET_MAX_TOKENS", 512)
        if snippet_max_tokens <= 0:
            return None
        if cls._snippet_tokenizer is None:
            tokenizer_path = os.getenv("SNIPPET_TOKENIZER_PATH")
            #〔讲解〕加载 tokenizer 同样阻塞,丢线程池;加载后缓存到类属性,后续调用复用。
            cls._snippet_tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, tokenizer_path
            )
        return cls._snippet_tokenizer

    #〔讲解〕工具被 LLM 调用时真正执行的入口(qwen_agent 约定的 call 方法)。
    #〔讲解〕入参 params 是 LLM 给的参数(此处应含 query);返回值会回灌给智能体作为这一步的"观察(observation)"。
    async def call(self, params: Union[str, dict], **kwargs) -> str:
        #〔讲解〕容错取 query:LLM 偶尔会给出不含 query 的非法参数,这里捕获后返回一句可读的错误提示而非抛异常,
        #〔讲解〕让智能体能据此自我纠正下一步动作。
        try:
            query = params["query"]
        except:
            return "[search] Invalid request format: Input must be a JSON object containing 'query' field"

        try:
            #〔讲解〕拿到(或首次构建)检索器,再读 top-K(默认 5)。
            searcher = await self.get_searcher()
            k = self._get_env_int("SEARCH_K", 5)
            #〔讲解〕真正的检索:同步的 searcher.search 同样丢线程池,避免阻塞事件循环。返回候选文档列表。
            candidates = await asyncio.to_thread(searcher.search, query, k)

            #〔讲解〕再次读截断上限,据此决定要不要加载 tokenizer(上限<=0 则不加载)。
            snippet_max_tokens = self._get_env_int("SNIPPET_MAX_TOKENS", 512)
            if snippet_max_tokens > 0:
                tokenizer = await self.get_snippet_tokenizer()
            else:
                tokenizer = None

            #〔讲解〕分支一:需要截断且 tokenizer 就绪——逐条把正文按 token 截断成 snippet。
            if snippet_max_tokens > 0 and tokenizer:
                for cand in candidates:
                    text = cand["text"]
                    #〔讲解〕先编码成 token id(不加特殊符),只有超过上限才截;否则原文照搬。
                    #〔讲解〕按 token 截而非按字符,是为了和模型的上下文计量口径一致,避免 snippet 实际占用超预算。
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    if len(tokens) > snippet_max_tokens:
                        truncated_tokens = tokens[:snippet_max_tokens]
                        #〔讲解〕把截断后的 token 解码回文本(跳过特殊符)作为 snippet。
                        cand["snippet"] = tokenizer.decode(
                            truncated_tokens, skip_special_tokens=True
                        )
                    else:
                        cand["snippet"] = text
            else:
                #〔讲解〕分支二:不截断——snippet 直接等于完整正文。
                for cand in candidates:
                    cand["snippet"] = cand["text"]

            #〔讲解〕组装返回结果:每条只保留智能体需要的字段(docid/snippet,有分数则附 score),
            #〔讲解〕丢掉原始 text 等冗余,精简回灌给 LLM 的内容。
            results: List[Dict[str, Any]] = []
            for cand in candidates:
                #〔讲解〕score 可能为 None(某些检索器不给分),此时省略该字段,避免输出 "score: null" 干扰模型。
                if cand.get("score") is None:
                    results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
                else:
                    results.append(
                        {
                            "docid": cand["docid"],
                            "score": cand["score"],
                            "snippet": cand["snippet"],
                        }
                    )
        except Exception as e:
            #〔讲解〕检索过程中任何异常都兜底成一句固定错误串返回,保证单次工具失败不会让整条 rollout 轨迹崩掉。
            return "[search] server error"
        #〔讲解〕注意:成功时返回的是 list 对象(非字符串),由上层框架负责序列化后回灌给智能体。
        return results

    #〔讲解〕返回 OpenAI function-calling 规范的工具定义(type=function + name/description/parameters)。
    #〔讲解〕供构造发给 LLM 的 tools 列表时使用,与类属性里的 qwen_agent 风格定义互为不同格式的同一份说明书。
    def get_tool_definitions(self):
        #〔讲解〕拷贝一份 parameters 再改,避免污染类属性;additionalProperties=False 表示不允许 query 之外的多余字段,
        #〔讲解〕收紧 schema 让 LLM 更不容易乱传参数。
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }
