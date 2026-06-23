# 〔讲解〕模块说明:本文件是 rollout 检索层(rollout/searchers/)里的「稠密向量检索器(dense retrieval)」。
# 〔讲解〕在 pipeline 里属于 rollout/ 这一环——ReAct 智能体跑搜索轨迹时,会调用 rollout/tools 里的搜索工具,
# 〔讲解〕而那些工具底层就委托给这里的 Searcher 真正去「查文档」。本文件实现两个检索器:
# 〔讲解〕  FaissSearcher  —— 用一个 embedding 模型把 query 编码成向量,再用 FAISS 在预建的向量索引里做近邻检索;
# 〔讲解〕  ReasonIrSearcher —— 继承上者,只换了 embedding 模型(ReasonIR)和 query 编码方式。
# 〔讲解〕上游=rollout 的工具/Agent;下游=返回检索到的 passage(docid/score/text)给 Agent 继续推理。
"""
FAISS searcher implementation for dense retrieval.
"""

import glob
import logging
import os
import pickle
from itertools import chain
from typing import Any, Dict, List, Optional

# 〔讲解〕faiss = Facebook 的向量近邻检索库(本文件核心:在百万级向量里快速找最相似的 k 个)。
import faiss
import numpy as np
import torch
# 〔讲解〕datasets / tevatron / transformers 都是为「加载语料 + 加载并运行 embedding 模型」服务的第三方库。
# 〔讲解〕tevatron 是一个稠密检索工具包,这里借它的 DenseModel(编码模型)和 FaissFlatSearcher(暴力扁平索引)。
from datasets import load_dataset
from tevatron.retriever.arguments import ModelArguments
from tevatron.retriever.driver.encode import DenseModel
from tevatron.retriever.searcher import FaissFlatSearcher
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# 〔讲解〕BaseSearcher 是所有检索器的抽象基类(定义 search/get_document 等统一接口),本文件的类都得遵守它。
from .base import BaseSearcher

import re

logger = logging.getLogger(__name__)


# 〔讲解〕FaissSearcher:稠密检索器主类。职责=①加载预建 FAISS 向量索引 ②加载 embedding 模型 ③加载语料文本,
# 〔讲解〕然后对外提供 search(query, k)(查 top-k 相关 passage)和 get_document(docid)(按 id 取原文)两个能力。
# 〔讲解〕它继承 BaseSearcher,被 rollout 的搜索工具按统一接口调用。
class FaissSearcher(BaseSearcher):
    # 〔讲解〕parse_args:声明本检索器需要的命令行参数。框架会把 parser 传进来,让每种 searcher 注册自己的专属参数。
    # 〔讲解〕这是「插件式」设计——不同 searcher 各自登记参数,主程序统一解析。
    @classmethod
    def parse_args(cls, parser):
        # 〔讲解〕--index-path:预建向量索引的 glob 通配路径(语料被切成多个分片 .pkl,故用通配符一次匹配全部)。
        parser.add_argument(
            "--index-path",
            required=True,
            help="Glob pattern for pickle files (e.g. /path/to/corpus.*.pkl).",
        )
        # 〔讲解〕--model-name:用来把 query 编码成向量的 embedding 模型名(必须和当初建索引用的模型一致,否则向量不可比)。
        parser.add_argument(
            "--model-name",
            required=True,
            help="Model name for FAISS search (e.g. 'Qwen/Qwen3-Embedding-0.6B').",
        )
        # 〔讲解〕--normalize:是否对向量做 L2 归一化。归一化后内积≈余弦相似度;要和建索引时的设置保持一致。
        parser.add_argument(
            "--normalize",
            action="store_true",
            default=False,
            help="Whether to normalize embeddings for FAISS search (default: False)",
        )
        # 〔讲解〕--pooling:句向量池化方式(eos=取序列末尾 token 的隐状态当整句表示,Qwen 系 embedding 模型常用)。
        parser.add_argument(
            "--pooling",
            default="eos",
            help="Pooling method for FAISS search (default: eos)",
        )
        # 〔讲解〕--torch-dtype:模型权重精度。float16/bfloat16 省显存更快,float32 最精确。
        parser.add_argument(
            "--torch-dtype",
            default="float16",
            choices=["float16", "bfloat16", "float32"],
            help="Torch dtype for FAISS search (default: float16)",
        )
        # 〔讲解〕--dataset-name:存放 docid→原文 的语料数据集名(索引里只有 docid+向量,原文得从这里查)。
        parser.add_argument(
            "--dataset-name",
            default="Tevatron/browsecomp-plus-corpus",
            help="Dataset name for document retrieval in FAISS search (default: Tevatron/browsecomp-plus-corpus)",
        )
        # 〔讲解〕--task-prefix:拼在 query 前的指令前缀。指令型 embedding 模型靠这种 "Instruct:..." 提示来对齐检索任务,
        # 〔讲解〕能显著影响向量质量(注意:必须与建索引/训练时的前缀约定一致)。
        parser.add_argument(
            "--task-prefix",
            default="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
            help="Task prefix for FAISS search queries",
        )
        # 〔讲解〕--max-length:query 编码时的最大 token 长度,超出会被截断(8192 是 Qwen3-Embedding 的上限)。
        parser.add_argument(
            "--max-length",
            type=int,
            default=8192,
            help="Maximum sequence length for FAISS search (default: 8192)",
        )

    # 〔讲解〕__init__:构造时一次性把三件重资源全部就绪——索引、模型、语料。
    # 〔讲解〕入参 args=解析好的命令行参数对象。先做参数校验,再依次加载,加载完即可直接 search。
    def __init__(self, args):
        # 〔讲解〕防御性校验:bm25 是稀疏检索(走另一个 searcher),本类只做稠密向量检索,故拒绝。
        if args.model_name == "bm25":
            raise ValueError("model_name cannot be 'bm25' for FAISS searcher")
        if not args.index_path:
            raise ValueError("index_path is required for FAISS searcher")

        self.args = args
        # 〔讲解〕下面五个先置 None,是「延迟初始化」占位:真正赋值发生在随后的 _load_* 方法里。
        self.retriever = None      # 〔讲解〕FAISS 索引对象(负责向量近邻搜索)
        self.model = None          # 〔讲解〕embedding 模型(把 query 文本编码成向量)
        self.tokenizer = None      # 〔讲解〕分词器(把 query 文本切成 token id 喂给模型)
        self.lookup = None         # 〔讲解〕行号→docid 的映射表(索引返回的是向量行号,要靠它换回真实 docid)
        self.docid_to_text = None  # 〔讲解〕docid→原文 的字典(拿到 docid 后查正文)

        logger.info("Initializing FAISS searcher...")

        # 〔讲解〕三步加载顺序固定:先把向量索引读进内存,再加载编码模型,最后加载语料正文。
        self._load_faiss_index()

        self._load_model()

        self._load_dataset()

        logger.info("FAISS searcher initialized successfully")

    # 〔讲解〕_load_faiss_index:把磁盘上多个 .pkl 分片(每片=一批向量 reps + 对应 docid 列表 lookup)合并成一个可搜索的 FAISS 索引。
    # 〔讲解〕为什么分片:大语料编码成的向量动辄几十 GB,建索引时按片产出,这里再逐片 add 进同一个索引。
    def _load_faiss_index(self) -> None:
        # 〔讲解〕内部小工具:读一个 pkl,反序列化出 (向量数组 reps, docid 列表 lookup)。reps 转成 np.array 供 FAISS 用。
        def pickle_load(path):
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            return np.array(reps), lookup

        # 〔讲解〕用 glob 把通配路径展开成实际文件列表(如 corpus.0.pkl, corpus.1.pkl ...)。
        index_files = glob.glob(self.args.index_path)
        logger.info(
            f"Pattern match found {len(index_files)} files; loading them into index."
        )

        if not index_files:
            raise ValueError(f"No files found matching pattern: {self.args.index_path}")

        # Load first shard
        # 〔讲解〕先单独加载第 0 片,并用它的向量初始化 FaissFlatSearcher(Flat=暴力精确检索,逐一算距离,慢但召回准)。
        # 〔讲解〕之所以要单独拿第 0 片,是因为构造索引需要一批初始向量来确定维度,后续片只能用 add 追加。
        p_reps_0, p_lookup_0 = pickle_load(index_files[0])
        self.retriever = FaissFlatSearcher(p_reps_0)

        # Load remaining shards
        # 〔讲解〕chain(...) 把「已读好的第 0 片」和「其余片的惰性加载(map 延迟读取)」串成一个统一可迭代序列,
        # 〔讲解〕这样下面的循环能用同一套逻辑处理所有片,且第 0 片不会被重复读盘。
        shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
        # 〔讲解〕多于一片时套上 tqdm 进度条(分片多、加载久,给用户可视反馈)。
        if len(index_files) > 1:
            shards = tqdm(
                shards, desc="Loading shards into index", total=len(index_files)
            )

        # 〔讲解〕逐片把向量 add 进索引、把 docid 列表累加进 self.lookup。
        # 〔讲解〕关键:add 的顺序 = lookup 拼接的顺序,二者行号严格对齐,搜索返回的行号才能正确还原成 docid。
        self.lookup = []
        for p_reps, p_lookup in shards:
            self.retriever.add(p_reps)
            self.lookup += p_lookup

        self._setup_gpu()

    # 〔讲解〕_setup_gpu:可选地把 FAISS 索引搬到 GPU 加速搜索。
    # 〔讲解〕注意下一行人为把 num_gpus 写死为 0(原本的 get_num_gpus 探测被注释掉了),所以本仓库实际强制走 CPU 分支。
    def _setup_gpu(self) -> None:
        # num_gpus = faiss.get_num_gpus()
        num_gpus = 0
        if num_gpus == 0:
            logger.info("No GPU found or using faiss-cpu. Using CPU.")
        else:
            logger.info(f"Using {num_gpus} GPU(s)")
            # 〔讲解〕单 GPU:把 CPU 索引克隆到 0 号 GPU,useFloat16 用半精度省显存。(当前 num_gpus=0,此分支不会执行。)
            if num_gpus == 1:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                res = faiss.StandardGpuResources()
                self.retriever.index = faiss.index_cpu_to_gpu(
                    res, 0, self.retriever.index, co
                )
            else:
                # 〔讲解〕多 GPU:shard=True 把索引分片到多张卡上,分摊显存与算力。
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = True
                self.retriever.index = faiss.index_cpu_to_all_gpus(
                    self.retriever.index, co, ngpu=num_gpus
                )

    # 〔讲解〕_load_model:加载把「query 文本→向量」的 embedding 模型与对应分词器。
    # 〔讲解〕这是稠密检索的核心组件:search 时用它编码 query,再去索引里找近邻。
    def _load_model(self) -> None:
        logger.info(f"Loading model: {self.args.model_name}")

        # 〔讲解〕优先用环境变量 HF_HOME 当 HuggingFace 缓存目录(离线/指定盘位时常用),没设就用默认(None)。
        hf_home = os.getenv("HF_HOME")
        if hf_home:
            cache_dir = hf_home
        else:
            cache_dir = None

        # 〔讲解〕把零散参数打包成 tevatron 约定的 ModelArguments 配置对象,供下面的 DenseModel.load 使用。
        model_args = ModelArguments(
            model_name_or_path=self.args.model_name,
            normalize=self.args.normalize,
            pooling=self.args.pooling,
            cache_dir=cache_dir,
        )

        # 〔讲解〕把字符串精度名翻译成真正的 torch dtype 对象。
        if self.args.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif self.args.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # 〔讲解〕DenseModel.load:tevatron 封装的稠密编码模型加载入口,把 pooling/normalize 等检索专用配置一并注入。
        # 〔讲解〕attn_implementation 被硬改成 "sdpa"(PyTorch 原生 scaled-dot-product attention,通用兼容,不依赖 flash-attn)。
        self.model = DenseModel.load(
            model_args.model_name_or_path,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            lora_name_or_path=model_args.lora_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            # attn_implementation=model_args.attn_implementation,
            attn_implementation="sdpa"
        )

        # self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        # 〔讲解〕这里被改成强制 .to("cpu")——本仓库刻意让 embedding 模型跑在 CPU 上(与上面 GPU 探测被关掉相呼应)。
        self.model = self.model.to("cpu")
        # 〔讲解〕eval() 切到推理模式(关 dropout 等),检索只前向编码、不训练。
        self.model.eval()

        # 〔讲解〕加载分词器。padding_side="left" 是为左填充——配合 eos pooling 取末尾 token 时,
        # 〔讲解〕填充放左边能保证真实内容靠右、末位 token 是有效的句末表示。
        self.tokenizer = AutoTokenizer.from_pretrained(
            (
                model_args.tokenizer_name
                if model_args.tokenizer_name
                else model_args.model_name_or_path
            ),
            cache_dir=model_args.cache_dir,
            padding_side="left",
        )

        logger.info("Model loaded successfully")

    # 〔讲解〕_load_dataset:把语料数据集读进内存,建成 docid→text 的字典,供搜索/取文档时还原原文。
    # 〔讲解〕大段 except 主要是把 HuggingFace 的鉴权/下载失败,翻译成对用户友好的排错指引。
    def _load_dataset(self) -> None:
        logger.info(f"Loading dataset: {self.args.dataset_name}")

        try:
            # 〔讲解〕同样支持用环境变量 HF_DATASETS_CACHE 指定数据集缓存目录。
            dataset_cache = os.getenv("HF_DATASETS_CACHE")
            if dataset_cache:
                cache_dir = dataset_cache
            else:
                cache_dir = None

            ds = load_dataset(
                self.args.dataset_name, split="train", cache_dir=cache_dir
            )
            # 〔讲解〕核心:用字典推导把数据集每行压成 {docid: text},之后按 docid O(1) 取正文。
            self.docid_to_text = {row["docid"]: row["text"] for row in ds}
            logger.info(f"Loaded {len(self.docid_to_text)} passages from dataset")
        except Exception as e:
            # 〔讲解〕识别「数据集在 Hub 上不存在或无权访问」这类典型错误,判定多半是没登录/没 token,给出补救步骤。
            if "doesn't exist on the Hub or cannot be accessed" in str(e):
                logger.error(
                    f"Dataset '{self.args.dataset_name}' access failed. This is likely an authentication issue."
                )
                logger.error("Possible solutions:")
                logger.error("1. Ensure you are logged in to Hugging Face:")
                logger.error("   huggingface-cli login")
                logger.error("2. Set environment variable:")
                logger.error("   export HF_TOKEN=your_token_here")
                logger.error(
                    "3. Check if the dataset name is correct and you have access"
                )
                logger.error(f"Current environment variables:")
                # 〔讲解〕只打印 token 是否「已设置」,不打印其值——避免把密钥泄漏到日志里(安全习惯)。
                logger.error(
                    f"   HF_TOKEN: {'Set' if os.getenv('HF_TOKEN') else 'Not set'}"
                )
                logger.error(
                    f"   HUGGINGFACE_HUB_TOKEN: {'Set' if os.getenv('HUGGINGFACE_HUB_TOKEN') else 'Not set'}"
                )

                # 〔讲解〕进一步尝试调 HfApi().whoami() 查当前登录身份,帮用户确认「到底登没登、登的是谁」。
                try:
                    from huggingface_hub import HfApi

                    api = HfApi()
                    user_info = api.whoami()
                    logger.error(
                        f"   Hugging Face user: {user_info.get('name', 'Unknown')}"
                    )
                except Exception as auth_e:
                    logger.error(
                        f"   Hugging Face authentication check failed: {auth_e}"
                    )

            # 〔讲解〕无论是否命中鉴权分支,最终都把原始异常包成 RuntimeError 抛出——数据集加载失败属致命错,不能静默继续。
            raise RuntimeError(
                f"Failed to load dataset '{self.args.dataset_name}': {e}"
            )

    # 〔讲解〕search:对外的核心检索接口。输入一条 query 和要返回的条数 k,返回 top-k 个最相关 passage 的列表。
    # 〔讲解〕流程=①编码 query 成向量 ②在 FAISS 索引里搜近邻 ③把行号还原成 docid 并附上原文与分数。
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        # 〔讲解〕前置校验:四件套(索引/模型/分词器/lookup)缺一不可,否则没法完成检索。
        if not all([self.retriever, self.model, self.tokenizer, self.lookup]):
            raise RuntimeError("Searcher not properly initialized")

        # 〔讲解〕给 query 拼上 task_prefix 指令前缀后分词。truncation+max_length 防止超长 query 撑爆模型。
        batch_dict = self.tokenizer(
            self.args.task_prefix + query,
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt",
        )

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # 〔讲解〕同样被改成强制 CPU——与模型加载在 CPU 上保持一致(本仓库整体走 CPU 推理)。
        device = "cpu"
        # 〔讲解〕把分词得到的每个张量搬到目标设备上(此处即 CPU)。
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        # 〔讲解〕autocast=自动混合精度(推理时按需用半精度,省内存/提速);no_grad=不建反向图(只前向,省内存)。
        with torch.amp.autocast(device):
            with torch.no_grad():
                # 〔讲解〕encode_query 把分词结果编码成 query 向量 q_reps,这就是要拿去索引里比对的「查询向量」。
                q_reps = self.model.encode_query(batch_dict)
                # 〔讲解〕搬回 CPU、断开计算图、转成 numpy——因为 FAISS 接口吃的是 numpy 数组而非 torch 张量。
                q_reps = q_reps.cpu().detach().numpy()

        # 〔讲解〕真正的近邻搜索:返回 top-k 的相似度分数 all_scores 和命中向量的「行号」psg_indices(不是 docid!)。
        all_scores, psg_indices = self.retriever.search(q_reps, k)

        # 〔讲解〕[0] 取第 0 条 query 的结果(这里一次只查一条 query,结果外层包了一层 batch 维度)。
        results = []
        for score, index in zip(all_scores[0], psg_indices[0]):
            # 〔讲解〕关键一步:用 lookup[行号] 把 FAISS 返回的内部行号翻译回真实 docid。
            passage_id = self.lookup[index]
            # 〔讲解〕再用 docid 去字典取正文;取不到给兜底字符串,避免 KeyError 中断。
            passage_text = self.docid_to_text.get(passage_id, "Text not found")

            # 〔讲解〕组装成统一结构 {docid, score, text};score 转 float 防止 numpy 类型在后续序列化(如 JSON)出问题。
            results.append(
                {"docid": passage_id, "score": float(score), "text": passage_text}
            )

        return results

    # 〔讲解〕keep_first_n_words:只保留前 n 个「单词」(按非空白片段计),超长就截断并打上 [Document is truncated.] 标记。
    # 〔讲解〕目的=把长文档裁短再喂给下游 Agent,控制 prompt 长度/token 成本。staticmethod 因为它不依赖实例状态。
    @staticmethod
    def keep_first_n_words(text: str, n: int = 1000) -> str:
        if not text:
            return ""
        count = 0
        # 〔讲解〕用正则 \S+(连续非空白)逐个匹配「词」并计数;数到第 n 个就在它结尾处切断。
        # 〔讲解〕m.end() 是第 n 个词在原文里的结束下标,text[:m.end()] 即「正好到第 n 个词为止」的子串,保留原始空白/格式。
        for m in re.finditer(r'\S+', text):
            count += 1
            if count == n:
                return text[:m.end()] + '\n[Document is truncated.]'
        # 〔讲解〕循环跑完都没到 n 个词,说明文档本就短于上限,原样返回。
        return text

    # 〔讲解〕get_document:按 docid 直接取一篇文档原文(不走向量检索)。Agent 命中某个 docid 想看全文时用。
    # 〔讲解〕返回 {docid, text},其中 text 已用 keep_first_n_words 截到前 4096 词;查无此 id 返回 None。
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        if not self.docid_to_text:
            raise RuntimeError("Dataset not loaded")

        text = self.docid_to_text.get(docid)
        if text is None:
            return None

        return {
            "docid": docid,
            "text": self.keep_first_n_words(text, 4096),
        }

    # 〔讲解〕search_type:只读属性,返回检索器类型标识 "FAISS",供上层日志/路由区分用的是哪种检索器。
    @property
    def search_type(self) -> str:
        return "FAISS"


# 〔讲解〕ReasonIrSearcher:FaissSearcher 的子类,继承全部索引/语料加载逻辑,只覆写 _load_model 和 search。
# 〔讲解〕差异=改用 ReasonIR 这一推理型 embedding 模型,且 query 编码方式不同(直接 model.encode + 专属 instruction)。
# 〔讲解〕这是典型的「换内核、复用外壳」:索引怎么建、文档怎么取都照旧,只替换「query→向量」这一步。
class ReasonIrSearcher(FaissSearcher):
    # 〔讲解〕覆写 _load_model:不走 tevatron 的 DenseModel,改用 transformers 通用 AutoModel 直接加载 ReasonIR。
    def _load_model(self) -> None:
        logger.info(f"Loading model: {self.args.model_name}")

        hf_home = os.getenv("HF_HOME")
        if hf_home:
            cache_dir = hf_home
        else:
            cache_dir = None

        model_args = ModelArguments(
            model_name_or_path=self.args.model_name,
            normalize=self.args.normalize,
            pooling=self.args.pooling,
            cache_dir=cache_dir,
        )

        if self.args.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif self.args.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # 〔讲解〕trust_remote_code=True:允许执行模型仓库自带的自定义建模代码(ReasonIR 有自己的 encode 实现,故必须开)。
        self.model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        # 〔讲解〕注意:与父类强制 CPU 不同,这里恢复了「有 GPU 就上 GPU」——ReasonIR 偏大,用 GPU 编码更现实。
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        logger.info("Model loaded successfully")

    # 〔讲解〕覆写 search:整体流程与父类相同,区别在于不分词、直接把 query 文本丢给 model.encode,并带 ReasonIR 专属 instruction。
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        # 〔讲解〕这里校验三件套即可(无 tokenizer)——因为本类的编码不经过单独的 tokenizer 步骤。
        if not all([self.retriever, self.model, self.lookup]):
            raise RuntimeError("Searcher not properly initialized")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.amp.autocast(device):
            with torch.no_grad():
                # 〔讲解〕ReasonIR 的 encode 直接吃「query 列表 + instruction 模板」,内部自行分词与池化,返回 query 向量。
                # 〔讲解〕instruction 里的 <|user|>/<|embed|> 是 ReasonIR 约定的特殊标记,用来界定指令段与待编码段。
                q_reps = self.model.encode(
                    [query],
                    instruction="<|user|>\nGiven a question, retrieve relevant passages that help answer the question\n<|embed|>\n",
                )

        # 〔讲解〕以下与父类 search 后半段完全一致:搜索→行号还原 docid→取正文→组装结果。
        all_scores, psg_indices = self.retriever.search(q_reps, k)

        results = []
        for score, index in zip(all_scores[0], psg_indices[0]):
            passage_id = self.lookup[index]
            passage_text = self.docid_to_text.get(passage_id, "Text not found")

            results.append(
                {"docid": passage_id, "score": float(score), "text": passage_text}
            )

        return results
