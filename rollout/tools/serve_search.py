# 〔讲解〕本文件 = rollout 阶段的「检索/网页工具」HTTP 微服务(FastAPI)。
# 〔讲解〕它把两个能力包成 REST 接口供 ReAct agent 调用:
#   /search → 走 Serper(Google 搜索 API)拿搜索结果列表;
#   /visit  → 用 crawl4ai 抓取某个网页/PDF 正文,再用 ROUGE/BM25 抽出与 goal 最相关的片段。
# 〔讲解〕在 pipeline 里属于 rollout/tools:上游是跑 ReAct 轨迹的 agent(它发 search/visit 工具调用),
# 〔讲解〕本服务返回的文本会被拼进 agent 的观察(observation),驱动下一步推理。带 LRU 缓存避免重复抓取。
import json
import os
import time
import asyncio
import logging
from functools import lru_cache
from typing import List, Dict, Optional, Tuple
import http.client
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
# 〔讲解〕rouge_score:文本重叠度打分库。这里用 ROUGE-L 衡量「网页片段 vs 检索目标」的匹配度,从而挑最相关段落。
from rouge_score import rouge_scorer

# 〔讲解〕crawl4ai:把网页渲染并清洗成 Markdown 的爬虫库。下面三件套分别负责:
# 〔讲解〕DefaultMarkdownGenerator(HTML→Markdown)、PruningContentFilter(剪掉广告/导航等噪声块)、爬虫运行/浏览器配置。
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai import CrawlerRunConfig, AsyncWebCrawler, BrowserConfig

# python serve_search.py --host 0.0.0.0 --port 8765 --workers 3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 〔讲解〕从 .env 文件加载环境变量(如 SERPER_KEY_ID),这样密钥不写死在代码里。
load_dotenv()

# Configuration
# 〔讲解〕Serper(Google 搜索代理服务)的 API key,从环境变量读;缺失时为 None,真正搜索时才会暴露问题。
SERPER_KEY = os.environ.get('SERPER_KEY_ID')
# 〔讲解〕下面四个是各级 LRU 缓存容量(可被环境变量覆盖),目的是同一 query/url 不重复打外网,省钱省时。
SEARCH_CACHE_SIZE = int(os.environ.get('SEARCH_CACHE_SIZE', 8192))
VISIT_CACHE_SIZE = int(os.environ.get('VISIT_CACHE_SIZE', 8192))
# 〔讲解〕访问缓存细分两层:抓取内容缓存 + 片段抽取缓存(默认都等于 VISIT_CACHE_SIZE)。分层是为了「同一网页、不同 goal」能复用已抓内容。
VISIT_CONTENT_CACHE_SIZE = int(os.environ.get('VISIT_CONTENT_CACHE_SIZE', VISIT_CACHE_SIZE))
VISIT_SNIPPET_CACHE_SIZE = int(os.environ.get('VISIT_SNIPPET_CACHE_SIZE', VISIT_CACHE_SIZE))

# HTTP headers for web requests
# 〔讲解〕伪装成浏览器的 User-Agent。很多网站会拒绝默认的 python-requests UA,带上这个能降低被拦概率。
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

# Initialize ROUGE scorer for snippet finding
# 〔讲解〕全局只建一次 ROUGE-L 打分器(use_stemmer=英文做词干还原),避免每次请求重复初始化的开销。
rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Initialize FastAPI app
# 〔讲解〕创建 FastAPI 应用实例;下方用 @app.get/@app.post 注册的就是对外 HTTP 接口。
app = FastAPI(title="Search & Visit Service", version="1.0.0")

# Request/Response Models
# 〔讲解〕以下用 pydantic 定义请求/响应的数据结构。FastAPI 会据此自动做 JSON 校验与序列化。
# 〔讲解〕SearchRequest:/search 的入参,只有一个查询词。
class SearchRequest(BaseModel):
    query: str

# 〔讲解〕VisitRequest:/visit 的入参。url=要访问的网页;goal=访问目的(用来从正文里挑相关片段)。
class VisitRequest(BaseModel):
    url: str
    goal: str

# 〔讲解〕SearchResponse:/search 的返回。results=拼好的搜索结果文本;cached=本次是否命中缓存(便于观测/调试)。
class SearchResponse(BaseModel):
    query: str
    results: str
    cached: bool = False

# 〔讲解〕VisitResponse:/visit 的返回。content=抽取后的网页正文片段;cached=是否命中缓存。
class VisitResponse(BaseModel):
    url: str
    content: str
    cached: bool = False


# 〔讲解〕判断文本里是否含中文字符。用途:据此切换搜索的地区/语言(中文→中国/zh-cn,否则→美国/en)。
def contains_chinese_basic(text: str) -> bool:
    """Check if text contains Chinese characters."""
    # 〔讲解〕逐字符检查码点是否落在 CJK 统一汉字基本区间 [一, 鿿],命中任一即判为含中文。
    return any('一' <= char <= '鿿' for char in text)


# Crawl4AI-based scraping functions
# 〔讲解〕探测 URL 指向的是 PDF 还是 HTML,决定后续用哪种抓取方式。返回字符串 'pdf' 或 'html'。
def detect_content_type(url: str) -> str:
    """Detect if URL points to PDF or HTML content."""
    parsed_url = urlparse(url)
    # 〔讲解〕第一招(最廉价):URL 路径以 .pdf 结尾,直接判 PDF,不必发请求。
    if parsed_url.path.lower().endswith('.pdf'):
        return 'pdf'

    try:
        # Use GET with stream to check first bytes (HEAD is unreliable for some servers)
        # 〔讲解〕第二招:用 stream=True 的 GET 只取响应头和头几个字节,避免下载整个文件。
        # 〔讲解〕注释说明为何不用 HEAD —— 部分服务器对 HEAD 返回不可靠。timeout=(连接3s, 读取10s)。
        response = requests.get(url, headers=HEADERS, timeout=(3, 10), stream=True)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')

        # Check Content-Type header
        # 〔讲解〕先看响应头 Content-Type 是否含 pdf;命中就关闭连接(stream 模式必须显式 close 释放)。
        if "pdf" in content_type:
            response.close()
            return "pdf"

        # Check magic bytes for PDF (%PDF)
        # 〔讲解〕第三招(兜底):读前 5 个字节看魔数。PDF 文件开头固定是 b'%PDF',比响应头更可靠。
        first_bytes = response.raw.read(5)
        response.close()
        if first_bytes.startswith(b'%PDF'):
            return "pdf"

        return "html"
    # 〔讲解〕任何异常(超时/连接失败等)都保守地当作 HTML 处理,让后续爬虫去尝试,不在探测阶段就报错。
    except Exception:
        return "html"


# 〔讲解〕异步抓取 PDF:下载二进制后用 PyMuPDF(fitz)逐页提取纯文本。
# 〔讲解〕返回三元组 (是否成功, 内容或错误信息, 原始内容)。后两个对 PDF 而言相同。
async def scrape_pdf(url: str) -> Tuple[bool, str, str]:
    """Scrape PDF content using PyMuPDF."""
    # 〔讲解〕局部 import fitz(PyMuPDF):只有真要解析 PDF 时才加载这个重依赖,加快服务启动。
    import fitz
    try:
        response = requests.get(url, headers=HEADERS, timeout=(3, 30))
        response.raise_for_status()
        # 〔讲解〕从内存字节流打开 PDF(不落盘);with 确保用完释放文档句柄。
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text = ""
            # 〔讲解〕遍历每一页累加文本,得到整篇 PDF 的纯文字。
            for page in doc:
                text += page.get_text()
        return True, text, text
    # 〔讲解〕失败时返回 (False, 错误字符串, "");上层据第一个布尔判断是否成功。
    except Exception as e:
        return False, str(e), ""


# 〔讲解〕异步抓取 HTML:用 crawl4ai 渲染页面并清洗成 Markdown。
# 〔讲解〕返回 (是否成功, fit_markdown=精炼后正文, raw_markdown=未剪枝原始 Markdown)。
async def scrape_html(url: str) -> Tuple[bool, str, str]:
    """Scrape HTML content using crawl4ai."""
    # 〔讲解〕内容剪枝过滤器:按动态阈值剔除低信息量区块(导航/页脚/广告),min_word_threshold=3 过滤过短块。
    prune_filter = PruningContentFilter(threshold=0.4, threshold_type="dynamic", min_word_threshold=3)
    # 〔讲解〕Markdown 生成器,挂上剪枝过滤器;ignore_links=False 表示保留链接(对后续可能的二跳访问有用)。
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter, options={"ignore_links": False})
    # 〔讲解〕无头浏览器配置。extra_args 是在服务器/容器里跑 Chromium 的常见稳定性参数(关 GPU、关共享内存限制、关沙箱)。
    browser_config = BrowserConfig(
        headless=True, verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox", "--disable-extensions"]
    )
    # 〔讲解〕单次爬取运行配置:挂 Markdown 生成器,单页加载超时 15s。
    crawler_config = CrawlerRunConfig(markdown_generator=md_generator, page_timeout=15000, verbose=False)

    try:
        # 〔讲解〕async with 进入爬虫上下文(自动开/关浏览器);wait_for 再套一层 30s 总超时,防止页面卡死拖垮服务。
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await asyncio.wait_for(crawler.arun(url=url, config=crawler_config), timeout=30)

        # 〔讲解〕crawl4ai 内部判定抓取失败(如 4xx/5xx),带上它给的错误信息返回。
        if not result.success:
            return False, f"Failed to scrape the page: {result.error_message}", ""

        # 〔讲解〕抓取成功但正文为空(去空白后长度 0)也算失败,避免把空内容喂给 agent。
        if len(result.markdown.raw_markdown.strip()) == 0:
            return False, "Failed to scrape the page: returned empty content.", ""

        # 〔讲解〕fit_markdown=经剪枝过滤后的精炼正文(给 agent 用);raw_markdown=完整原文(保底/调试)。
        fit_markdown = result.markdown.fit_markdown
        raw_markdown = result.markdown.raw_markdown

        return True, fit_markdown, raw_markdown
    # 〔讲解〕超时与其他异常分别返回对应错误信息,统一成 (False, msg, "") 的失败形态。
    except asyncio.TimeoutError:
        return False, "Timeout while scraping the page.", ""
    except Exception as e:
        return False, str(e), ""


# 〔讲解〕核心算法:把长文本切成若干块,找出与 query 最相关的那块,并把它周围一段窗口拼起来返回。
# 〔讲解〕作用 = 在抓到的网页正文里"定位最相关段落",避免把整篇塞给 agent(省 token、更聚焦)。
# 〔讲解〕入参:texts=已切好的文本块列表;query=检索目标;num_characters=返回总字符预算;scoring_func=打分法(rouge/bm25)。
def find_snippet(texts: List[str], query: str, num_characters: int = 10000, scoring_func: str = "rouge") -> str:
    """
    Find the most relevant snippet from texts based on the query.
    Uses ROUGE or BM25 scoring to find the best matching chunk.
    """
    # 〔讲解〕positions 记录每块在"拼接全文"中的 [起, 止] 字符偏移;best_idx 指向最相关块;start 是滚动游标。
    positions = []
    start = 0
    best_recall = 0
    best_idx = 0

    # 〔讲解〕BM25 分支:经典稀疏检索打分。一次性把所有块建索引,再用 query 检索 top-1,直接得到最相关块下标。
    if scoring_func == 'bm25':
        import bm25s
        import Stemmer
        # 〔讲解〕英文词干还原器;先把语料分词建库,再把 query 分词去检索。
        stemmer = Stemmer.Stemmer('english')
        corpus_tokens = bm25s.tokenize(texts, stopwords='en', stemmer=stemmer)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        query_tokens = bm25s.tokenize(query, stopwords='en', stemmer=stemmer)
        # 〔讲解〕k=1 取最相关一块;results[0,0] 是它的下标,转 int 存入 best_idx。
        results, scores = retriever.retrieve(query_tokens, k=1)
        best_idx = int(results[0, 0])

    # 〔讲解〕遍历每块:一边(rouge 模式)算与 query 的 ROUGE-L recall 选最高者,一边记录每块的字符区间。
    for i, text in enumerate(texts):
        if scoring_func == "rouge":
            # 〔讲解〕用 query 作 target、当前块作 prediction;取 recall(query 内容被该块覆盖的比例)作为相关度。
            score = rouge_scorer_instance.score(target=query, prediction=text)
            recall = score['rougeL'].recall
            if recall > best_recall:
                best_recall = recall
                best_idx = i
        # 〔讲解〕记录该块区间 [start, start+len);+1 是为还原后面 "\n".join 拼接时多出的那个换行符,保证偏移对齐。
        positions.append((start, start + len(text)))
        start += len(text) + 1

    # 〔讲解〕字符预算先扣掉"最相关块本身"的长度,剩下的预算用于向其前后扩展上下文窗口。
    best_len = len(texts[best_idx])
    num_characters = num_characters - best_len
    final_text = []
    # 〔讲解〕以 best 块为中心、左右各 num_characters/2 字符为窗口:区间落在窗口内的块都纳入;best 块本身无条件纳入。
    for i, pos in enumerate(positions):
        if (pos[0] >= positions[best_idx][0] - num_characters / 2 and
            pos[1] <= positions[best_idx][1] + num_characters / 2) or i == best_idx:
            final_text.append(texts[i])

    # 〔讲解〕把选中的相邻块按原顺序用换行拼回,作为"最相关片段"返回。
    return "\n".join(final_text)


# 〔讲解〕同步包装器:把"探测类型→异步抓取"统一成一个阻塞调用,方便上层在线程池里跑。
# 〔讲解〕注意它用 asyncio.run 自起事件循环跑协程(因此本函数本身是同步的)。
def get_content_sync(url: str) -> Tuple[bool, str, str]:
    """Synchronous wrapper to get raw content from URL."""
    try:
        content_type = detect_content_type(url)
        # 〔讲解〕据探测结果分派:PDF 走 scrape_pdf,否则走 scrape_html;asyncio.run 把异步抓取同步化。
        if content_type == "pdf":
            result = asyncio.run(scrape_pdf(url))
        else:
            result = asyncio.run(scrape_html(url))
        return result
    except Exception as e:
        return False, str(e), ""


# 〔讲解〕把整篇 content 先切块(chunking),再调 find_snippet 选最相关片段。
# 〔讲解〕chunking_func 控制切法:newline=按行切;words_N=按每 N 个词一块切。返回片段;无内容时返回 None。
def find_snippet_in_content(content: str, query: str, num_characters: int = 10000,
                            scoring_func: str = "rouge", chunking_func: str = "newline") -> Optional[str]:
    """Find relevant snippet in content based on query."""
    # 〔讲解〕没有 goal/query 时无从排序,直接截前 num_characters 字符返回。
    if not query:
        return content[:num_characters]

    # 〔讲解〕按行切块:拆成行后丢掉纯空白行。
    if chunking_func == "newline":
        content_lines = content.split("\n")
        content_lines = [line for line in content_lines if line.strip()]
    # 〔讲解〕按词数切块:从 "words_50" 这种名字里解析出 N=50,把全文按空格分词,再每 N 词打包成一块。
    elif "words" in chunking_func:
        num_words = int(chunking_func.split("_")[1])
        content_lines = content.split(" ")
        content_lines = [line for line in content_lines if line.strip()]
        # 〔讲解〕滑窗步长=num_words 做不重叠切片,再把每片词拼回字符串,得到"每块约 N 词"的块列表。
        content_lines = [content_lines[i:i + num_words] for i in range(0, len(content_lines), num_words)]
        content_lines = [" ".join(line) for line in content_lines]
    # 〔讲解〕未知切法兜底:按行切。
    else:
        content_lines = content.split("\n")
        content_lines = [line for line in content_lines if line.strip()]

    # 〔讲解〕切完一块都没有(空内容)→返回 None,上层据此回报"抽取失败"。
    if len(content_lines) == 0:
        return None

    return find_snippet(content_lines, query, num_characters, scoring_func)


# 〔讲解〕"访问 URL"的未缓存全流程:抓内容 → 抽片段 → 拼成给 agent 的结果文本。
# 〔讲解〕成功返回带 <Content> 包裹的文本;任一步失败返回对应失败说明。这是 use_cache=False 时走的路径。
def _visit_url_with_crawl4ai_uncached(
    url: str,
    goal: str,
    content_length: int = 10000,
    scoring_func: str = "rouge",
    chunking_func: str = "newline",
) -> str:
    # 〔讲解〕第一步抓取;失败则把错误信息直接回给 agent(content_or_error 此时是错误串)。
    success, content_or_error, raw_content = get_content_sync(url)
    if not success:
        return f"Failed to visit the url {url}.\nError: {content_or_error}"

    # 〔讲解〕第二步:在正文里按 goal 抽最相关片段。
    final_content = find_snippet_in_content(content_or_error, goal, content_length, scoring_func, chunking_func)
    if final_content is None:
        return f"Failed to extract content from {url}"

    # 〔讲解〕用 <Content>…</Content> 包裹片段,给 agent 一个清晰的可解析边界。
    return f"Successfully visited the url {url}.\n<Content>\n{final_content}\n</Content>"


# 〔讲解〕@lru_cache 给"按 URL 抓内容"加缓存:同一 url 第二次直接返回上次结果,不再打外网。
# 〔讲解〕特意只用 url 做缓存键(不含 goal),这样同一网页配不同 goal 时仍能复用抓取结果。
@lru_cache(maxsize=VISIT_CONTENT_CACHE_SIZE)
def _get_content_cached(url: str) -> Tuple[bool, str, str]:
    """Cache the expensive URL fetching/scraping by URL only."""
    return get_content_sync(url)


# 〔讲解〕给"片段抽取"单独加一层缓存,缓存键含 (content, goal, …)。
# 〔讲解〕与上面的抓取缓存分层:抓取贵、抽取也不便宜;分开后"同网页不同 goal"只重算抽取、复用抓取。
@lru_cache(maxsize=VISIT_SNIPPET_CACHE_SIZE)
def _find_snippet_cached(
    content: str,
    goal: str,
    content_length: int = 10000,
    scoring_func: str = "rouge",
    chunking_func: str = "newline",
) -> Optional[str]:
    """Cache snippet extraction separately so different goals can reuse fetched content."""
    return find_snippet_in_content(content, goal, content_length, scoring_func, chunking_func)


# 〔讲解〕对外的"访问 URL"主函数。use_cache 决定走缓存版还是直算版。
# 〔讲解〕缓存路径下:先用 _get_content_cached 抓内容(按 url 缓存),再用 _find_snippet_cached 抽片段(按 content+goal 缓存)。
def visit_url_with_crawl4ai(url: str, goal: str, content_length: int = 10000,
                            scoring_func: str = "rouge", chunking_func: str = "newline",
                            use_cache: bool = True) -> str:
    """Main function to visit a URL and extract relevant content based on goal using crawl4ai."""
    # 〔讲解〕不走缓存就直接调上面的 uncached 全流程。
    if not use_cache:
        return _visit_url_with_crawl4ai_uncached(url, goal, content_length, scoring_func, chunking_func)

    success, content_or_error, raw_content = _get_content_cached(url)
    if not success:
        return f"Failed to visit the url {url}.\nError: {content_or_error}"

    final_content = _find_snippet_cached(content_or_error, goal, content_length, scoring_func, chunking_func)
    if final_content is None:
        return f"Failed to extract content from {url}"

    return f"Successfully visited the url {url}.\n<Content>\n{final_content}\n</Content>"


# Core Search Function
# 〔讲解〕未缓存的核心搜索:调 Serper 的 Google 搜索接口,把返回 JSON 解析成给 agent 看的格式化文本。
def _google_search_with_serp_uncached(query: str) -> str:
    """
    Perform Google search using Serper API.
    """
    # 〔讲解〕开一条到 Serper 的 HTTPS 连接(超时 30s)。用底层 http.client 而非 requests,是逐字节可控、轻量。
    conn = http.client.HTTPSConnection("google.serper.dev", timeout=30)

    # 〔讲解〕据 query 是否含中文切换检索地区/语言:中文→中国结果(gl=cn, hl=zh-cn),否则→美国/英文。
    # 〔讲解〕这直接影响搜索召回的语种与本地化排序,对中英混合 benchmark 很关键。
    if contains_chinese_basic(query):
        payload = json.dumps({
            "q": query,
            "location": "China",
            "gl": "cn",
            "hl": "zh-cn"
        })
    else:
        payload = json.dumps({
            "q": query,
            "location": "United States",
            "gl": "us",
            "hl": "en"
        })

    # 〔讲解〕请求头携带 Serper API key 做鉴权,内容为 JSON。
    headers = {
        'X-API-KEY': SERPER_KEY,
        'Content-Type': 'application/json'
    }

    # Retry logic
    # 〔讲解〕重试逻辑:最多发 5 次 POST /search。任何异常就 warning 并退避 0.5s 重试;第 5 次仍失败则返回超时提示。
    # 〔讲解〕注意:break 在成功发请求后跳出,res 在循环外被读取(下方)。
    for i in range(5):
        try:
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            break
        except Exception as e:
            logger.warning(f"Search attempt {i+1} failed: {e}")
            if i == 4:
                return "Google search timeout. Please try again later."
            time.sleep(0.5)
            continue

    # 〔讲解〕读取并解析响应体为 dict。Serper 把自然搜索结果放在 "organic" 字段里。
    data = res.read()
    results = json.loads(data.decode("utf-8"))

    try:
        # 〔讲解〕没有 "organic" 字段说明查无结果,抛异常进下方 except 给出友好提示。
        if "organic" not in results:
            raise Exception(f"No results found for query: '{query}'")

        web_snippets = []
        idx = 0

        # 〔讲解〕逐条遍历自然结果,把每条整理成带序号的 Markdown 条目(标题+链接+可选日期/来源/摘要)。
        for page in results["organic"]:
            idx += 1
            # 〔讲解〕以下字段都是"有则带、无则空串",做防御式取值,避免缺字段 KeyError。
            date_published = ""
            if "date" in page:
                date_published = "\nDate published: " + page["date"]

            source = ""
            if "source" in page:
                source = "\nSource: " + page["source"]

            snippet = ""
            if "snippet" in page:
                snippet = "\n" + page["snippet"]

            # 〔讲解〕组装单条结果:形如 "1. [标题](链接)\n日期\n来源\n摘要"。title/link 是必有字段所以直接取。
            redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
            # 〔讲解〕清掉视频结果里常见的占位噪声句,避免污染喂给 agent 的文本。
            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)

        # 〔讲解〕把所有条目拼成一段带标题的结果文本(结果间空一行分隔),这是 /search 最终返回给 agent 的内容。
        content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        return content

    # 〔讲解〕解析阶段任何异常(含上面手动抛的"查无结果")都降级成一句可读提示,引导 agent 换更宽泛的 query。
    except Exception as e:
        logger.error(f"Search parsing error: {e}")
        return f"No results found for '{query}'. Try with a more general query."


# 〔讲解〕给搜索加 LRU 缓存:同一 query 直接复用上次结果。Serper 按次计费,缓存能显著省调用量。
@lru_cache(maxsize=SEARCH_CACHE_SIZE)
def _google_search_with_serp_cached(query: str) -> str:
    return _google_search_with_serp_uncached(query)


# 〔讲解〕对外搜索主函数:use_cache 切换走缓存版/直算版。供下方 /search 接口调用。
def google_search_with_serp(query: str, use_cache: bool = True) -> str:
    """
    Perform Google search using Serper API with optional in-memory LRU caching.
    """
    if use_cache:
        return _google_search_with_serp_cached(query)
    return _google_search_with_serp_uncached(query)

# API Endpoints
# 〔讲解〕根路径健康检查:GET / 返回服务状态与可用端点列表,用于探活/部署校验。
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Search & Visit Service",
        "endpoints": ["/search", "/visit", "/clear_cache"]
    }


# 〔讲解〕搜索接口:POST /search,接收 query,返回格式化搜索结果(SearchResponse)。这是 agent "搜索工具"的实际落点。
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform a Google search and return formatted results.
    """
    try:
        # 〔讲解〕通过比较调用前后的缓存命中数,判断本次是否命中缓存(下方做差)。这是一种用 lru_cache.cache_info() 间接探测命中的技巧。
        search_hits_before = _google_search_with_serp_cached.cache_info().hits
        # Run blocking function in thread pool
        # 〔讲解〕google_search_with_serp 是阻塞型(同步网络 IO);用 asyncio.to_thread 丢到线程池跑,避免阻塞事件循环、保住并发。
        results = await asyncio.to_thread(google_search_with_serp, request.query)
        # 〔讲解〕命中数变大 → 说明这次是从缓存拿的。
        cached = _google_search_with_serp_cached.cache_info().hits > search_hits_before
        logger.info(f"Search request: {request.query} (cached={cached})")
        return SearchResponse(
            query=request.query,
            results=results,
            cached=cached
        )
    # 〔讲解〕异常统一转成 HTTP 500,把错误信息回给调用方(agent 侧能据此判断工具失败)。
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 〔讲解〕访问接口:POST /visit,接收 url+goal,返回抽取后的网页相关片段(VisitResponse)。这是 agent "浏览网页工具"的落点。
@app.post("/visit", response_model=VisitResponse)
async def visit(request: VisitRequest):
    """
    Visit a URL and extract relevant content based on goal using crawl4ai.
    """
    try:
        # 〔讲解〕分别记录两层缓存(抓取/抽取)的命中基线,后面任一层命中数上升都算"本次命中缓存"。
        content_hits_before = _get_content_cached.cache_info().hits
        snippet_hits_before = _find_snippet_cached.cache_info().hits
        # Run blocking function in thread pool
        # 〔讲解〕同样把阻塞的 visit 全流程丢线程池;这里把默认参数(10000/rouge/newline)显式写出传入。
        content = await asyncio.to_thread(
            visit_url_with_crawl4ai,
            request.url,
            request.goal,
            10000,  # content_length
            "rouge",  # scoring_func
            "newline"  # chunking_func
        )
        # 〔讲解〕两层缓存任一命中即视为 cached=True。
        cached = (
            _get_content_cached.cache_info().hits > content_hits_before
            or _find_snippet_cached.cache_info().hits > snippet_hits_before
        )
        logger.info(f"Visit request: {request.url} (cached={cached})")
        return VisitResponse(
            url=request.url,
            content=content,
            cached=cached
        )
    except Exception as e:
        logger.error(f"Visit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 〔讲解〕清缓存接口:POST /clear_cache,可指定 cache_type='search'/'visit' 清单类,或不传清全部。便于实验间重置缓存保证可复现。
@app.post("/clear_cache")
async def clear_cache(cache_type: Optional[str] = None):
    """
    Clear in-memory LRU caches. Specify cache_type ('search' or 'visit') or clear all.
    """
    try:
        cleared = []
        # 〔讲解〕None(清全部)或显式 'search' → 清搜索缓存。
        if cache_type in (None, "search"):
            _google_search_with_serp_cached.cache_clear()
            cleared.append("search")
        # 〔讲解〕None 或 'visit' → 同时清"抓取"和"片段"两层访问缓存。
        if cache_type in (None, "visit"):
            _get_content_cached.cache_clear()
            _find_snippet_cached.cache_clear()
            cleared.append("visit")
        # 〔讲解〕传了非法 cache_type → 报 400,提示合法取值。
        if cache_type not in (None, "search", "visit"):
            raise HTTPException(status_code=400, detail="cache_type must be one of: search, visit")

        return {
            "status": "success",
            "cleared": cleared,
            "type": cache_type or "all",
        }
    # 〔讲解〕先放行已构造好的 HTTPException(如上面的 400),其余异常再统一转 500,避免把 400 误吞成 500。
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 〔讲解〕脚本入口:解析命令行参数并用 uvicorn 起服务。直接 `python serve_search.py` 时执行。
if __name__ == "__main__":
    import argparse

    # 〔讲解〕三个参数:绑定地址、端口、worker 进程数。
    parser = argparse.ArgumentParser(description="Search & Visit Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8006, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    logger.info(f"Starting Search & Visit Service on {args.host}:{args.port}")
    logger.info(f"Search cache size: {SEARCH_CACHE_SIZE}")
    logger.info(f"Visit content cache size: {VISIT_CONTENT_CACHE_SIZE}")
    logger.info(f"Visit snippet cache size: {VISIT_SNIPPET_CACHE_SIZE}")
    logger.info(f"Workers: {args.workers}")

    # 〔讲解〕多 worker 时 uvicorn 需要"导入字符串"形式才能在子进程里重建 app(故须从 tools 目录启动,让 serve_search 可被 import)。
    # 〔讲解〕注意:多 worker 各自持有独立的进程内 LRU 缓存,缓存不跨进程共享。
    if args.workers > 1:
        # Multiple workers require import string - must run from tools directory
        uvicorn.run(
            "serve_search:app",
            host=args.host,
            port=args.port,
            workers=args.workers
        )
    else:
        # Single worker can use app object directly
        # 〔讲解〕单 worker 直接传 app 对象即可,无需导入字符串。
        uvicorn.run(
            app,
            host=args.host,
            port=args.port
        )
