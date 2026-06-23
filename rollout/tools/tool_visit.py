# 〔讲解〕本文件 = rollout 阶段「网页访问工具」visit 的客户端封装。
# 〔讲解〕在 pipeline 里属于 rollout/tools:ReAct 智能体跑搜索轨迹时,除了 search(搜索)还能 visit(打开某个网页深读)。
# 〔讲解〕它本身不爬网页,而是把 (url, goal) 发给一个独立的 serve_search.py HTTP 后端,由后端用 crawl4ai 抓取页面、再用 ROUGE/BM25 选出与 goal 最相关的片段返回。
# 〔讲解〕上游 = ReAct agent 决定要 visit 时调用本工具;下游 = serve_search 后端服务。本类靠 @register_tool 注册进 qwen_agent 工具体系,模型即可按工具协议调用。
"""
Client wrapper for Visit tool that uses serve_search.py backend.
Uses crawl4ai for scraping with ROUGE/BM25 for snippet finding.
"""

import asyncio
from typing import Union, Optional
import requests
import os
from qwen_agent.tools.base import BaseTool, register_tool
from dotenv import load_dotenv

# 〔讲解〕从 .env 文件把配置读进环境变量(如下面的 SEARCH_SERVER_URL),方便不改代码就切换后端地址。
load_dotenv()

# 〔讲解〕后端服务地址:优先读环境变量 SEARCH_SERVER_URL,没设置就默认本机 8765 端口。所有 visit 请求都发到这里。
SEARCH_SERVER_URL = os.environ.get('SEARCH_SERVER_URL', 'http://localhost:8765')


# 〔讲解〕@register_tool('visit') 把这个类注册成名为 visit 的工具;allow_overwrite=True 允许重复注册时覆盖(避免多次 import 报错)。
@register_tool('visit', allow_overwrite=True)
# 〔讲解〕Visit 类:继承 qwen_agent 的 BaseTool,是 visit 工具的具体实现。职责 = 把模型给的 (url, goal) 转发给后端并返回抓取结果。
class Visit(BaseTool):
    # 〔讲解〕name / description / parameters 是工具的「自我描述」,会被喂给大模型,让它知道有这个工具、何时用、要传什么参数。
    name = 'visit'
    description = 'Visit a webpage and return the relevant content based on the goal.'
    # 〔讲解〕parameters 用 JSON Schema 声明入参:url(要访问的网址)和 goal(此次访问想达成的目标),二者都必填(required)。
    # 〔讲解〕goal 很关键——后端正是靠它做相关性筛选,只返回与 goal 匹配的网页片段,而非整页原文。
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to visit."
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for the webpage."
            }
        },
        "required": ["url", "goal"]
    }

    # 〔讲解〕构造函数:接收可选配置 cfg,交给父类初始化后,把后端地址存到实例属性 self.server_url 备用。
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.server_url = SEARCH_SERVER_URL

    # 〔讲解〕call 是工具被调用时的统一入口(异步)。入参 params 可能是 dict 或 JSON 字符串,返回值是给模型看的文本结果。
    async def call(self, params: Union[str, dict], **kwargs) -> str:
        # 〔讲解〕尝试从 params 取出 url 和 goal;若缺字段/格式不对会抛异常,被 except 捕获并返回友好错误,而非让整条轨迹崩掉。
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        # 〔讲解〕核心一步:visit_with_server 是同步阻塞的网络请求,这里用 asyncio.to_thread 丢到线程池跑,避免阻塞事件循环(多条轨迹可并发)。
        response = await asyncio.to_thread(self.visit_with_server, url, goal)
        # 〔讲解〕strip 去掉首尾空白后返回,保证给模型的结果干净。
        return response.strip()

    # 〔讲解〕visit_with_server:真正发 HTTP 请求的函数。把 (url, goal) POST 给后端 /visit 接口,拿回抓取并筛选好的正文。
    def visit_with_server(self, url: str, goal: str) -> str:
        """Send visit request to serve_search.py backend."""
        try:
            # 〔讲解〕向后端 /visit 发 POST,body 是 url+goal 的 JSON;timeout=60 秒——网页抓取较慢,给足时间但仍设上限防卡死。
            response = requests.post(
                f"{self.server_url}/visit",
                json={"url": url, "goal": goal},
                timeout=60
            )
            # 〔讲解〕若后端返回 4xx/5xx,raise_for_status 会抛异常,统一落到下面的 except 兜底。
            response.raise_for_status()
            # 〔讲解〕把响应体解析成 dict;后端约定返回里含 'content' 字段(已抓取+筛选后的网页内容)。
            data = response.json()

            # if data.get('cached'):
            #     print(f"[Visit] Cache hit for: {url}")

            # 〔讲解〕取出并返回 content 字段,这就是最终给模型阅读的网页相关正文。
            return data['content']

        # 〔讲解〕下面三个 except 分门别类兜底:连不上后端 / 请求超时 / 其它任意异常,都返回可读错误字符串而非抛出,保证 ReAct 轨迹能继续走。
        except requests.exceptions.ConnectionError:
            return "[Visit] Error: Cannot connect to server."
        except requests.exceptions.Timeout:
            return "[Visit] Error: Visit request timed out."
        except Exception as e:
            return f"[Visit] Error: {str(e)}"

    # 〔讲解〕get_tool_definitions:按 OpenAI function-calling 的格式导出本工具的定义,供需要该 schema 的调用方使用。
    def get_tool_definitions(self):
        # 〔讲解〕复制一份 parameters 再改,避免污染类属性原对象。
        parameters = self.parameters.copy()
        # 〔讲解〕additionalProperties=False 表示不允许模型传 schema 之外的多余字段,约束更严格。
        parameters["additionalProperties"] = False
        # 〔讲解〕返回标准的 function 工具定义结构:type/function(name+description+parameters)。
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }
