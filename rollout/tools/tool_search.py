#〔讲解〕本文件 = rollout 环节里 ReAct 智能体调用的「网页搜索」工具(客户端侧)。
#〔讲解〕它本身不直接调 Google/Serper,而是把 query 通过 HTTP POST 发给一个独立的后端服务 serve_search.py,
#〔讲解〕由后端去真正检索并做缓存(同一 query 命中缓存可省钱省时)。pipeline 位置:rollout/ 跑搜索轨迹时,
#〔讲解〕智能体每次想「搜一下」就触发这里的 search 工具;返回的检索结果文本再喂回模型继续推理。
#〔讲解〕上游=ReAct agent(qwen_agent 框架按工具名 "search" 调度过来);下游=serve_search.py 缓存后端。
"""
Client wrapper for Search tool that uses serve_search.py backend.
Maintains the same API as tool_search.py but with caching benefits.
"""

import json
import asyncio
from typing import List, Union, Optional
import requests
#〔讲解〕qwen_agent 的工具基类与注册装饰器:继承 BaseTool 即成为一个「可被智能体调用的工具」,
#〔讲解〕register_tool 把工具按名字登记进框架的工具注册表,模型输出 "调用 search" 时框架就能找到本类。
from qwen_agent.tools.base import BaseTool, register_tool
import os
from dotenv import load_dotenv

#〔讲解〕从 .env 文件把配置读进环境变量,这样下面 os.environ.get 才能拿到 SEARCH_SERVER_URL。
load_dotenv()

#〔讲解〕搜索后端地址:优先读环境变量,没配则默认本机 8765 端口(serve_search.py 监听处)。
SEARCH_SERVER_URL = os.environ.get('SEARCH_SERVER_URL', 'http://localhost:8765')


#〔讲解〕装饰器把本类注册为名为 "search" 的工具;allow_overwrite=True 允许重复 import 时覆盖旧注册,避免报错。
@register_tool("search", allow_overwrite=True)
#〔讲解〕Search 类 = 搜索工具的定义。三个类属性(name/description/parameters)是给「模型/框架」看的工具说明书:
#〔讲解〕模型据此知道这个工具叫什么、干什么、要传什么参数(类似 OpenAI function-calling 的 schema)。
class Search(BaseTool):
    name = "search"
    description = "Performs a web search: supply a string 'query'; the tool retrieves the top 10 results for the query."
    #〔讲解〕参数 JSON Schema:声明本工具只接收一个必填字段 query(字符串)。框架据此校验/构造模型的调用参数。
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query string for the search."
            },
        },
        "required": ["query"],
    }

    #〔讲解〕构造函数:cfg 是框架传入的工具配置(可为空)。先走父类初始化,再把后端地址存为实例属性备用。
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.server_url = SEARCH_SERVER_URL

    #〔讲解〕真正干活的「同步」方法:把单个 query POST 给后端 /search,拿回检索结果文本。
    #〔讲解〕入参 query=搜索词;返回=结果字符串(成功时是后端 results 文本,失败时是 [Search] Error 开头的提示)。
    def search_with_server(self, query: str) -> str:
        """Send search request to serve_search.py backend."""
        try:
            #〔讲解〕向后端发 POST 请求,body 是 {"query": ...};timeout=30 防止后端卡死时无限等待。
            response = requests.post(
                f"{self.server_url}/search",
                json={"query": query},
                timeout=30
            )
            #〔讲解〕若 HTTP 状态码是 4xx/5xx(后端报错),主动抛异常,交给下面 except 统一处理。
            response.raise_for_status()
            data = response.json()

            # if data.get('cached'):
            #     print(f"[Search] Cache hit for: {query}")

            #〔讲解〕只取后端返回 JSON 里的 results 字段(检索结果正文)作为工具输出。
            return data['results']

        #〔讲解〕下面三个 except 把不同故障都「转成普通字符串」返回,而不是抛出异常——
        #〔讲解〕这样搜索失败不会中断整条 ReAct 轨迹,模型能读到错误提示后自行决定下一步。
        except requests.exceptions.ConnectionError:
            #〔讲解〕连不上后端(服务没起/地址错)。
            return "[Search] Error: Cannot connect to server."
        except requests.exceptions.Timeout:
            #〔讲解〕超过 30s 仍无响应。
            return "[Search] Error: Search request timed out."
        except Exception as e:
            #〔讲解〕兜底:其他任何异常(如 JSON 解析失败、缺 results 字段)都包成错误串返回。
            return f"[Search] Error: {str(e)}"

    #〔讲解〕框架实际调用工具的入口(异步)。params 是模型给的参数,**kwargs 收框架附带的其他信息。
    #〔讲解〕角色:统一解析参数 → 调底层同步检索 → 返回结果字符串给智能体。
    async def call(self, params: Union[str, dict], **kwargs) -> str:
        #〔讲解〕从参数里取 query;若结构不对(取不到该键)用裸 except 兜住,返回格式错误提示而非崩溃。
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        #〔讲解〕query 是单个字符串的常规情况。
        if isinstance(query, str):
            #〔讲解〕关键点:search_with_server 是阻塞的同步网络调用,这里用 asyncio.to_thread 丢到线程池跑,
            #〔讲解〕避免阻塞事件循环——多条轨迹并发时各自的搜索请求才能真正并行,不互相卡住。
            response = await asyncio.to_thread(self.search_with_server, query)
        else:
            # Handle list of queries
            #〔讲解〕兼容模型一次传「多个 query」(列表)的情况:断言确为 list,然后逐个搜。
            assert isinstance(query, list)
            responses = []
            #〔讲解〕逐个查询(串行 await),把每条结果收集起来。
            for q in query:
                res = await asyncio.to_thread(self.search_with_server, q)
                responses.append(res)
            #〔讲解〕用分隔线把多条结果拼成一整段文本返回,便于模型阅读区分。
            response = "\n=======\n".join(responses)

        return response

    #〔讲解〕输出本工具的 function-calling 定义(OpenAI 风格)。供需要标准 tool schema 的调用方使用。
    def get_tool_definitions(self):
        #〔讲解〕复制一份 parameters 再补 additionalProperties=False(禁止模型塞多余字段),不污染类属性原件。
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        #〔讲解〕组装成 {type:"function", function:{name,description,parameters}} 的标准结构返回。
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }
