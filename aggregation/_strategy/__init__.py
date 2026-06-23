# 〔讲解〕本文件是 aggregation/_strategy 子包的"门面 / 注册中心"(package __init__)。
# 〔讲解〕在 pipeline 里属于「聚合」这一环:rollout/ 先跑出多条 ReAct 搜索轨迹,
# 〔讲解〕然后由这里登记的各种"聚合策略"把多条轨迹合成一份最终答案,最后交给 evaluation/ 判分。
# 〔讲解〕作用有三:① 把分散在各子模块里的策略类统一 import 进来做一个出口;
# 〔讲解〕② 用 STRATEGIES / HEURISTIC_STRATEGIES 两张"名字→类"字典做注册表;
# 〔讲解〕③ 提供 get_strategy / get_heuristic_strategies 两个工厂函数,让上层只凭字符串名就能拿到策略实例。
"""Aggregation strategies."""

# 〔讲解〕下面这组 import 把各策略类从同包内的子模块"提"到本包顶层,这样外部可以写
# 〔讲解〕`from aggregation._strategy import AggAgent` 而不必关心它具体藏在哪个文件。
# 〔讲解〕base 里是抽象基类 Strategy(所有策略的共同接口)以及两个结果数据结构。
from .base import Strategy, MetricResult, CostBreakdown
# 〔讲解〕PassStrategy:最朴素的"直通/不聚合"基线(通常直接取第一条或单条轨迹的答案)。
from .pass_strategy import PassStrategy
# 〔讲解〕MV=Majority Vote 多数投票,WMV=Weighted MV 加权多数投票;均为启发式(不调 LLM)。
from .mv import MVStrategy, WMVStrategy
# 〔讲解〕BON=Best-of-N,从 N 条候选里按某种打分挑最好的一条。
from .bon import BONStrategy
# 〔讲解〕FewTool:少/按工具维度筛选的启发式策略。
from .fewtool import FewToolStrategy
# 〔讲解〕llm_based 里是"靠大模型来聚合"的策略:SolAgg(聚合各条 solution 答案)、SummAgg(先摘要再聚合)。
from .llm_based import SolAgg, SummAgg
# 〔讲解〕AggAgent:本框架的主角——一个会调用工具、能回读原始轨迹后再产出聚合答案的"聚合智能体"。
from .aggagent import AggAgent

# 〔讲解〕__all__ 显式声明 `from ... import *` 时对外暴露的名字清单(含类、两张注册表、两个工厂函数),
# 〔讲解〕起到"公开 API 白名单"的作用,避免把内部实现细节也一并导出。
__all__ = [
    "Strategy",
    "MetricResult",
    "CostBreakdown",
    "PassStrategy",
    "MVStrategy",
    "WMVStrategy",
    "BONStrategy",
    "FewToolStrategy",
    "SolAgg",
    "SummAgg",
    "AggAgent",
    "STRATEGIES",
    "HEURISTIC_STRATEGIES",
    "get_strategy",
    "get_heuristic_strategies",
]

# 〔讲解〕STRATEGIES:全部策略的注册表,键是命令行/配置里用的短名字符串,值是对应的策略类(注意存的是"类"本身,不是实例)。
# 〔讲解〕这种"字符串→类"的查表方式,让实验脚本只需传 `--strategy aggagent` 就能动态选用策略,无需到处写 if/else。
# All strategies (heuristic + LLM-based)
STRATEGIES = {
    "pass": PassStrategy,
    "mv": MVStrategy,
    "wmv": WMVStrategy,
    "bon": BONStrategy,
    "fewtool": FewToolStrategy,
    "solagg": SolAgg,
    "summagg": SummAgg,
    "aggagent": AggAgent,
}

# 〔讲解〕HEURISTIC_STRATEGIES:只包含"纯启发式、不调用 LLM"的子集(因此没有 solagg/summagg/aggagent)。
# 〔讲解〕意义在于这些策略零 LLM 成本、可批量快速跑,常用作 baseline 对照,或一次性把它们全部跑一遍做对比。
# Heuristic-only (no LLM calls)
HEURISTIC_STRATEGIES = {
    "pass": PassStrategy,
    "mv": MVStrategy,
    "wmv": WMVStrategy,
    "bon": BONStrategy,
    "fewtool": FewToolStrategy,
}


# 〔讲解〕get_strategy:按名字取单个策略实例的"工厂函数"。
# 〔讲解〕入参 name=策略短名;**kwargs=透传给策略构造函数的参数(各策略各取所需)。返回:一个已实例化的 Strategy 对象。
def get_strategy(name: str, **kwargs) -> Strategy:
    """
    Get a strategy instance by name.

    Args:
        name: Strategy name (pass, mv, wmv, bon, fewtool, sol, summ)
        **kwargs: Forwarded to the strategy constructor; unknown keys are absorbed via **kwargs
    """
    # 〔讲解〕先做合法性校验:名字不在注册表里就抛 ValueError,并在报错信息里列出所有可用名字,方便排错。
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    # 〔讲解〕STRATEGIES[name] 取出"类",紧跟 (**kwargs) 即调用其构造函数完成实例化——这就是"按名建对象"的核心一行。
    return STRATEGIES[name](**kwargs)


# 〔讲解〕get_heuristic_strategies:一次性实例化"所有启发式策略",返回它们的实例列表。
# 〔讲解〕典型用途:在一次实验里把全部非 LLM 基线一并跑出来做横向对比。**kwargs 同样透传给每个构造函数。
def get_heuristic_strategies(**kwargs) -> list[Strategy]:
    """
    Get instances of all heuristic (non-LLM) strategies.

    Args:
        **kwargs: Forwarded to each constructor; unknown keys are absorbed via **kwargs
    """
    # 〔讲解〕列表推导:遍历 HEURISTIC_STRATEGIES 的所有"类"(.values()),逐个 cls(**kwargs) 实例化,收集成列表返回。
    return [cls(**kwargs) for cls in HEURISTIC_STRATEGIES.values()]
