# 〔讲解〕本文件 = 聚合(aggregation)环节的「策略基类与工具箱」。
# 〔讲解〕AggAgent pipeline:rollout/ 先为同一道题跑出 N 条 ReAct 搜索轨迹(每条含答案+判分+成本),
# 〔讲解〕aggregation/ 再用某种「策略(Strategy)」把这 N 条挑选/合并成一份最终答案,evaluation/ 负责判分。
# 〔讲解〕这里定义抽象基类 Strategy(子类如 majority-vote、llm_based、aggagent 都继承它),
# 〔讲解〕外加成本数据结构(CostBreakdown)、读判分/答案/置信度的公共取数函数。它是聚合层所有具体策略的「地基」。
"""Base strategy class for aggregation evaluation."""

import asyncio
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import threading
import time

# Suppress LiteLLM logging-worker noise that fires when litellm.acompletion()
# is called from worker threads (each thread has its own event loop, which
# triggers a queue-reinit race in LiteLLM's GLOBAL_LOGGING_WORKER).
# 〔讲解〕下面两行是「降噪」:LLM 策略会在多线程里各开一个事件循环调 litellm,
# 〔讲解〕会触发一堆无害但刷屏的 asyncio/RuntimeWarning 报错,这里把它们压成静默,纯日志卫生,不影响逻辑。
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="coroutine.*was never awaited"
)


# Model costs per 1M tokens (input, output) in USD
# 〔讲解〕模型价目表:键=模型名,值=(每百万 input token 美元, 每百万 output token 美元)。
# 〔讲解〕用于把 token 用量折算成「真实美元成本」,符合本项目「成本用真实货币不用 token count」的口径。
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "GLM-4.7-Flash": (0.07, 0.40),
    "MiniMax-M2.5": (0.30, 1.20),
    "gpt-oss-120b": (0.15, 0.60),
    "gemini-3-flash-preview": (0.5, 3.0),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.1": (1.25, 10),
    "Qwen3.5-122B-A10B": (0.26, 2.08),
}


# 〔讲解〕MetricResult:一条「指标结果」的轻量容器。value=指标值(如准确率),cost=可选的平均成本。
# 〔讲解〕@dataclass 让 Python 自动生成 __init__ 等样板代码,这里只是把 (值, 成本) 打包方便返回。
@dataclass
class MetricResult:
    """Result of a metric calculation with optional cost."""
    value: float
    cost: float | None = None


# 〔讲解〕CostBreakdown:把一次运行的成本拆成三块——rollout(跑轨迹的 LLM token 费)、
# 〔讲解〕tool(轨迹里搜索/抓网页等工具调用费)、aggregation(聚合阶段额外的 LLM 费)。
# 〔讲解〕拆开记是为了在报告里看清「钱花在哪一环」。
@dataclass
class CostBreakdown:
    """Breakdown of costs by component."""
    rollout: float = 0.0      # LLM token costs during rollout
    tool: float = 0.0         # Tool call costs during rollout (search, scrape)
    aggregation: float = 0.0  # Aggregation LLM costs (added by LLM strategies)

    # 〔讲解〕total:三块成本之和。@property 让它像普通属性一样用(写 obj.total 而非 obj.total()),只读不存储。
    @property
    def total(self) -> float:
        return self.rollout + self.tool + self.aggregation

    # 〔讲解〕重载「+」:两个 CostBreakdown 相加 = 各分项分别相加。这样累加多条轨迹的成本时可直接写 a + b。
    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            rollout=self.rollout + other.rollout,
            tool=self.tool + other.tool,
            aggregation=self.aggregation + other.aggregation,
        )

    # 〔讲解〕重载「/」:各分项同除以 n,常用于把「总成本」摊成「每题平均成本」。
    def __truediv__(self, n: float) -> "CostBreakdown":
        return CostBreakdown(
            rollout=self.rollout / n,
            tool=self.tool / n,
            aggregation=self.aggregation / n,
        )


# 〔讲解〕Strategy:聚合策略的抽象基类(ABC=不能直接实例化,必须被子类继承)。
# 〔讲解〕它定义了所有聚合策略的统一骨架:怎么算 metric@k、怎么算成本、怎么读判分/答案/置信度、怎么随机打破平局。
# 〔讲解〕具体「怎么从 N 条轨迹挑/合成一份答案」由子类实现 calculate_at_k。
class Strategy(ABC):
    """Abstract base class for aggregation strategies."""

    # 〔讲解〕name:策略名,打印结果时作标签(如 "majority@4")。子类会覆盖成自己的名字。
    name: str = "base"

    # 〔讲解〕构造函数:task=任务/数据集名(不同任务判分口径不同),seed=随机种子。
    # 〔讲解〕**kwargs 吞掉多余参数,方便子类传额外配置而不报错。
    def __init__(self, task: str = "browsecomp", seed: int = 42, **kwargs):
        """Initialize strategy.

        Args:
            task: Task type (browsecomp, browsecomp-plus, deepsearchqa, hle, researchrubrics, healthbench)
            seed: Random seed for tie-breaking
        """
        self.task = task
        # 〔讲解〕self.rng:独立的随机数发生器并用 seed 固定,保证「打破平局」可复现(同 seed → 同结果)。
        self.rng = random.Random(seed)

    # 〔讲解〕calculate_at_k:核心抽象方法,子类必须实现。给定每题的若干条轨迹和预算 k,
    # 〔讲解〕算出该策略在 k 条候选下的指标值(若开了成本统计则额外返回平均成本)。
    # 〔讲解〕@abstractmethod 强制子类提供实现,否则无法实例化——这是「框架挖坑、子类填坑」的契约。
    @abstractmethod
    def calculate_at_k(
        self, results: dict[str, list[dict]], k: int
    ) -> float | tuple[float, float]:
        """
        Calculate the metric at k for all problems.

        Returns:
            If model costs are set: tuple of (metric_value, average_cost)
            Otherwise: metric_value only
        """
        pass

    # 〔讲解〕run:对外的总入口。对一串 k 值(默认 1,2,4,...,N 的等比序列)逐个调 calculate_at_k,
    # 〔讲解〕收集结果、格式化打印一张「metric@k」表,返回 {k: MetricResult}。
    # 〔讲解〕这就是「随着可用轨迹数 k 增大,指标如何变化」的扫描(scaling 曲线)。
    def run(
        self,
        results: dict[str, list[dict]],
        n: int,
        k_values: list[int] | None = None,
    ) -> dict[int, MetricResult]:
        """
        Run strategy for k=1, 2, 4, ..., N (or specified k_values).

        Returns dict mapping k to MetricResult.
        """
        scores = {}
        # 〔讲解〕未显式指定 k_values 时,自动生成 1,2,4,8,... 直到不超过 N 的等比序列(典型的 power-of-2 扫描)。
        if k_values is None:
            k_values = []
            k = 1
            while k <= n:
                k_values.append(k)
                k *= 2

        # 〔讲解〕预先算出最长标签宽度(如 "majority@128"),好让后面打印的表格左列对齐。
        label_width = max(len(f"{self.name}@{k}") for k in k_values)

        for k in k_values:
            # 〔讲解〕调子类实现算本 k 的指标;返回可能是「只有指标」或「(指标, 成本)」两种形态。
            result = self.calculate_at_k(results, k)
            # 〔讲解〕取标准差字符串(若子类在算时存了每题分数才有值);标准差反映跨题波动。
            std_str = self._format_std()

            # 〔讲解〕用 isinstance 判返回是不是元组,以区分「带成本」与「不带成本」两种返回形态,分别落库。
            if isinstance(result, tuple):
                metric, cost = result
                scores[k] = MetricResult(value=metric, cost=cost)
            else:
                metric, cost = result, None
                scores[k] = MetricResult(value=metric)

            # 〔讲解〕下面三行是排版细节:固定各列宽度,让百分比/标准差/成本三列在终端里对齐。
            label = f"{self.name}@{k}"
            std_col = std_str.ljust(9)  # fixed width so cost column always aligns
            cost_str = f"  (${cost:.4f}/prob)" if cost is not None else ""
            # 〔讲解〕metric 是 0~1 的小数,乘 100 显示成百分比;有成本时附「$x/prob」即每题平均花费。
            print(f"  {label:<{label_width}}  {metric * 100:6.2f}%{std_col}{cost_str}")

        return scores

    # 〔讲解〕calculate_trajectory_cost_breakdown:读「单条轨迹」已在 rollout 阶段算好的成本。
    # 〔讲解〕成本不在这里现算,而是 rollout 时由 rollout/utils.compute_rollout_cost 算好存进 data["cost"]。
    # 〔讲解〕本方法只是「读出来打包」;LLM 类策略之后会再叠加 aggregation 那一块。
    def calculate_trajectory_cost_breakdown(self, data: dict) -> CostBreakdown:
        """Return pre-computed rollout cost from result data.

        Cost is computed at rollout time by rollout/utils.compute_rollout_cost
        and saved as data["cost"] = {"rollout": ..., "tool": ...}.  LLM-based
        strategies will add aggregation cost on top via CostBreakdown.aggregation.
        """
        # 〔讲解〕.get(..., 默认值) 防止某条数据缺字段时报 KeyError,缺了就当 0,稳健取数。
        cost = data.get("cost", {})
        return CostBreakdown(
            rollout=cost.get("rollout", 0.0),
            tool=cost.get("tool", 0.0),
        )

    # 〔讲解〕calculate_combination_cost_breakdown:把一组(k 条)轨迹的成本逐条相加,得到这次组合的总成本。
    # 〔讲解〕这里直接用了上面重载的「+」运算符把多个 CostBreakdown 累加起来。
    def calculate_combination_cost_breakdown(self, runs: list[dict]) -> CostBreakdown:
        """Calculate cost breakdown for a combination of k trajectories."""
        breakdown = CostBreakdown()
        for r in runs:
            breakdown = breakdown + self.calculate_trajectory_cost_breakdown(r)
        return breakdown

    # 〔讲解〕get_n:推断 N(每题跑了几条轨迹)。@staticmethod 表示不依赖实例状态,是纯工具函数。
    @staticmethod
    def get_n(results: dict[str, list[dict]]) -> int:
        """Get N (number of runs per problem) from results."""
        if not results:
            return 0
        # 〔讲解〕results 是 {题目id: [轨迹1, 轨迹2, ...]};取「第一道题」的轨迹列表长度即 N。
        # 〔讲解〕next(iter(dict.values())) 是「取字典里第一个值」的惯用法,假设每题轨迹数一致。
        return len(next(iter(results.values())))

    # 〔讲解〕is_correct:把一条轨迹的判分结果归一成「对/错」或「分数」。
    # 〔讲解〕不同任务判分口径不同,所以按 self.task 分支处理——这是适配多个 benchmark 的关键开关。
    # 〔讲解〕返回:二元任务返 bool(对/错),连续打分任务(healthbench/researchrubrics)返 float 分数。
    def is_correct(self, data: dict) -> bool | float:
        """
        Get correctness score for a result.

        Returns bool for binary tasks, float for continuous-score tasks.
        """
        if self.task == "deepsearchqa":
            # 〔讲解〕deepsearchqa:判分器给的 all_correct 必须严格等于 True 才算对(== True 排除 None/缺失)。
            return data.get("auto_judge", {}).get("all_correct") == True
        elif self.task in ("healthbench", "researchrubrics"):
            # 〔讲解〕这两类是「打分制」:优先直接取判分器给出的总分 overall_score。
            score = data.get("auto_judge", {}).get("metrics", {}).get("overall_score")
            if score is not None:
                return score
            # 〔讲解〕没有现成总分时,退回到「逐条评分项(rubric)」自己算:先拿到评分项列表(两个来源任选其一)。
            rubric_items = (
                data.get("auto_judge", {}).get("rubric_items_with_grades")
                or data.get("rubrics", [])
            )
            if rubric_items:
                # 〔讲解〕两个数据集字段名不同:healthbench 用 "points",researchrubrics 用 "weight"。
                field = "points" if self.task == "healthbench" else "weight"
                # 〔讲解〕total_possible = 所有「正分项」的满分总和,作为归一化分母。
                total_possible = sum(r.get(field, 0) for r in rubric_items if r.get(field, 0) > 0)
                if total_possible > 0:
                    # 〔讲解〕min_val = 所有「负分项」之和(惩罚项),除以满分总和得到这条结果可达的最低归一化分。
                    # 〔讲解〕注:这里返回的是负分项占比,作为该轨迹的保底/下界估计。
                    min_val = sum(r.get(field, 0) for r in rubric_items if r.get(field, 0) < 0)
                    return min_val / total_possible
            return 0.0
        # 〔讲解〕其余任务(browsecomp 系、hle 等)默认二元判分:correctness 字段为 "correct" 才算对。
        return data.get("auto_judge", {}).get("correctness") == "correct"

    # 〔讲解〕extract_answer:从一条轨迹里取「最终答案文本」。优先判分器抽取的 extracted_final_answer,
    # 〔讲解〕没有就退回模型原始 prediction。`A or B` 模式:A 为空(None/"")时取 B。
    @staticmethod
    def extract_answer(data: dict) -> str | None:
        """Extract answer."""
        return data.get("auto_judge", {}).get("extracted_final_answer") or data.get("prediction", "")

    # 〔讲解〕extract_confidence:取模型自报的置信度。原始是 0~100,这里除以 100 归一到 0~1,缺失则默认 0。
    # 〔讲解〕置信度可被某些策略用作「该信哪条答案」的权重。
    @staticmethod
    def extract_confidence(data: dict) -> float:
        """Extract confidence from auto_judge (0–100 scale), normalized to 0–1. Default 0.0."""
        confidence = data.get("auto_judge", {}).get("confidence")
        if confidence is not None:
            return float(confidence) / 100
        return 0.0

    # 〔讲解〕get_tool_count:统计一条轨迹里总共调了多少次工具(搜索/抓网页等),反映「检索努力程度」。
    @staticmethod
    def get_tool_count(data: dict) -> int:
        """Get total tool call count from a result dict."""
        try:
            # 〔讲解〕首选路径:从 debug_data.tool_usage(形如 {工具名: 次数})把各工具次数求和。
            tool_usage = data.get("debug_data", {}).get("tool_usage", {})
            if tool_usage:
                return sum(tool_usage.values())
        except (KeyError, TypeError):
            # 〔讲解〕字段结构异常时不抛错,直接落到下面的兜底统计法。
            pass
        # Fallback: count tool-role messages
        # 〔讲解〕兜底:没有 tool_usage 时,数对话记录里 role=="tool" 的消息条数当作工具调用次数。
        return sum(1 for m in data.get("messages", []) if m.get("role") == "tool")

    # 〔讲解〕_format_std:把「每题分数」算成标准差字符串(如 " ± 3.21%"),供 run() 打印时显示波动。
    # 〔讲解〕下划线开头表示「内部方法」。依赖子类是否事先把每题分数存进 self._last_problem_scores。
    def _format_std(self) -> str:
        """Return ' ± X.XX' std string or '' if not available."""
        # 〔讲解〕getattr(self, 名, 默认) 安全取属性:子类没设这个属性也不会报错,返回 None。
        scores = getattr(self, "_last_problem_scores", None)
        # 〔讲解〕样本不足 2 个算不出标准差,直接返回空串(表里那一列就空着)。
        if not scores or len(scores) < 2:
            return ""
        n = len(scores)
        mean = sum(scores) / n
        # 〔讲解〕总体标准差公式:sqrt(Σ(x-mean)² / n)。这里用 n 而非 n-1(总体口径,非样本无偏)。
        std = (sum((s - mean) ** 2 for s in scores) / n) ** 0.5
        return f" ± {std * 100:.2f}%"

    # 〔讲解〕select_with_tie_breaking:通用「选最优、平局随机打破」工具。
    # 〔讲解〕入参:items=候选列表;key_func=从每个候选取出「比较值」的函数(None=纯随机选);select_func=max 或 min。
    # 〔讲解〕「随机打破平局」配合固定 seed,保证多个并列最优时选择稳定可复现——这是很多投票/排序策略的底座。
    def select_with_tie_breaking(self, items: list, key_func=None, select_func=max):
        """
        Select item with best key value, breaking ties randomly.

        Args:
            items: List of items
            key_func: Function to extract comparison value (None → random selection)
            select_func: max or min

        Returns:
            Selected item, or None if items is empty.
        """
        if not items:
            return None
        # 〔讲解〕没给比较函数 → 纯随机:多于一个就随机挑一个,否则直接返回唯一项。
        if key_func is None:
            return self.rng.choice(items) if len(items) > 1 else items[0]

        # 〔讲解〕给每个候选算比较值,再用 select_func(max/min)找出「最优值」。
        values = [key_func(item) for item in items]
        best_value = select_func(values)
        # 〔讲解〕zip 把候选与其比较值配对,挑出所有「等于最优值」的并列项;若多于一个则随机选一个打破平局。
        tied_items = [item for item, v in zip(items, values) if v == best_value]
        return self.rng.choice(tied_items) if len(tied_items) > 1 else tied_items[0]


# ---------------------------------------------------------------------------
# Scoring helpers (shared by llm_based and aggagent strategies)
# ---------------------------------------------------------------------------


# 〔讲解〕_compute_score:给一段「聚合后的预测」临时判分用的同步封装。
# 〔讲解〕背景:evaluation 模块的判分器是异步(async)接口,但聚合策略里有些地方是同步代码,
# 〔讲解〕这里用 asyncio.run 把异步判分「跑成同步」,方便在普通函数里直接拿分数。
# 〔讲解〕入参:prediction=待判文本,item=题目(含标准答案/评分细则),task=任务名,llm=判分用的大模型(默认 gpt-4.1)。
def _compute_score(prediction: str, item: dict, task: str, llm: str = "gpt-4.1") -> dict:
    """Synchronous wrapper for the async evaluator compute_score."""
    # 〔讲解〕函数内 import:避免模块顶层就强依赖 evaluation(打破潜在循环依赖,也省得未用到时加载)。
    from evaluation import get_evaluator
    # 〔讲解〕按任务名拿到对应的判分器(不同 benchmark 有各自的评分逻辑)。
    evaluator = get_evaluator(task)

    # 〔讲解〕把异步判分调用包成内部协程,交给 asyncio.run 执行并取回结果。
    async def _run():
        return await evaluator.compute_score(prediction=prediction, item=item, llm=llm)

    # 〔讲解〕asyncio.run 新建事件循环跑完 _run 再关掉,返回判分结果字典。
    return asyncio.run(_run())
