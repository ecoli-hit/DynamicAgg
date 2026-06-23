# 〔讲解〕本文件实现 BON@k(Best of N)这个「聚合评估指标」,属于 aggregation/ 一环里
# 〔讲解〕的「非 LLM、纯统计」策略之一(同目录还有 mv.py 多数投票、pass_strategy.py 等)。
# 〔讲解〕用途:rollout/ 对同一道题跑了 N 条独立轨迹后,本策略不真的去"融合"答案,而是
# 〔讲解〕回答一个评估问题——"如果我手上有 k 条轨迹,按 confidence 挑出最自信的那条当最终答案,
# 〔讲解〕正确率是多少?"。它通过枚举所有 C(N,k) 个 k 元子集来无偏估计这个 k 下的期望准确率。
# 〔讲解〕上游 = rollout 产出 + evaluation 判分后的结果(每条轨迹带 auto_judge);下游 = 报表/对比表。
"""BON (Best of N) strategy for aggregation evaluation."""

import math
from collections import defaultdict
# 〔讲解〕combinations 用来枚举"从 N 条里取 k 条"的所有组合,是本策略的核心——
# 〔讲解〕BON@k 的定义就建立在遍历全部 C(N,k) 个子集之上。
from itertools import combinations
# 〔讲解〕Strategy = 抽象基类(提供 get_n / is_correct / extract_confidence /
# 〔讲解〕select_with_tie_breaking 等公共方法);CostBreakdown = 把成本拆成
# 〔讲解〕rollout/tool/aggregation 三项的小数据类,支持 + 和 / 运算便于求和取平均。
from .base import Strategy, CostBreakdown


# 〔讲解〕BONStrategy:实现 BON@k 指标的具体策略类,继承 Strategy。
# 〔讲解〕角色 = 一个"评估器",不改写答案、不调 LLM,只统计"挑最自信那条"的准确率与成本。
# 〔讲解〕核心方法是 calculate_at_k(给定 k 返回 准确率+平均成本),由基类 run() 对 k=1,2,4,...,N 循环调用。
class BONStrategy(Strategy):
    """
    BON@k (Best of N) strategy.

    For each C(N,k) combination, select the run with the highest confidence
    and check its correctness.
    """

    # 〔讲解〕name 是该策略的标识符,用于日志/报表打印(如 "BON@4"),覆盖基类默认的 "base"。
    name = "BON"

    # 〔讲解〕calculate_at_k:本策略的主入口。给定全部题目的结果与一个 k,算出 BON@k 的准确率和每题平均成本。
    # 〔讲解〕入参 results:{题目id -> 该题的 N 条轨迹结果列表},每条轨迹是一个 dict(含 auto_judge、cost 等)。
    # 〔讲解〕入参 k:每个子集取几条轨迹。返回 (metric, avg_cost) 二元组——metric 是 0~1 的准确率。
    def calculate_at_k(
        self, results: dict[str, list[dict]], k: int
    ) -> tuple[float, float]:
        # 〔讲解〕没有任何题目时直接返回 (0,0),避免后续 next(iter(...)) / 除零报错。
        if not results:
            return (0.0, 0.0)

        # 〔讲解〕从结果里推断 N(每题轨迹条数):get_n 取第一道题的列表长度,默认所有题 N 一致。
        n = self.get_n(results)
        # 〔讲解〕k 不能超过 N——"从 N 条里取 k 条"在 k>N 时无意义,属调用方传参错误,直接抛异常拦住。
        if k > n:
            raise ValueError(f"k ({k}) cannot be greater than N ({n})")

        # 〔讲解〕C(N,k):每道题会枚举出多少个 k 元子集。后面用它把每题累加的成本平均回"单个子集"口径。
        num_combinations = math.comb(n, k)
        # 〔讲解〕combo_correct:{组合下标 -> 各题在该组合上的对错列表}。
        # 〔讲解〕关键设计:用"同一个组合下标 combo_idx 跨题对齐"。每道题枚举子集的顺序由 combinations 决定且固定,
        # 〔讲解〕因此 combo_idx=0 在所有题里都指"同一种取法",最后按 combo_idx 分组求准确率 = N 个并行评估通道的逐通道准确率。
        combo_correct: dict[int, list] = defaultdict(list)
        # 〔讲解〕problem_breakdowns:收集每道题"单个子集"的平均成本(已除以 num_combinations),供最后跨题再平均。
        problem_breakdowns = []

        # 〔讲解〕外层循环:逐题处理。problem_id=题目id,run_results=该题的轨迹列表。
        for problem_id, run_results in results.items():
            # 〔讲解〕轨迹条数不等于 N 的题(数据缺失/未跑满)直接跳过,保证组合枚举与统计口径一致。
            if len(run_results) != n:
                continue

            # 〔讲解〕本题所有子集的成本累加器,从全 0 开始;CostBreakdown 重载了 + 故可直接相加。
            total_breakdown = CostBreakdown()

            # 〔讲解〕内层循环:枚举本题全部 C(N,k) 个子集。combo 是一个长度为 k 的下标元组,如 (0,2,3)。
            for combo_idx, combo in enumerate(combinations(range(n), k)):
                # 〔讲解〕按下标把这次子集对应的 k 条轨迹取出来。
                combo_runs = [run_results[i] for i in combo]
                # 〔讲解〕BON 核心一步:在这 k 条里挑 confidence 最高的那条当代表(max)。
                # 〔讲解〕select_with_tie_breaking 用 extract_confidence 取每条的置信度;若并列最高则用固定 seed 的 rng 随机打破平局(保证可复现)。
                best_run = self.select_with_tie_breaking(combo_runs, self.extract_confidence, max)
                # 〔讲解〕记录"这条最自信的轨迹答对没有"。is_correct 据 task 类型返回 bool 或连续分(如 healthbench 的 rubric 得分)。
                combo_correct[combo_idx].append(self.is_correct(best_run))
                # 〔讲解〕把这 k 条轨迹的成本(rollout+tool)累加进本题总账——注意成本算的是"这一子集真实跑出来要花多少",与挑哪条无关,故是 k 条之和。
                total_breakdown = total_breakdown + self.calculate_combination_cost_breakdown(combo_runs)

            # 〔讲解〕本题成本除以子集数,折算成"单个 k 元子集"的平均成本,使其与 metric 的"每子集"口径一致。
            problem_breakdowns.append(total_breakdown / num_combinations)

        # 〔讲解〕对每个组合通道(combo_idx)算其跨题准确率 = 对错列表求和/题数。得到 C(N,k) 个准确率。
        combo_accuracies = [sum(v) / len(v) for v in combo_correct.values()]
        # 〔讲解〕BON@k 最终指标 = 这些通道准确率的平均(即对"随机抽一个 k 元子集"求期望准确率);无数据则记 0。
        metric = sum(combo_accuracies) / len(combo_accuracies) if combo_accuracies else 0.0
        # 〔讲解〕把逐通道准确率存到基类约定的 _last_problem_scores,供外部算标准差/误差棒等使用。
        self._last_problem_scores = combo_accuracies

        # 〔讲解〕以下把各题的"单子集平均成本"再跨题求平均,得到最终 avg_breakdown。
        avg_breakdown = CostBreakdown()
        for b in problem_breakdowns:
            avg_breakdown = avg_breakdown + b
        # 〔讲解〕有题才除以题数取平均;无题则保持全 0,避免除零。
        if problem_breakdowns:
            avg_breakdown = avg_breakdown / len(problem_breakdowns)
        # 〔讲解〕把成本明细缓存到基类约定字段,供报表读取 rollout/tool/aggregation 分项。
        self._last_cost_breakdown = avg_breakdown
        # 〔讲解〕返回 (准确率, 平均总成本)。.total = rollout+tool+aggregation 三项之和。
        return (metric, avg_breakdown.total)
