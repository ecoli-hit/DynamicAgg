#〔讲解〕模块说明:本文件实现 Pass@k 评测指标(metric strategy),属于 aggregation/_strategy/ 这一环。
#〔讲解〕在 AggAgent pipeline 里,rollout 先为每道题跑出 N 条独立轨迹并各自判分,本文件把这 N 条轨迹
#〔讲解〕的判分结果汇总成"Pass@k"曲线 —— 即"从 N 条里任取 k 条,至少有 1 条答对"的平均概率。
#〔讲解〕它继承 base.py 的 Strategy 抽象类(复用 get_n / is_correct / 成本核算等),与 llm_based/aggagent
#〔讲解〕等其他聚合策略并列,共同回答"给我 k 条预算,能拿到多高的准确率/花多少钱"。
"""Pass@k strategy for aggregation evaluation."""

import math
from collections import defaultdict
#〔讲解〕combinations:标准库的组合迭代器,用来枚举"从 N 条轨迹里挑 k 条"的全部 C(N,k) 种取法。
from itertools import combinations
#〔讲解〕从基类模块导入:Strategy(抽象基类,定义评测策略的统一接口)、CostBreakdown(成本三分账:rollout/tool/aggregation)。
from .base import Strategy, CostBreakdown


#〔讲解〕PassStrategy:Pass@k 指标的"显式枚举"实现。
#〔讲解〕角色 —— 它是一种具体评测策略,本身不做聚合,只统计"多采样几条能否命中正确答案"。
#〔讲解〕做法是真的把 C(N,k) 种组合全列出来逐一算(而非用概率公式估计),所以 N 大时组合数会爆炸,但结果精确。
class PassStrategy(Strategy):
    """
    Pass@k strategy using explicit enumeration.

    For each C(N,k) combination, checks if at least one trajectory is correct
    (taking max score across the k trajectories). Averages across all combinations.
    """

    #〔讲解〕name:策略标识,基类 run() 打印结果时会拼成 "Pass@k" 这样的标签。
    name = "Pass"

    #〔讲解〕calculate_at_k:核心方法,计算"给定 k 时"的 Pass@k 指标与平均成本。
    #〔讲解〕入参 results:{题目id -> 该题 N 条轨迹的结果列表},每条结果是含判分/成本的 dict。
    #〔讲解〕入参 k:本次评测的采样预算(挑几条)。返回 (metric, 平均总成本) 二元组。
    def calculate_at_k(
        self, results: dict[str, list[dict]], k: int
    ) -> tuple[float, float]:
        #〔讲解〕没有任何题目数据时,指标与成本都返回 0,避免后续除零。
        if not results:
            return (0.0, 0.0)

        #〔讲解〕n = 每题的轨迹条数 N(基类 get_n 取第一题的列表长度作为统一 N)。
        n = self.get_n(results)
        #〔讲解〕k 不能超过实际有的轨迹数 N,否则"从 N 里挑 k"无意义,直接报错。
        if k > n:
            raise ValueError(f"k ({k}) cannot be greater than N ({n})")

        #〔讲解〕num_combinations = C(n,k),即每道题需要枚举的组合总数,后面用它把成本摊平。
        num_combinations = math.comb(n, k)
        #〔讲解〕combo_correct:按"组合编号 combo_idx"分桶,收集每个组合在各题上的命中情况(0/1 或连续分)。
        #〔讲解〕用 defaultdict(list) 是为了第一次访问某编号时自动建空列表,省去手动初始化。
        combo_correct: dict[int, list] = defaultdict(list)
        #〔讲解〕problem_breakdowns:逐题收集"该题的平均成本三分账",最后再跨题平均。
        problem_breakdowns = []

        #〔讲解〕遍历每道题及其 N 条轨迹结果。
        for problem_id, run_results in results.items():
            #〔讲解〕数据卫生检查:若某题轨迹数不等于 N(缺跑/多跑),跳过该题,保证所有题在同一 N 下可比。
            if len(run_results) != n:
                continue

            #〔讲解〕total_breakdown:累加本题所有组合的成本,初始化为全 0 的成本账本。
            total_breakdown = CostBreakdown()

            #〔讲解〕枚举本题的每一种 k 元组合;combo 是被选中的 k 个轨迹下标元组,combo_idx 是该组合的固定编号。
            for combo_idx, combo in enumerate(combinations(range(n), k)):
                #〔讲解〕按下标取出这 k 条被选中的轨迹结果。
                combo_runs = [run_results[i] for i in combo]
                #〔讲解〕Pass@k 的精髓:取 k 条里"最高"的判分。is_correct 返回 bool/float,
                #〔讲解〕对二元任务 max 即"只要有一条对就算 1(命中)";对连续打分任务则取最好那条的分。
                max_score = max(self.is_correct(r) for r in combo_runs)
                #〔讲解〕把该组合在本题上的命中分记入对应编号的桶,供之后"按组合编号横向平均"。
                combo_correct[combo_idx].append(max_score)
                #〔讲解〕把这 k 条轨迹的成本(rollout+tool)累加进本题总账;k 条的成本是相加关系。
                total_breakdown = total_breakdown + self.calculate_combination_cost_breakdown(combo_runs)

            #〔讲解〕本题总成本 ÷ 组合数 = 本题"每个组合的平均成本",作为该题的代表成本存起来。
            problem_breakdowns.append(total_breakdown / num_combinations)

        #〔讲解〕对每个组合编号桶:把它在所有题上的命中分求平均 = 该"特定组合"的整体准确率。
        combo_accuracies = [sum(v) / len(v) for v in combo_correct.values()]
        #〔讲解〕最终 Pass@k = 所有组合准确率再求平均(等价于"随机抽一个 k 组合的期望准确率");无数据则 0。
        metric = sum(combo_accuracies) / len(combo_accuracies) if combo_accuracies else 0.0
        #〔讲解〕把每个组合的准确率列表暂存到实例上,供基类 _format_std() 计算并打印标准差(衡量组合间波动)。
        self._last_problem_scores = combo_accuracies

        #〔讲解〕下面把"各题平均成本"再跨题平均,得到整个数据集层面的平均成本三分账。
        avg_breakdown = CostBreakdown()
        #〔讲解〕先把所有题的成本账本逐个相加(CostBreakdown 重载了 + 号,按 rollout/tool/aggregation 分项相加)。
        for b in problem_breakdowns:
            avg_breakdown = avg_breakdown + b
        #〔讲解〕再除以题数得到每题平均;有题目时才除,避免空列表除零。
        if problem_breakdowns:
            avg_breakdown = avg_breakdown / len(problem_breakdowns)
        #〔讲解〕暂存成本明细到实例,外部(如报告/打印)可读取分项成本而不只是总额。
        self._last_cost_breakdown = avg_breakdown
        #〔讲解〕返回 (Pass@k 指标, 平均总成本);.total 把 rollout+tool+aggregation 三项加成一个数。
        return (metric, avg_breakdown.total)
