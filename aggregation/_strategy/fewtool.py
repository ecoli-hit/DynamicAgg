#〔讲解〕本文件 = aggregation 评测层里的一个「聚合策略」实现:FewTool@k。
#〔讲解〕它属于 pipeline 的「评测/聚合度量」环:rollout/ 先把每道题跑出 N 条 ReAct 搜索轨迹,
#〔讲解〕这里不调任何大模型,而是用一条朴素规则从 k 条轨迹里"选一条"当代表——选工具调用次数最少的那条,
#〔讲解〕再看它对不对。它是一个「成本敏感」的对照基线(baseline),和真正会调工具读轨迹做聚合的 aggagent 策略形成对比。
#〔讲解〕上游:rollout 产出的结果字典(每题 N 条轨迹);下游:run()/汇报脚本拿它的 (准确率, 平均成本) 画 cost-accuracy 曲线。
"""FewTool strategy for aggregation evaluation."""

import math
from collections import defaultdict
#〔讲解〕combinations:从 N 条轨迹里枚举所有 C(N,k) 个「选 k 条」的子集,FewTool@k 的核心就是对每个子集做一次选择。
from itertools import combinations
#〔讲解〕Strategy = 所有策略的抽象基类(提供 get_n / is_correct / select_with_tie_breaking 等公共方法);
#〔讲解〕CostBreakdown = 成本三分量容器(rollout / tool / aggregation),支持 + 和 / 运算用于累加和求平均。
from .base import Strategy, CostBreakdown


#〔讲解〕FewToolStrategy:把"用最少工具调用的那条轨迹"作为答案的策略类,继承 Strategy。
#〔讲解〕直觉:工具调得越少 = 越省钱/越自信,赌它也对——用来衡量"省工具"和"答对"是否兼得。
class FewToolStrategy(Strategy):
    """
    FewTool@k strategy.

    For each C(N,k) combination, select the run that used the fewest tool calls
    and check its correctness.
    """

    #〔讲解〕name:策略标识,打印结果(如 "FewTool@4  62.50%")和注册表里都用它。
    name = "FewTool"

    #〔讲解〕calculate_at_k:本策略的核心方法。给定全部题目的轨迹结果和参数 k,
    #〔讲解〕算出 FewTool@k 的整体准确率,并返回 (准确率 metric, 平均每题成本 total)。
    #〔讲解〕入参 results: {题目id: [N条轨迹dict, ...]};k: 每个组合取几条。返回 tuple[float, float]。
    def calculate_at_k(
        self, results: dict[str, list[dict]], k: int
    ) -> tuple[float, float]:
        #〔讲解〕没有任何数据时直接返回 (0,0),避免后面除零/取 next() 报错。
        if not results:
            return (0.0, 0.0)

        #〔讲解〕N = 每题跑了多少条轨迹(取任一题的列表长度);k 不能超过 N,否则组合无意义。
        n = self.get_n(results)
        if k > n:
            raise ValueError(f"k ({k}) cannot be greater than N ({n})")

        #〔讲解〕C(N,k):一道题能切出多少个「选 k 条」的组合,后面拿它给单题成本求平均。
        num_combinations = math.comb(n, k)
        #〔讲解〕combo_correct:键=组合编号(第几个 C(N,k) 组合),值=该组合在所有题上的对错列表(0/1 或分数)。
        #〔讲解〕这样设计是为了「按组合位置」横向聚合——每个组合编号在所有题里都用同一套切法,可算该切法的平均准确率。
        combo_correct: dict[int, list] = defaultdict(list)
        #〔讲解〕problem_breakdowns:每道题"对所有组合求平均后"的成本明细,最后再跨题求平均。
        problem_breakdowns = []

        #〔讲解〕逐题处理:problem_id 是题目标识,run_results 是这题的 N 条轨迹。
        for problem_id, run_results in results.items():
            #〔讲解〕防御:这题轨迹数不等于 N(跑挂/缺失)就跳过,保证所有题都按统一 N 比较。
            if len(run_results) != n:
                continue

            #〔讲解〕total_breakdown:这道题"所有组合"的成本累加器,从空成本(0,0,0)起步。
            total_breakdown = CostBreakdown()

            #〔讲解〕枚举这题的每个 C(N,k) 组合;combo 是被选中的 k 个轨迹下标(如 (0,2,3)),combo_idx 是组合序号。
            for combo_idx, combo in enumerate(combinations(range(n), k)):
                #〔讲解〕按下标把这 k 条轨迹取出来,组成本次组合的候选集。
                combo_runs = [run_results[i] for i in combo]
                #〔讲解〕本策略的"灵魂行":在 k 条里挑工具调用次数最少(min + get_tool_count)的那条;
                #〔讲解〕select_with_tie_breaking 在并列最少时用固定随机种子打破平局,保证可复现。
                #〔讲解〕对比 BON 策略这里是 (extract_confidence, max)——唯一差别就在这个选择键。
                min_run = self.select_with_tie_breaking(combo_runs, self.get_tool_count, min)
                #〔讲解〕把"被选中那条轨迹对不对"记到该组合编号下;is_correct 二分类返 True/False,评分类任务返 float。
                combo_correct[combo_idx].append(self.is_correct(min_run))
                #〔讲解〕成本按「整组 k 条都算」累加(不是只算被选中那条)——衡量为了产出这个答案花的总钱。
                total_breakdown = total_breakdown + self.calculate_combination_cost_breakdown(combo_runs)

            #〔讲解〕这道题:把所有组合的总成本 ÷ 组合数 = 该题的平均单组合成本,存起来。
            problem_breakdowns.append(total_breakdown / num_combinations)

        #〔讲解〕对每个组合编号:在所有题上的对错列表求均值 = 该切法的准确率。
        combo_accuracies = [sum(v) / len(v) for v in combo_correct.values()]
        #〔讲解〕再把所有组合切法的准确率求平均 = FewTool@k 最终准确率(没数据则 0)。
        metric = sum(combo_accuracies) / len(combo_accuracies) if combo_accuracies else 0.0
        #〔讲解〕把每个组合的准确率列表缓存到实例,供基类 _format_std() 算跨组合标准差(报告里的 ± X%)。
        self._last_problem_scores = combo_accuracies

        #〔讲解〕下面把各题的平均成本明细再跨题求平均,得到全数据集层面的平均成本明细。
        avg_breakdown = CostBreakdown()
        #〔讲解〕先逐题累加(CostBreakdown 重载了 +,按 rollout/tool/aggregation 三分量分别相加)。
        for b in problem_breakdowns:
            avg_breakdown = avg_breakdown + b
        #〔讲解〕有题目才除以题数(否则除零);除法也是三分量各自 ÷ 题数。
        if problem_breakdowns:
            avg_breakdown = avg_breakdown / len(problem_breakdowns)
        #〔讲解〕缓存成本明细到实例,供外部按 rollout/tool 分项查看。
        self._last_cost_breakdown = avg_breakdown
        #〔讲解〕返回 (准确率, 平均总成本);.total = rollout+tool+aggregation 三项之和(本策略无 aggregation 项)。
        return (metric, avg_breakdown.total)
