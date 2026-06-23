#〔讲解〕本文件实现「多数投票(MV)」和「加权多数投票(WMV)」两种聚合 baseline 策略。
#〔讲解〕属于 aggregation/ 环节:rollout 阶段对每道题跑了 N 条独立轨迹(各自给出一个答案),
#〔讲解〕本文件把这 N 条轨迹的答案做投票,选出"票数最多"的那条作为最终答案,再判分。
#〔讲解〕它是 aggagent(会调工具读轨迹的聪明聚合体)的对照组——纯靠投票、不读轨迹内容。
#〔讲解〕上游 = rollout 跑出的多条 run(dict);下游 = 评测脚本读取本类算出的 metric/cost 做对比。
"""Majority Voting strategies for aggregation evaluation."""

import math
#〔讲解〕combinations:从 N 条 run 里枚举所有 C(N,k) 个 k 元子集,用于算 MV@k(见下方)。
from itertools import combinations
#〔讲解〕defaultdict:取不存在的 key 时自动建默认值(这里 float 默认 0.0 / list 默认 []),省去手动判空。
from collections import defaultdict
#〔讲解〕Strategy = 所有策略基类(提供 extract_answer/is_correct 等公共方法);CostBreakdown = 成本核算容器。
from .base import Strategy, CostBreakdown


#〔讲解〕MVStrategy:多数投票策略类。核心思想——同一答案出现次数越多越可信,选票最高的答案对应的 run。
#〔讲解〕它继承 Strategy,复用基类的答案抽取、判分、随机 tie-break 等能力。
class MVStrategy(Strategy):
    """
    MV@k (Majority Voting) strategy.

    For each C(N,k) combination, select the run whose answer has the most votes,
    then check its correctness.
    """

    #〔讲解〕name 是策略标识符,评测结果表里用它区分是 MV 还是 WMV(子类会覆盖成 "WMV")。
    name = "MV"

    #〔讲解〕majority_vote:本类的"投票核心"。入参 runs=一组候选 run;weighted=是否按置信度加权。
    #〔讲解〕返回:得票最高答案所对应的那条 run(dict);若全员都抽不出答案则走 fallback,理论不会返回 None。
    def majority_vote(self, runs: list[dict], weighted: bool = False) -> dict | None:
        """
        Return the run with the answer that has the highest vote count.

        If weighted=True, each vote is weighted by the run's confidence.
        Ties are broken randomly via self.rng.
        """
        #〔讲解〕votes:答案文本 -> 累计票数(加权时是浮点);answer_to_runs:答案文本 -> 投了这个答案的 run 列表。
        votes: dict[str, float] = defaultdict(float)
        answer_to_runs: dict[str, list] = defaultdict(list)

        #〔讲解〕遍历每条 run,从中抽出它的最终答案并计票。
        for r in runs:
            #〔讲解〕extract_answer:基类方法,从一条 run 里解析出归一化后的答案字符串;抽不到则为 None。
            answer = Strategy.extract_answer(r)
            #〔讲解〕只对成功抽到答案的 run 计票(抽不到的视为弃权)。
            if answer is not None:
                #〔讲解〕普通 MV 每票权重恒为 1.0;WMV 时权重=该 run 的置信度(extract_confidence)。
                weight = Strategy.extract_confidence(r) if weighted else 1.0
                #〔讲解〕给该答案累加票权,并把这条 run 归入对应答案的桶里。
                votes[answer] += weight
                answer_to_runs[answer].append(r)

        #〔讲解〕兜底:如果整组 run 一个答案都抽不出(votes 为空),无法投票,改用基类的选择逻辑。
        if not votes:
            # No extractable answers: random (MV) or highest-confidence (WMV) fallback
            #〔讲解〕WMV 兜底:在所有 run 里挑置信度最高的那条(key=置信度, select=max)。
            if weighted:
                return self.select_with_tie_breaking(runs, Strategy.extract_confidence, max)
            #〔讲解〕普通 MV 兜底:无 key 时基类默认随机选一条(用 self.rng 保证可复现)。
            return self.select_with_tie_breaking(runs)

        #〔讲解〕正常路径:在所有候选答案里,用票数(votes[ans])作为打分,选票数最大的答案。
        #〔讲解〕select_with_tie_breaking 在并列最高时随机打破平局,避免总是偏向某个固定答案。
        best_answer = self.select_with_tie_breaking(
            list(votes.keys()),
            key_func=lambda ans: votes[ans],
            select_func=max,
        )
        #〔讲解〕得到胜出答案后,该答案可能对应多条 run,再从这些 run 里随机选一条返回(它们答案相同,选谁判分都一样)。
        return self.select_with_tie_breaking(answer_to_runs[best_answer])

    #〔讲解〕calculate_at_k:计算 MV@k 指标。给定所有题目的 run 结果和子集大小 k,
    #〔讲解〕对每道题枚举所有 C(N,k) 子集做投票判分,返回 (平均准确率 metric, 平均成本 total)。
    def calculate_at_k(
        self, results: dict[str, list[dict]], k: int
    ) -> tuple[float, float]:
        #〔讲解〕没有任何结果时直接返回零,避免后续除零。
        if not results:
            return (0.0, 0.0)

        #〔讲解〕get_n:基类方法,推断每道题统一的 run 总数 N(rollout 跑了多少条)。
        n = self.get_n(results)
        #〔讲解〕k 不能超过 N(子集不可能比全集大),否则参数非法直接报错。
        if k > n:
            raise ValueError(f"k ({k}) cannot be greater than N ({n})")

        #〔讲解〕math.comb(n,k)=C(N,k),即每道题要枚举的子集总数,用于后面做成本平均的分母。
        num_combinations = math.comb(n, k)
        #〔讲解〕combo_correct:子集编号 combo_idx -> 该子集在各题上的对错列表(1/0)。
        #〔讲解〕按"子集编号"而非"题目"聚合,是为了让每个固定子集组合在所有题上的准确率可单独统计。
        combo_correct: dict[int, list] = defaultdict(list)
        #〔讲解〕problem_breakdowns:每道题的平均成本(CostBreakdown),最后再跨题平均。
        problem_breakdowns = []

        #〔讲解〕逐题处理:problem_id=题目标识,run_results=该题的 N 条 run。
        for problem_id, run_results in results.items():
            #〔讲解〕数据完整性校验:run 数不等于 N 的题(可能跑挂/缺数据)直接跳过,保证组合枚举对齐。
            if len(run_results) != n:
                continue

            #〔讲解〕本题的成本累加器,初始为空(0 成本)。
            total_breakdown = CostBreakdown()

            #〔讲解〕枚举本题所有 k 元子集;combo 是 N 个 run 里被选中的 k 个下标元组,combo_idx 是其序号。
            for combo_idx, combo in enumerate(combinations(range(n), k)):
                #〔讲解〕按下标取出本子集对应的 k 条 run。
                combo_runs = [run_results[i] for i in combo]
                #〔讲解〕在这 k 条 run 上做普通(非加权)多数投票,选出代表答案的 run。
                selected = self.majority_vote(combo_runs, weighted=False)
                #〔讲解〕判分:选中 run 正确记 1、错误记 0;万一 selected 为 None(理论不会)记 0。记入该子集编号。
                combo_correct[combo_idx].append(self.is_correct(selected) if selected is not None else 0)
                #〔讲解〕累加本子集的成本(无论投票选谁,k 条 run 都已实际跑过,成本算整个子集)。
                total_breakdown = total_breakdown + self.calculate_combination_cost_breakdown(combo_runs)

            #〔讲解〕本题成本 = 所有子集成本之和 / 子集个数,得到"每子集平均成本",再存起来跨题平均。
            problem_breakdowns.append(total_breakdown / num_combinations)

        #〔讲解〕对每个子集编号:它在所有题上的对错均值 = 该子集的准确率。
        combo_accuracies = [sum(v) / len(v) for v in combo_correct.values()]
        #〔讲解〕最终 metric = 所有子集准确率再平均(等价于 MV@k 的期望准确率);无数据则 0。
        metric = sum(combo_accuracies) / len(combo_accuracies) if combo_accuracies else 0.0
        #〔讲解〕把分项准确率挂到实例上,供外部(如画分布图/做统计)读取。
        self._last_problem_scores = combo_accuracies

        #〔讲解〕跨题汇总成本:把每道题的平均成本逐个累加。
        avg_breakdown = CostBreakdown()
        for b in problem_breakdowns:
            avg_breakdown = avg_breakdown + b
        #〔讲解〕再除以题数,得到全数据集的平均成本(有题才除,防除零)。
        if problem_breakdowns:
            avg_breakdown = avg_breakdown / len(problem_breakdowns)
        #〔讲解〕同样把成本明细挂到实例上供外部读取。
        self._last_cost_breakdown = avg_breakdown
        #〔讲解〕返回 (准确率, 总成本)。.total 是 CostBreakdown 汇总出的单一成本数值。
        return (metric, avg_breakdown.total)


#〔讲解〕WMVStrategy:加权多数投票。继承 MVStrategy,唯一区别——投票时每票按 run 的置信度加权。
#〔讲解〕直觉:模型越"有把握"的答案话语权越大,而非简单一人一票。
class WMVStrategy(MVStrategy):
    """
    WMV@k (Weighted Majority Voting) strategy.

    Same as MV but votes are weighted by each run's confidence score.
    """

    #〔讲解〕覆盖父类 name,使结果表里标记为 "WMV"。
    name = "WMV"

    #〔讲解〕calculate_at_k:与父类几乎逐行相同,唯一差异是下方投票调用传 weighted=True。
    #〔讲解〕(本可复用父类只切换 weighted 标志,这里选择整段重写,属于代码冗余但行为等价。)
    def calculate_at_k(
        self, results: dict[str, list[dict]], k: int
    ) -> tuple[float, float]:
        if not results:
            return (0.0, 0.0)

        n = self.get_n(results)
        if k > n:
            raise ValueError(f"k ({k}) cannot be greater than N ({n})")

        num_combinations = math.comb(n, k)
        combo_correct: dict[int, list] = defaultdict(list)
        problem_breakdowns = []

        for problem_id, run_results in results.items():
            if len(run_results) != n:
                continue

            total_breakdown = CostBreakdown()

            for combo_idx, combo in enumerate(combinations(range(n), k)):
                combo_runs = [run_results[i] for i in combo]
                #〔讲解〕这里 weighted=True 是 WMV 与 MV 的唯一实质区别:按置信度加权投票。
                selected = self.majority_vote(combo_runs, weighted=True)
                combo_correct[combo_idx].append(self.is_correct(selected) if selected is not None else 0)
                total_breakdown = total_breakdown + self.calculate_combination_cost_breakdown(combo_runs)

            problem_breakdowns.append(total_breakdown / num_combinations)

        combo_accuracies = [sum(v) / len(v) for v in combo_correct.values()]
        metric = sum(combo_accuracies) / len(combo_accuracies) if combo_accuracies else 0.0
        self._last_problem_scores = combo_accuracies

        avg_breakdown = CostBreakdown()
        for b in problem_breakdowns:
            avg_breakdown = avg_breakdown + b
        if problem_breakdowns:
            avg_breakdown = avg_breakdown / len(problem_breakdowns)
        self._last_cost_breakdown = avg_breakdown
        return (metric, avg_breakdown.total)
