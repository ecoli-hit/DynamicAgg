#〔讲解〕本文件 = AggAgent 框架「聚合评测」环节里的 AggAgent@k 策略类(harness 层)。
#〔讲解〕pipeline 位置:rollout/ 先跑出每道题的多条 ReAct 搜索轨迹 → 本文件把其中 k 条「轨迹组合」交给
#〔讲解〕   核心聚合智能体 aggagent.AggAgent(会调工具读轨迹) → 综合出一份答案 → 用 evaluation 打分。
#〔讲解〕上游:rollout 产出的 results(每题一串 run 记录);下游:_compute_score 判分 + 把日志/统计落盘。
#〔讲解〕本类只负责「组合采样 / 多线程调度 / 重试 / 日志 / 成本核算」,真正的聚合推理在 aggagent 包里。
"""AggAgent strategy for aggregation evaluation."""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
#〔讲解〕combinations:从 N 条轨迹里枚举所有「选 k 条」的组合(C(N,k)),用于 AggAgent@k 的采样。
from itertools import combinations

from tqdm import tqdm

#〔讲解〕从同目录 base 引入:Strategy=策略基类;MODEL_COSTS=各模型输入/输出单价表;
#〔讲解〕   CostBreakdown=成本三段(rollout/tool/aggregation)结构;_compute_score=调判分模型给预测打分。
from .base import Strategy, MODEL_COSTS, CostBreakdown, _compute_score
#〔讲解〕真正的聚合智能体本体,改名 _AggAgent 以免和本文件的策略类 AggAgent 重名(两者同名易混)。
from aggagent import AggAgent as _AggAgent


#〔讲解〕把「每个轨迹组合跑出来的 agent 运行统计」汇总成一份 dict,和日志一起落盘。
#〔讲解〕入参:all_stats=各次运行的统计列表;k=本次组合规模;metric=该 k 下的总得分;metadata=附带元信息。
#〔讲解〕返回:含迭代次数/报错数/工具调用计数等的聚合 dict(纯统计,不影响主流程结果)。
def _aggregate_stats(all_stats: list[dict], k: int, metric: float, metadata: dict | None = None) -> dict:
    """Summarize per-combo agent stats into a single dict saved alongside logs."""
    total = len(all_stats)
    #〔讲解〕下面三行用「列表推导 + .get(默认0)」从每条 stats 抽出关键指标,缺字段时给 0 防 KeyError。
    iterations_list = [s.get("iterations", 0) for s in all_stats]
    server_errors_list = [s.get("server_errors", 0) for s in all_stats]
    tool_call_errors_list = [s.get("tool_call_errors", 0) for s in all_stats]

    aggregated = {
        "metadata": metadata or {},
        "k": k,
        "metric_at_k": metric,
        "total_runs": total,
        "iterations": {
            "total": sum(iterations_list),
            #〔讲解〕avg/min/max 都先判 total/列表是否为空,空则给 0,避免除零或对空序列取 min/max 报错。
            "avg": sum(iterations_list) / total if total else 0,
            "min": min(iterations_list) if iterations_list else 0,
            "max": max(iterations_list) if iterations_list else 0,
        },
        "server_errors": {
            "total": sum(server_errors_list),
            "avg": sum(server_errors_list) / total if total else 0,
        },
        "tool_call_errors": {
            "total": sum(tool_call_errors_list),
            "avg": sum(tool_call_errors_list) / total if total else 0,
        },
        #〔讲解〕统计有多少次运行「撞到上下文长度上限」(context_limit_reached=True),反映轨迹是否过长。
        "context_limit_reached_count": sum(1 for s in all_stats if s.get("context_limit_reached", False)),
        "tool_calls": {},
    }
    #〔讲解〕把每次运行里「各工具被调用的次数」按工具名累加到一个总表,得到全局工具调用分布。
    for s in all_stats:
        for tool_name, count in s.get("tool_calls", {}).items():
            aggregated["tool_calls"][tool_name] = aggregated["tool_calls"].get(tool_name, 0) + count
    return aggregated


#〔讲解〕AggAgent@k 策略主类:对每题采样若干 k-轨迹组合,逐组合调聚合智能体合成答案再判分,
#〔讲解〕   最终给出该 k 下的平均得分与平均成本。继承 base.Strategy(共用 is_correct/get_n/rng/成本核算等)。
class AggAgent(Strategy):
    """
    AggAgent@k (Agent-based Aggregation) strategy.

    Runs an AggregatorAgent over a sampled combination of k trajectories to
    synthesize a final answer, then checks correctness.
    """

    #〔讲解〕策略名,用于写进日志 metadata 和命名输出文件(aggagent_logs_k*.jsonl 等)。
    name = "AggAgent"

    #〔讲解〕构造函数:配置聚合用的模型/接口、并发数、输出目录,以及 resume(断点续跑)/skip_score(只跑不判分)开关。
    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        task: str = "browsecomp",
        #〔讲解〕max_workers=线程池并发题数,默认 10,决定同时跑多少道题的聚合。
        max_workers: int = 10,
        output_dir: str | None = None,
        #〔讲解〕resume=True 时会读已有日志跳过做过的题;skip_score=True 时只产预测不调判分模型(省判分成本)。
        resume: bool = False,
        skip_score: bool = False,
        **kwargs,
    ):
        #〔讲解〕调父类构造,把 task(browsecomp/healthbench 等)传上去,父类据此初始化 rng、判分口径等。
        super().__init__(task=task)
        self.model = model
        self.api_base = api_base or ""
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.resume = resume
        self.skip_score = skip_score

        #〔讲解〕日志缓冲区 + 线程锁:多线程并发写日志,先暂存内存再批量 flush,锁保证写入不被并发破坏。
        self._log_entries: list[dict] = []
        self._log_lock = threading.Lock()
        self._current_log_file: str | None = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    #〔讲解〕线程安全地把一条日志记录追加进内存缓冲(还没落盘),由 _flush_logs 统一写文件。
    def _write_log(self, entry: dict):
        with self._log_lock:
            self._log_entries.append(entry)

    #〔讲解〕把内存缓冲的日志一次性追加写入 jsonl 文件后清空缓冲;没配输出目录或缓冲为空则直接返回。
    def _flush_logs(self):
        if not self.output_dir or not self._current_log_file:
            return
        with self._log_lock:
            if not self._log_entries:
                return
            #〔讲解〕以追加模式("a")写文件:每条记录一行 JSON(jsonl 格式),ensure_ascii=False 保留中文原文。
            with open(self._current_log_file, "a") as f:
                for entry in self._log_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._log_entries.clear()

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    #〔讲解〕根据聚合智能体逐步上报的 token 用量,按模型单价折算出本次聚合的 LLM 美元成本。
    #〔讲解〕入参 stats 里含 token_usage_each_step(每步累计的 input/output token);返回:美元成本(float)。
    def _calculate_agg_cost(self, stats: dict) -> float:
        """Calculate aggregation LLM cost from agent token_usage_each_step."""
        #〔讲解〕模型不在价目表里就算不出价钱,直接返回 0,避免用错价。
        if self.model not in MODEL_COSTS:
            return 0.0
        input_rate, output_rate = MODEL_COSTS[self.model]
        total_input_delta = 0
        total_output = 0
        #〔讲解〕prev_input 记录上一步的累计 input token,用「本步累计 − 上步累计」得到这一步新增的 input。
        prev_input = 0
        for step in stats.get("token_usage_each_step", []):
            curr_input = step.get("input_tokens")
            curr_output = step.get("output_tokens")
            if curr_input is not None:
                #〔讲解〕关键:input_tokens 是「累计值」,故取差分(并 max(...,0) 防止偶发回退成负)累加成真正消耗。
                total_input_delta += max(curr_input - prev_input, 0)
                prev_input = curr_input
            #〔讲解〕output_tokens 视为每步单独产出,直接累加(与 input 的累计差分处理不同,易混点)。
            if curr_output is not None:
                total_output += curr_output
        #〔讲解〕单价单位是「每百万 token」,故 token×单价后除以 1e6 得美元;输入/输出分别计价再相加。
        return (total_input_delta * input_rate + total_output * output_rate) / 1_000_000

    # ------------------------------------------------------------------
    # Per-combo processing
    # ------------------------------------------------------------------

    #〔讲解〕处理「单个 k-轨迹组合」:实例化聚合智能体跑出答案 → 带重试 → 判分 → 写日志。
    #〔讲解〕入参:question=题面;item/instance=题目元数据;combo_runs=被选中的那 k 条轨迹;combo=它们的下标元组;k=组合规模。
    #〔讲解〕返回三元组:(is_correct 即 0/1 得分, agent_stats 运行统计, llm_cost 本组合聚合成本)。
    def _process_single_combo(
        self,
        question: str,
        item: dict,
        instance: dict,
        combo_runs: list[dict],
        combo: tuple,
        k: int,
    ) -> tuple[float, dict, float]:
        """Process one combo. Returns (is_correct, agent_stats, llm_cost)."""
        #〔讲解〕先记录这 k 条原始轨迹各自对不对、各自的判分结果,便于事后对比「聚合后 vs 聚合前」。
        combo_correct = [self.is_correct(r) for r in combo_runs]
        combo_judgements = [r.get("auto_judge") for r in combo_runs]

        #〔讲解〕实例化核心聚合智能体(每个组合一个独立实例,避免线程间状态串扰)。
        agent = _AggAgent(
            model=self.model,
            api_base=self.api_base,
            task=self.task,
        )

        #〔讲解〕重试机制:聚合可能因网络/模型抽风失败或返回畸形结果,最多试 3 次,失败用指数退避。
        max_retries = 3
        last_error = None
        agent_result = None
        agent_messages = None
        agent_stats = None
        elapsed = None
        attempts_made = 0

        for attempt in range(max_retries):
            attempts_made = attempt + 1
            try:
                t_start = time.time()
                #〔讲解〕核心调用:agent._run(题面, k条轨迹) 让聚合智能体调工具读轨迹并合成答案。
                agent_output = agent._run(question, combo_runs)
                elapsed = time.time() - t_start
                agent_result = agent_output["result"]
                agent_messages = agent_output["messages"]
                agent_stats = agent_output.get("stats", {})

                #〔讲解〕成功判定:结果必须是 dict 且含 "solution" 字段,满足才跳出重试循环。
                if agent_result is not None and isinstance(agent_result, dict) and "solution" in agent_result:
                    break
                else:
                    #〔讲解〕结果畸形(None 或缺 solution):记下原因,若还有重试机会则退避后再试。
                    last_error = f"Invalid agent_result: {'None' if agent_result is None else 'missing solution field'}"
                    if attempt < max_retries - 1:
                        #〔讲解〕2**attempt = 指数退避:第1次失败等1秒、第2次等2秒,缓解瞬时过载。
                        time.sleep(2 ** attempt)
            except Exception as e:
                #〔讲解〕抛异常也按重试处理(不让单个组合崩掉整轮评测),记录异常文本待写日志。
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        #〔讲解〕只要拿到 stats 就算成本(即便最终结果畸形,这部分 token 也已真实消耗,需计入)。
        combo_llm_cost = self._calculate_agg_cost(agent_stats) if agent_stats else 0.0

        #〔讲解〕重试用尽仍是畸形结果:记一条「失败日志」(预测为空、is_correct=False)并提前返回 0 分。
        if agent_result is None or not isinstance(agent_result, dict) or "solution" not in agent_result:
            log_entry = {
                "metadata": self._current_metadata,
                "auto_judge": None,
                "question": question,
                "instance": instance,
                "prediction": None,
                "is_correct": False,
                "aggregation_cost": combo_llm_cost,
                "aggregation_time": elapsed,
                "aggagent_result": agent_result,
                "aggagent_stats": agent_stats,
                "aggagent_messages": agent_messages,
                "combo": list(combo),
                "combo_correct": combo_correct,
                "combo_judgements": combo_judgements,
                "error": last_error or "Invalid agent result",
                "retries": max_retries,
            }
            self._write_log(log_entry)
            #〔讲解〕失败也回传 stats(可能为 None→给空 dict)和已花成本,让上层正确累计成本/统计。
            return (0.0, agent_stats or {}, combo_llm_cost)

        #〔讲解〕走到这里说明结果合法,取出聚合给出的最终答案文本作为本组合的预测。
        prediction = agent_result["solution"]

        #〔讲解〕skip_score 模式:只产预测不判分(省判分模型调用),分数置 None 待后续单独评。
        if self.skip_score:
            judgement = None
            is_correct = None
        else:
            #〔讲解〕正常模式:调判分函数对预测打分,再用基类 is_correct 把判分结果归一成对/错。
            judgement = _compute_score(prediction, item, self.task)
            is_correct = self.is_correct({"auto_judge": judgement})

        log_entry = {
            "metadata": self._current_metadata,
            "auto_judge": judgement,
            "question": question,
            "instance": instance,
            "prediction": prediction,
            "is_correct": is_correct,
            "aggregation_cost": combo_llm_cost,
            "aggregation_time": elapsed,
            "aggagent_result": agent_result,
            "aggagent_stats": agent_stats,
            "aggagent_messages": agent_messages,
            "combo": list(combo),
            "combo_correct": combo_correct,
            "combo_judgements": combo_judgements,
        }
        #〔讲解〕只有发生过重试(>1 次)才在日志里额外记 attempts,方便事后排查不稳定的组合。
        if attempts_made > 1:
            log_entry["attempts"] = attempts_made
        self._write_log(log_entry)

        #〔讲解〕返回得分(skip_score 时 is_correct 为 None,折算成 0.0 不计正确)、统计、成本。
        return (float(is_correct) if is_correct is not None else 0.0, agent_stats, combo_llm_cost)

    #〔讲解〕处理「单道题」:对该题所有采样到的组合各跑一遍 _process_single_combo,再求平均得分与平均成本。
    #〔讲解〕入参:problem_id=题号;run_results=该题全部轨迹;sampled_combos=已采样的组合列表;k=组合规模。
    #〔讲解〕返回:(score 平均得分, avg_agg_cost 平均聚合成本, stats_list 各组合统计列表)。
    def _process_single_problem(
        self,
        problem_id: str,
        run_results: list[dict],
        sampled_combos: list[tuple],
        k: int,
    ) -> tuple[float, float, list[dict]]:
        """Process all sampled combos for one problem. Returns (score, avg_agg_cost, stats_list)."""
        #〔讲解〕题面/实例信息从第 0 条轨迹取(同题的各条轨迹共享同一题面与 instance)。
        question = run_results[0].get("question", "")
        instance = run_results[0].get("instance", {})
        # item = {"question": question, **instance}
        #〔讲解〕item 把题面和首条轨迹全部字段合并,作为判分函数所需的「题目+参考」上下文。
        item = {"question": question, **instance}

        #〔讲解〕healthbench 特例:题面是多轮对话,需把 prompt 里每条 {role,content} 拼成多行字符串当 question。
        if self.task == "healthbench":
            question = "\n".join(
                str({"role": r["role"], "content": r["content"]})
                for r in instance.get("prompt", [])
            )

        correct_sum = 0.0
        total_agg_cost = 0.0
        stats_list = []

        #〔讲解〕遍历本题的每个采样组合:combo 是下标元组,据此从 run_results 取出对应的 k 条轨迹。
        for combo in sampled_combos:
            combo_runs = [run_results[i] for i in combo]
            is_correct, stats, llm_cost = self._process_single_combo(
                question, item, instance, combo_runs, combo, k
            )
            correct_sum += is_correct
            total_agg_cost += llm_cost
            stats_list.append(stats)

        #〔讲解〕该题得分 = 各组合得分均值(即 AggAgent@k 在这题上的期望正确率);组合为空则记 0,防除零。
        score = correct_sum / len(sampled_combos) if sampled_combos else 0.0
        avg_agg_cost = total_agg_cost / len(sampled_combos) if sampled_combos else 0.0
        return (score, avg_agg_cost, stats_list)

    # ------------------------------------------------------------------
    # calculate_at_k
    # ------------------------------------------------------------------

    #〔讲解〕本类对外主入口:给定所有题的轨迹 results 和组合规模 k,跑完整 AggAgent@k 评测。
    #〔讲解〕返回:(metric 全题平均得分, 平均总成本)。内部负责组合采样、断点续跑、多线程调度、落盘。
    def calculate_at_k(
        self, results: dict[str, list[dict]], k: int
    ) -> tuple[float, float]:
        if not results:
            return (0.0, 0.0)

        #〔讲解〕n=每题的轨迹条数(N);k 不能大于 N,否则没法从 N 条里选 k 条,直接报错。
        n = self.get_n(results)
        if k > n:
            raise ValueError(f"k ({k}) cannot be greater than N ({n})")

        #〔讲解〕枚举 C(N,k) 全部组合;为控成本最多只评 3 个组合,超过就用带种子的 rng 随机抽 3 个(保证可复现)。
        all_combos = list(combinations(range(n), k))
        max_combos = 3
        sampled_combos = self.rng.sample(all_combos, max_combos) if len(all_combos) > max_combos else all_combos

        #〔讲解〕只保留轨迹数恰好为 N 的题(轨迹不全的题组合下标会越界,直接剔除)。
        valid_problems = [(pid, rr) for pid, rr in results.items() if len(rr) == n]

        #〔讲解〕从任一题首条轨迹取 rollout 阶段的 metadata,叠加本策略信息,作为写进每条日志的统一元信息。
        rollout_metadata = next(iter(results.values()))[0].get("metadata", {})
        self._current_metadata = {
            **rollout_metadata,
            "strategy": self.name,
            "k": k,
            "aggregation_model": self.model,
        }

        # Setup log file; load existing entries when resuming
        #〔讲解〕断点续跑准备:existing 用「题面→已有日志条目」索引,resume 时据此跳过做过的题。
        existing: dict[str, list[dict]] = {}
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            #〔讲解〕日志文件名带 k(如 aggagent_logs_k4.jsonl),不同 k 各自一份互不覆盖。
            self._current_log_file = os.path.join(self.output_dir, f"aggagent_logs_k{k}.jsonl")
            if self.resume and os.path.exists(self._current_log_file):
                #〔讲解〕续跑:逐行读旧 jsonl,按 question 归类进 existing(同题多组合会有多条)。
                with open(self._current_log_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        existing.setdefault(entry["question"], []).append(entry)
            #〔讲解〕非续跑或无旧记录:以写模式打开再立刻关闭,等于清空/新建该日志文件。
            if not existing:
                open(self._current_log_file, "w").close()

        problem_scores: list[float] = []
        problem_breakdowns: list[CostBreakdown] = []
        all_stats: list[dict] = []

        # Recover from existing log entries
        #〔讲解〕第一步:对「已在旧日志里」的题直接用历史结果恢复分数/成本,不重跑;其余进 new_valid_problems 待算。
        new_valid_problems = []
        for pid, rr in valid_problems:
            question = rr[0].get("question", "") if rr else ""
            if question in existing:
                entries = existing[question]
                #〔讲解〕该题历史得分 = 各组合 is_correct 均值(None 用 0 兜底)。
                combo_scores = [float(e.get("is_correct") or 0) for e in entries]
                problem_scores.append(sum(combo_scores) / len(combo_scores))
                avg_cost = sum(e.get("aggregation_cost", 0.0) for e in entries) / len(entries)
                #〔讲解〕轨迹/工具成本无法从日志直接拿,需按当前采样组合重新核算 rollout+tool 成本(聚合成本则用历史 avg_cost)。
                traj_breakdown = CostBreakdown()
                for combo in sampled_combos:
                    combo_runs = [rr[i] for i in combo]
                    #〔讲解〕CostBreakdown 重载了 + 号,可逐组合累加成本三段;此处累加所有组合的轨迹成本。
                    traj_breakdown = traj_breakdown + self.calculate_combination_cost_breakdown(combo_runs)
                #〔讲解〕重载了 / 号:除以组合数得平均,得到该题每组合平均的 rollout/tool 成本。
                traj_breakdown = traj_breakdown / len(sampled_combos)
                problem_breakdowns.append(CostBreakdown(
                    rollout=traj_breakdown.rollout,
                    tool=traj_breakdown.tool,
                    #〔讲解〕聚合段用历史日志里的平均聚合成本,拼成该题完整三段成本。
                    aggregation=avg_cost,
                ))
                all_stats.extend(e["aggagent_stats"] for e in entries if e.get("aggagent_stats"))
            else:
                new_valid_problems.append((pid, rr))

        if existing:
            print(f"Resuming: {len(existing)} already done, {len(new_valid_problems)} remaining")

        #〔讲解〕第二步:对剩余未做的题用线程池并发跑,每题一个 future(并发上限 = max_workers)。
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            #〔讲解〕submit 把任务排进线程池,futures 字典记下 future→(题号,轨迹) 便于完成时回查是哪题。
            futures = {
                executor.submit(self._process_single_problem, pid, rr, sampled_combos, k): (pid, rr)
                for pid, rr in new_valid_problems
            }
            #〔讲解〕as_completed:谁先跑完先返回(非提交顺序);tqdm 包一层显示进度条。
            pbar = tqdm(as_completed(futures), total=len(futures), desc=f"AggAgent@{k}")
            for future in pbar:
                pid, rr = futures[future]
                #〔讲解〕future.result() 取该题计算结果;若线程内抛异常,这里会重新抛出。
                score, agg_cost, stats_list = future.result()
                problem_scores.append(score)
                all_stats.extend(stats_list)

                #〔讲解〕和续跑分支同理:重算该题 rollout/tool 成本,再拼上本次实跑的聚合成本 agg_cost。
                traj_breakdown = CostBreakdown()
                for combo in sampled_combos:
                    combo_runs = [rr[i] for i in combo]
                    traj_breakdown = traj_breakdown + self.calculate_combination_cost_breakdown(combo_runs)
                traj_breakdown = traj_breakdown / len(sampled_combos)
                problem_breakdowns.append(CostBreakdown(
                    rollout=traj_breakdown.rollout,
                    tool=traj_breakdown.tool,
                    aggregation=agg_cost,
                ))

                #〔讲解〕每完成 15 题落一次盘:既防中途崩溃丢日志,又不必每题都开文件(平衡 IO 与安全)。
                if len(problem_scores) % 15 == 0:
                    self._flush_logs()

                #〔讲解〕进度条后缀实时显示当前累计平均正确率(%),方便跑的时候肉眼看趋势。
                if problem_scores:
                    pbar.set_postfix_str(f"{sum(problem_scores)/len(problem_scores)*100:.2f}%")

        #〔讲解〕循环结束再 flush 一次,把最后不足 15 题的尾巴日志全部落盘。
        self._flush_logs()

        #〔讲解〕最终指标 = 全部题得分的平均(即 AggAgent@k 在整个数据集上的正确率)。
        metric = sum(problem_scores) / len(problem_scores) if problem_scores else 0.0
        #〔讲解〕缓存逐题分数,供外部(如算方差/置信区间)取用。
        self._last_problem_scores = problem_scores

        # Write aggregated stats file
        #〔讲解〕把所有运行统计汇总成一份 stats json 落盘(文件名同样带 k),供事后分析工具调用/报错分布。
        if self.output_dir and all_stats:
            aggregated_stats = _aggregate_stats(all_stats, k, metric, metadata=self._current_metadata)
            stats_file = os.path.join(self.output_dir, f"aggagent_stats_k{k}.json")
            with open(stats_file, "w") as f:
                json.dump(aggregated_stats, f, indent=2)

        #〔讲解〕把各题成本三段逐题相加再除以题数,得到全数据集「每题平均」的成本拆解。
        avg_breakdown = CostBreakdown()
        for b in problem_breakdowns:
            avg_breakdown = avg_breakdown + b
        if problem_breakdowns:
            avg_breakdown = avg_breakdown / len(problem_breakdowns)
        #〔讲解〕缓存平均成本拆解(rollout/tool/aggregation 三段),供外部出成本报告。
        self._last_cost_breakdown = avg_breakdown

        #〔讲解〕返回 (平均得分, 平均总成本);.total 是 CostBreakdown 里三段相加的属性。
        return (metric, avg_breakdown.total)
