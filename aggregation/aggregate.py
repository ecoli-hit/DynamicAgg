# 〔讲解〕 模块说明:本文件是「聚合阶段」的命令行总入口(CLI driver)。
# 〔讲解〕 在 AggAgent 两阶段 MoA 流程里属于第二环:rollout/ 先跑出 N 条 ReAct 搜索轨迹(每条存成一个目录的若干 .json),
# 〔讲解〕 本脚本把这些轨迹按「同一道题」分组,再选一种或多种聚合策略(_strategy 包里的 mv/bon/aggagent/...)算出指标。
# 〔讲解〕 上游=rollout 产出的结果目录;下游=_strategy 里各策略的 run() 以及它们内部调用的 evaluation 判分。
# 〔讲解〕 本文件自己只做「找目录→读 json→分组→分发策略→打印」,真正的聚合与打分逻辑都在 _strategy 包内。
from pathlib import Path
import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm

# 〔讲解〕 从 _strategy 包导入聚合策略相关的核心符号(该包是 aggregation/_strategy/ 目录)。
# 〔讲解〕 Strategy=所有策略的抽象基类;MetricResult=单个 k 值的指标结果(value + 可选 cost)。
# 〔讲解〕 get_strategy/get_heuristic_strategies=工厂函数,按名字造策略实例;
# 〔讲解〕 STRATEGIES=全部策略名→类的字典;HEURISTIC_STRATEGIES=其中「不调用 LLM 的启发式策略」子集。
from _strategy import (
    Strategy,
    MetricResult,
    get_strategy,
    get_heuristic_strategies,
    STRATEGIES,
    HEURISTIC_STRATEGIES,
)


# 〔讲解〕 递归找出所有「叶子目录」(没有子目录的目录)。
# 〔讲解〕 角色:rollout 结果常按 任务/模型/批次 多层嵌套,真正装 .json 的是最底层目录;
# 〔讲解〕 每个叶子目录 = 一次完整 rollout(对全部题各跑一遍),所以叶子目录数 = 后面的 N(rollout 次数)。
def find_leaf_directories(root: Path) -> list[Path]:
    """
    Find all leaf directories (directories with no subdirectories).
    """
    leaves = []
    # 〔讲解〕 os.scandir 比 listdir 快,因为它顺带返回 is_dir() 等元数据,无需再 stat 一次。
    for entry in os.scandir(root):
        if entry.is_dir():
            # 〔讲解〕 看这个子目录下还有没有更深的子目录:有→它不是叶子,继续往下递归;没有→它就是叶子。
            subdirs = [e for e in os.scandir(entry.path) if e.is_dir()]
            if subdirs:
                # 〔讲解〕 递归把更深层的叶子全部收集进来(extend 而非 append,摊平成一维列表)。
                leaves.extend(find_leaf_directories(Path(entry.path)))
            else:
                leaves.append(Path(entry.path))
    return leaves


# 〔讲解〕 把 N 个叶子目录里的 json 结果读进内存,并按「题目」重新分组。
# 〔讲解〕 关键返回:dict{problem_id -> [N 个结果 dict]},以及 N(目录数)。
# 〔讲解〕 角色:这是聚合的输入准备——同一道题在 N 次 rollout 里有 N 个候选答案/轨迹,聚合就是从这 N 个里挑/合成一个。
def load_results(directories: list[Path]) -> tuple[dict[str, list[dict]], int]:
    """
    Load results from N directories.

    Returns:
        Tuple of (dict mapping problem_id to list of N result dicts, N)
    """
    # 〔讲解〕 defaultdict(list):访问不存在的 key 自动给个空列表,省去手动判空,方便后面直接 append。
    results = defaultdict(list)
    n = len(directories)

    # 〔讲解〕 tqdm 包一层即可显示读取进度条;每个 dir_path 是一次 rollout 的结果目录。
    for dir_path in tqdm(directories):
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {dir_path}")

        # 〔讲解〕 只挑该目录下的 .json 文件(每个 json 通常是一道题在这次 rollout 的完整记录)。
        json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
        # 〔讲解〕 dir_results 先在「单个目录内」按 problem_id 去重,防止同一目录里同题出现多份时重复计入。
        dir_results = {}

        for filename in json_files:
            filepath = os.path.join(dir_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

            # 〔讲解〕 题目唯一标识的取值优先级:先用题面文本 question,没有就用 instance_id,再没有就退化用文件名兜底。
            # 〔讲解〕 (不同 benchmark 字段命名不同,这里做了兼容)。
            problem_id = data.get("question") or data.get("instance_id") or filename
            dir_results[problem_id] = data

        # 〔讲解〕 把本目录(本次 rollout)每道题的结果,追加到全局 results 对应题目的列表里。
        # 〔讲解〕 循环完所有目录后,理想情况下每道题恰好积累到 N 个结果(来自 N 次 rollout)。
        for problem_id, data in dir_results.items():
            results[problem_id].append(data)

    # Verify all problems have N results
    # 〔讲解〕 一致性检查:若某题结果数 ≠ N,说明某次 rollout 漏跑/缺文件,只告警不报错(便于带残缺数据继续算)。
    for problem_id, res_list in results.items():
        if len(res_list) != n:
            print(f"Warning: Problem {problem_id} has {len(res_list)} results, expected {n}")

    return results, n


# 〔讲解〕 LLM 类策略的名字 = 全部策略名里「不属于启发式」的那些(即需要调大模型的:solagg/summagg/aggagent)。
# 〔讲解〕 用集合差集思路一行筛出,后面据此判断「哪些策略必须传 --model、要建输出目录」。
_LLM_STRATEGY_NAMES = [k for k in STRATEGIES if k not in HEURISTIC_STRATEGIES]
# 〔讲解〕 纯打印用的分隔横线(两空格缩进 + 44 个横线),让多策略的控制台输出更易读。
_SEPARATOR = "  " + "─" * 44


# 〔讲解〕 跑「单个」策略:把分组好的 results、N、k 列表交给该策略的 run(),返回 {k -> MetricResult}。
# 〔讲解〕 这里只是薄薄一层转发(thin wrapper),真正逻辑在 strategy.run() 内。
def run_strategy(
    strategy: Strategy, results: dict[str, list[dict]], n: int,
    k_values: list[int] | None = None,
) -> dict[int, MetricResult]:
    """Run a single strategy and return its scores."""
    return strategy.run(results, n, k_values=k_values)


# 〔讲解〕 跑「全部启发式策略」(pass/mv/wmv/bon/fewtool,均不调 LLM)。
# 〔讲解〕 返回 {策略名 -> {k -> MetricResult}};**kwargs 透传给各策略构造器(如 task/seed 等)。
def run_heuristic_strategies(
    results: dict[str, list[dict]], n: int, k_values: list[int] | None = None,
    **kwargs
) -> dict[str, dict[int, MetricResult]]:
    """Run all heuristic strategies and return their scores."""
    all_scores = {}
    # 〔讲解〕 工厂函数一次性实例化所有启发式策略对象。
    strategies = get_heuristic_strategies(**kwargs)

    for i, strategy in enumerate(strategies):
        all_scores[strategy.name] = strategy.run(results, n, k_values=k_values)
        # 〔讲解〕 除最后一个外,每跑完一个策略打印一条分隔线(末尾不多打,排版更干净)。
        if i < len(strategies) - 1:
            print(_SEPARATOR)

    return all_scores


# 〔讲解〕 跑「所有策略」:先全部启发式,再逐个 LLM 策略。
# 〔讲解〕 顺序设计=便宜的启发式(本地计算)先出结果,再跑贵的、要调大模型的聚合策略。
def run_all_strategies(
    results: dict[str, list[dict]], n: int, k_values: list[int] | None = None,
    **kwargs
) -> dict[str, dict[int, MetricResult]]:
    """Run all strategies: heuristic first, then LLM-based."""
    all_scores = {}
    # 〔讲解〕 先把启发式那批的结果并入总表(update 合并字典)。
    all_scores.update(run_heuristic_strategies(results, n, k_values=k_values, **kwargs))
    for name in _LLM_STRATEGY_NAMES:
        print(_SEPARATOR)
        # 〔讲解〕 LLM 策略逐个按名实例化再跑(每个内部都会调大模型 + 走 evaluation 判分)。
        strategy = get_strategy(name, **kwargs)
        all_scores[name] = strategy.run(results, n, k_values=k_values)
    return all_scores


# 〔讲解〕 程序主入口:解析命令行 → 找目录读数据 → 按 --strategy 分发执行。
# 〔讲解〕 这是用户 `python aggregate.py [选项] 目录...` 实际触发的函数。
def main():
    # 〔讲解〕 --strategy 的可选值 = 所有具体策略名 + 两个聚合档位:"heuristic"(全跑启发式) / "all"(全跑)。
    strategy_choices = list(STRATEGIES.keys()) + ["heuristic", "all"]

    parser = argparse.ArgumentParser(description="Test-Time Scaling evaluation (modular)")
    ### Aggregation Strategy
    parser.add_argument("--strategy", type=str, default="heuristic", choices=strategy_choices,
                        help="Evaluation strategy (default: heuristic)")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Number of parallel workers (default: 10)")
    # 〔讲解〕 --k 接收一串整数(nargs="+"),指定要评估的 k(聚合时取多少条候选);不传则默认跑 k=1,2,4,...,N。
    parser.add_argument("--k", type=int, nargs="+", default=None,
                        help="k values to evaluate (e.g. --k 1 2 4). If not set, runs all k=1,2,4,...,N")
    ### LLM-based Aggregation
    parser.add_argument("--model", type=str, default=None,
                        help="Model for LLM-based aggregation")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Base URL for vLLM deployment (optional)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory for logs")
    ### Evaluation
    # 〔讲解〕 --task 指定题目所属 benchmark,决定 evaluation 阶段用哪套判分逻辑(不同任务判分标准不同)。
    parser.add_argument("--task", type=str, default="browsecomp", choices=["browsecomp", "browsecomp-plus", "deepsearchqa", "hle", "researchrubrics", "healthbench"],
                        help="Task type for judging (default: browsecomp)")
    # 〔讲解〕 --skip_score:聚合时先不判分,只产出日志,事后再从日志补算分数(action="store_true" = 出现即为 True 的开关)。
    parser.add_argument("--skip_score", action="store_true",
                        help="Skip compute_score during aggregation; score posthoc from logs")

    # 〔讲解〕 位置参数:一个或多个「父目录」,脚本会在其下递归找叶子目录(即各次 rollout 的结果目录)。
    parser.add_argument("directories", nargs="+",
                        help="Parent directories to search for leaf directories (json)")

    args = parser.parse_args()

    # 〔讲解〕 校验:LLM 类策略(或 "all")必须给 --model,否则没法调大模型——缺则直接报错退出。
    if args.strategy in set(_LLM_STRATEGY_NAMES) | {"all"} and args.model is None:
        parser.error(f"--model is required for --strategy {args.strategy}")

    # Find all leaf directories from the given parent directories
    # 〔讲解〕 把命令行给的每个父目录都展开成叶子目录,汇总成一个大列表。
    all_leaves = []
    for d in args.directories:
        all_leaves.extend(find_leaf_directories(Path(d)))
    # 〔讲解〕 排序保证目录顺序稳定 → 结果可复现(读取/编号不随文件系统返回顺序漂移)。
    all_leaves = sorted(all_leaves)

    # 〔讲解〕 N = 叶子目录数 = rollout 总次数(每道题最多有 N 个候选)。
    n = len(all_leaves)
    if n == 0: parser.error("No leaf directories found")

    print(f"Found {n} leaf directories")
    # 〔讲解〕 真正读盘:把 N 个目录的 json 读出并按题分组(返回的 n 与上面一致)。
    results, n = load_results(all_leaves)

    n_problems = len(results)
    # 〔讲解〕 下面几行只为在控制台打印一个对齐的运行信息头(task / N / 题数 / 模型),纯展示无业务逻辑。
    header_parts = [f"task: {args.task}", f"N: {n}", f"problems: {n_problems}"]
    if args.model:
        header_parts.append(f"model: {args.model}")
    header = "  " + "   ".join(header_parts) + "  "
    # 〔讲解〕 横线长度按 header 字符数算,让上下边框与文字等宽。
    bar = "─" * len(header)
    print(f"\n{bar}\n{header}\n{bar}\n")

    # Extra kwargs forwarded to strategy constructors
    output_dir = args.output_dir
    # 〔讲解〕 仅当跑 LLM 类策略时才需要落盘日志目录(启发式策略无需建目录)。
    if args.strategy in _LLM_STRATEGY_NAMES + ["all"]:
        # 〔讲解〕 取第一个输入目录路径里的「非空片段」,用来给自动生成的输出目录起一个有辨识度的标签。
        _parts = [p for p in args.directories[0].split("/") if p]
        # 〔讲解〕 取路径最后 4 段用 "-" 拼成标签(如 task-model-batch-... ),路径太短则直接用原路径兜底。
        _dir_tag = "-".join(_parts[-4:]) if _parts else args.directories[0]
        # 〔讲解〕 用户没显式给 --output_dir 时,自动拼一个默认输出路径:output/aggregation/<模型>/<标签>。
        output_dir = args.output_dir or f"output/aggregation/{args.model}/{_dir_tag}"
        # 〔讲解〕 exist_ok=True:目录已存在也不报错(配合下面 resume 续跑很关键)。
        os.makedirs(output_dir, exist_ok=True)
    # 〔讲解〕 这一坨 kwargs 会被透传给每个策略的构造器;LLM 策略用得上 model/api_base/output_dir 等,
    # 〔讲解〕 启发式策略接到不认识的 key 也无妨(基类 __init__ 用 **kwargs 把多余参数吸收掉)。
    strategy_kwargs = {
        "model": args.model,
        "api_base": args.api_base,
        "task": args.task,
        "max_workers": args.max_workers,
        "output_dir": output_dir,
        # 〔讲解〕 resume=True:支持断点续跑,已算过的题可从 output_dir 复用,不重复烧 LLM token。
        "resume": True,
        "skip_score": args.skip_score,
    }

    k_values = args.k

    # 〔讲解〕 三路分发:档位 "heuristic" 跑全部启发式;"all" 跑全部;否则按名只跑指定的单个策略。
    if args.strategy == "heuristic":
        run_heuristic_strategies(results, n, k_values=k_values, **strategy_kwargs)
    elif args.strategy == "all":
        run_all_strategies(results, n, k_values=k_values, **strategy_kwargs)
    else:
        strategy = get_strategy(args.strategy, **strategy_kwargs)
        run_strategy(strategy, results, n, k_values=k_values)


if __name__ == "__main__":
    main()
