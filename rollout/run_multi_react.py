# 〔讲解〕本文件是 rollout/ 这一环的「命令行总入口脚本」(直接 python 跑它来产数据)。
# 〔讲解〕在 AggAgent pipeline 中它处于最上游:负责对某个 benchmark 数据集,对【每道题】用 ReAct 智能体
# 〔讲解〕独立跑 N 次搜索轨迹(roll_out_count 条 rollout),每条轨迹判分后落盘成一个 JSON 文件。
# 〔讲解〕这些 JSON 就是后续 aggregation/ + aggagent/(多源聚合)和 evaluation/(判分)的输入原料。
# 〔讲解〕关键工程能力:数据分片(多 worker 并行)、断点续跑(扫已完成的题跳过)、并发限流(semaphore)、单任务超时兜底。
# 〔讲解〕上游 = data/<dataset>.jsonl 题目文件;核心被驱动对象 = react_agent.MultiTurnReactAgent;下游 = aggregation/aggagent/evaluation。
import argparse
import json
import os
import asyncio
from tqdm import tqdm
from datetime import datetime
# 〔讲解〕MultiTurnReactAgent = 真正干活的「多轮 ReAct 搜索智能体」,本脚本只是它的批量调度外壳。
from react_agent import MultiTurnReactAgent
# 〔讲解〕get_evaluator = 按数据集名拿对应的 LLM 判分器;出错兜底时也要给一条轨迹打 0 分,故这里要 import。
from evaluation import get_evaluator
import math
import uuid

# 〔讲解〕整份脚本逻辑全包在 __main__ 守卫内:它是被当作命令行程序运行的,不是被 import 复用的库。
if __name__ == "__main__":
    # 〔讲解〕用 argparse 定义全部命令行参数。default 多与论文/官方脚本对齐(复现保真),勿凭记忆改默认值。
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="", help="Model path")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for results")
    # 〔讲解〕dataset 用 choices 限定为 6 个支持的 benchmark;传别的名字 argparse 会直接报错拦下。
    parser.add_argument("--dataset", type=str, default="browsecomp", choices=["browsecomp", "browsecomp-plus", "hle", "deepsearchqa", "healthbench", "researchrubrics"], help="Dataset name")
    # 〔讲解〕roll_out_count = 每道题独立重跑几条轨迹(N),正是「多源聚合」里的「多源」数量来源。
    parser.add_argument("--roll_out_count", type=int, default=4, help="Number of rollouts per question")
    # 〔讲解〕max_workers = 异步并发上限(下面用 semaphore 实现),控制同时打到本地模型服务的请求数。
    parser.add_argument("--max_workers", type=int, default=20, help="Maximum number of concurrent workers")
    parser.add_argument("--max_instances", type=int, default=None, help="Optional cap on number of instances to process after split")
    # 〔讲解〕total_splits / worker_split = 把题目集均分成多份、本进程只跑第几份,用于多机/多卡并行(见下方分片逻辑)。
    parser.add_argument("--total_splits", type=int, default=1, help="Total number of worker splits")
    parser.add_argument("--worker_split", type=int, default=1, help="Index of this worker's split")
    # 〔讲解〕api_base = 被测模型的 OpenAI 兼容服务地址(通常是本地 vLLM 起的 /v1)。
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1", help="API base URL for serving local model")
    parser.add_argument("--max_llm_call_per_run", type=int, default=100, help="Max LLM calls per run")
    # 〔讲解〕max_tokens 是「整条轨迹的上下文 token 预算」,超了就逼模型停手给最终答案(注意与下面单次生成的 max_tokens 区分)。
    parser.add_argument("--max_tokens", type=int, default=108000, help="Context token limit (triggers final answer when exceeded)")
    args = parser.parse_args()

    model = args.model
    # 〔讲解〕从模型路径里取末段目录名当 model_name(rstrip('/') 防末尾斜杠导致 basename 取到空串)。
    model_name = os.path.basename(model.rstrip('/'))
    dataset = args.dataset
    # 〔讲解〕数据集文件按约定固定在 data/<dataset>.jsonl,体现「输出位置/文件命名按原仓库默认」的复现纪律。
    dataset_dir = f"data/{dataset}.jsonl"
    # 〔讲解〕输出目录:用户没传 --output_dir 时,按 output/rollout/<模型名>/<数据集> 的默认结构自动落盘。
    output_dir = args.output_dir or f"output/rollout/{model_name}/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    roll_out_count = args.roll_out_count
    max_instances = args.max_instances
    total_splits = args.total_splits
    worker_split = args.worker_split

    # Validate worker_split
    # 〔讲解〕分片下标必须落在 [1, total_splits] 内,否则后面切片会算出空集或越界,这里提前报错退出更安全。
    if worker_split < 1 or worker_split > total_splits:
        print(f"Error: worker_split ({worker_split}) must be between 1 and total_splits ({total_splits})")
        exit(1)

    print(f"Model name: {model_name}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of rollouts: {roll_out_count}")
    print(f"Data splitting: {worker_split}/{total_splits}")

    # Read dataset
    # 〔讲解〕读题目文件:同时兼容 .json(整个文件是一个 list)与 .jsonl(每行一个 JSON 对象)两种格式。
    try:
        if dataset_dir.endswith(".json"):
            with open(dataset_dir, "r", encoding="utf-8") as f:
                items = json.load(f)
            # 〔讲解〕.json 格式要求顶层必须是 list、且元素是 dict(每道题一个对象),否则视为格式非法。
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
            if items and not isinstance(items[0], dict):
                raise ValueError("Input JSON list items must be objects.")
        elif dataset_dir.endswith(".jsonl"):
            # 〔讲解〕jsonl 逐行 json.loads 解析成 dict 列表,这是本仓库默认的数据格式。
            with open(dataset_dir, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
    # 〔讲解〕文件不存在 / 解析失败都分别给明确错误并退出,避免后续代码在坏数据上裸奔。
    except FileNotFoundError:
        print(f"Error: Input file not found at {dataset_dir}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {dataset_dir}: {e}")
        exit(1)

    # Apply data splitting
    # 〔讲解〕数据分片核心:把全部题目均分成 total_splits 份,本 worker 只取自己那一段 [start_idx, end_idx)。
    total_items = len(items)
    # 〔讲解〕用 ceil 向上取整算每份大小,保证 total_splits 份能完整覆盖所有题(最后一份可能略少)。
    items_per_split = math.ceil(total_items / total_splits)
    # 〔讲解〕worker_split 从 1 开始计数,故减 1 转成从 0 开始的偏移再乘份大小,得到本份起点。
    start_idx = (worker_split - 1) * items_per_split
    # 〔讲解〕终点取 min 防越界:最后一份不足整段时不会读到 list 末尾之外。
    end_idx = min(worker_split * items_per_split, total_items)

    # Split the dataset
    # 〔讲解〕真正切片:此后 items 只剩本 worker 负责的题,实现多进程互不重叠地分担工作量。
    items = items[start_idx:end_idx]

    # Optional cap after split selection
    # 〔讲解〕可选地再截断到前 max_instances 道(常用于 smoke test 只跑几题验证流程),负数视为非法。
    if max_instances is not None:
        if max_instances < 0:
            print(f"Error: max_instances ({max_instances}) must be >= 0")
            exit(1)
        items = items[:max_instances]

    print(f"Total items in dataset: {total_items}")
    print(f"Processing items {start_idx} to {end_idx-1} ({len(items)} items)")
    if max_instances is not None:
        print(f"Max instances cap applied: {max_instances}")

    # Create subdirectories per rollout
    # 〔讲解〕为每条 rollout(第 1..N 次重跑)各建一个子目录,N 条轨迹的结果分目录存放、互不混淆。
    print(f"Creating rollout directories...")
    rollout_dirs = {}
    for rollout_idx in range(1, roll_out_count + 1):
        # 〔讲解〕多 worker 时目录名带上 split 后缀(iterX_splitYofZ),避免不同 worker 写进同名目录造成冲突。
        if total_splits > 1:
            rollout_dir = os.path.join(output_dir, f"iter{rollout_idx}_split{worker_split}of{total_splits}")
        else:
            rollout_dir = os.path.join(output_dir, f"iter{rollout_idx}")
        os.makedirs(rollout_dir, exist_ok=True)
        print(f"Rollout directory {rollout_idx}: {rollout_dir}")
        rollout_dirs[rollout_idx] = rollout_dir

    # Check for already processed queries by scanning subdirectories
    # 〔讲解〕断点续跑机制(上半):逐个 rollout 目录扫已有结果文件,把「已成功跑过的题」收集成集合,后面用来跳过。
    processed_queries_per_rollout = {}
    for rollout_idx in range(1, roll_out_count + 1):
        rollout_dir = rollout_dirs[rollout_idx]
        processed_queries = set()
        if os.path.exists(rollout_dir):
            # Scan all JSON files in the rollout directory
            for filename in os.listdir(rollout_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(rollout_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # 〔讲解〕只有「有 question 字段且没有 error」的文件才算成功:失败/超时的结果不计入已完成,留待重跑。
                            if "question" in data and "error" not in data:
                                processed_queries.add(data["question"].strip())
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        # Skip invalid files
                        # 〔讲解〕坏掉/读不动的结果文件直接忽略,不让一个损坏文件中断整个扫描。
                        pass
        processed_queries_per_rollout[rollout_idx] = processed_queries

    # 〔讲解〕断点续跑机制(下半):构建「真正要跑的任务列表」tasks_to_run_all,每项 = (一道题, 属于第几条 rollout)。
    tasks_to_run_all = []
    per_rollout_task_counts = {i: 0 for i in range(1, roll_out_count + 1)}
    for rollout_idx in range(1, roll_out_count + 1):
        processed_queries = processed_queries_per_rollout[rollout_idx]
        for item in items:
            # 〔讲解〕不同 benchmark 把「题面」放在不同字段,这里统一抽出来塞进 item["question"],下游只认 question。
            # 〔讲解〕browsecomp / deepsearchqa 用 "problem",healthbench 用 "prompt_id",researchrubrics 用 "prompt"。
            if dataset == "browsecomp" or dataset == "browsecomp-plus":
                question = item.get("problem", "").strip()
                item["question"] = question
            elif dataset == "deepsearchqa":
                question = item.get("problem", "").strip()
                item["question"] = question
            elif dataset == "healthbench":
                question = item.get("prompt_id", "").strip()
                item["question"] = question
            elif dataset == "researchrubrics":
                question = item.get("prompt", "").strip()
                item["question"] = question
            question = item.get("question", "").strip()
            # 〔讲解〕兜底抽题:若上面没抽到,尝试从 OpenAI 风格 messages 的第 2 条(index 1,即首条 user 消息)里取内容。
            if question == "":
                try:
                    user_msg = item["messages"][1]["content"]
                    # 〔讲解〕有的数据把题面写成 "User: <真题>" 前缀格式,这里按 "User:" 切分只留真正的问题文本。
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                    item["question"] = question
                except Exception as e:
                    print(f"Extract question from user message failed: {e}")
            # 〔讲解〕实在抽不到题面的脏数据直接跳过,不浪费一次模型调用。
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            # 〔讲解〕续跑关键判断:这道题在本 rollout 没被成功跑过才加入任务列表(已完成的自动跳过)。
            if question not in processed_queries:
                tasks_to_run_all.append({
                    # 〔讲解〕用 item.copy() 存浅拷贝,防同一 item 对象被多条 rollout 共享后相互污染字段。
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                })
                per_rollout_task_counts[rollout_idx] += 1

    print(f"Total questions in current split: {len(items)}")
    for rollout_idx in range(1, roll_out_count + 1):
        print(f"Rollout {rollout_idx}: already successfully processed: {len(processed_queries_per_rollout[rollout_idx])}, to run: {per_rollout_task_counts[rollout_idx]}")

    # 〔讲解〕若所有题都已跑完(任务列表为空),直接打印提示、不进入下面的执行分支。
    if not tasks_to_run_all:
        print("All rollouts have been completed and no execution is required.")
    else:
        # 〔讲解〕llm_cfg = 传给 ReAct 智能体的模型配置。generate_cfg 是真正发请求时的采样/连接参数。
        llm_cfg = {
            'model': model,
            'model_name': model_name,
            'model_type': model,
            'generate_cfg': {
                # 〔讲解〕model 前缀 hosted_vllm/ 是 litellm 的路由约定,告诉它走「自托管 vLLM(OpenAI 兼容)」通道。
                'model': f"hosted_vllm/{model_name}",
                # 〔讲解〕这里的 max_tokens 是「单次生成」的输出上限,与最上面那个「整条轨迹上下文预算」不是一回事,别混。
                'max_tokens': 10000,
                # 〔讲解〕关掉并行工具调用:强制模型一次只发一个工具调用,让 ReAct 轨迹是严格串行的「思考→调一次工具→看结果」。
                "parallel_tool_calls": False,
                # 〔讲解〕temperature=1.0 + top_p=0.95 留足随机性,这正是「同一道题重跑 N 次能得到不同轨迹」的来源(多源多样性)。
                "temperature": 1.0,
                "top_p": 0.95,
                "api_base": args.api_base,
                # 〔讲解〕本地服务不校验密钥,按 OpenAI SDK 要求随便填个非空占位 "EMPTY" 即可。
                "api_key": "EMPTY",
            },
        }

        # 〔讲解〕按模型家族注入各自的「思考模式」开关:不同后端开启 chain-of-thought / reasoning 的参数名不同,这里分别适配。
        if "glm" in model.lower():
            llm_cfg["model_type"] = "glm"
            llm_cfg["generate_cfg"]["extra_body"] = { "chat_template_kwargs": { "enable_thinking": True, "clear_thinking": False } }
        elif "qwen" in model.lower():
            llm_cfg["model_type"] = "qwen"
            llm_cfg["generate_cfg"]["extra_body"] = { "chat_template_kwargs": { "enable_thinking": True } }
        elif "minimax" in model.lower():
            llm_cfg["model_type"] = "minimax"
            llm_cfg["generate_cfg"]["extra_body"] = {"reasoning_split": True}

        # 〔讲解〕实例化一个共享的 ReAct 智能体:所有任务都复用它(它内部根据 task 决定挂哪套搜索/访问工具)。
        # 〔讲解〕function_list=None → 由 Agent 内部按 task 自己选工具集(见 react_agent 构造函数)。
        test_agent = MultiTurnReactAgent(
            llm=llm_cfg,
            function_list=None,
            task=dataset,
            max_llm_call_per_run=args.max_llm_call_per_run,
            max_tokens=args.max_tokens,
        )

        # 〔讲解〕失败/超时结果用的占位 debug 数据:正常轨迹会有真实的 token 长度/全量消息/工具用量,出错时填空壳保持字段齐整。
        empty_debug_data = {
            "token_lengths_each_step": [],
            "full_messages": [],
            "tool_usage": {},
        }

        # Metadata to include in all output files
        # 〔讲解〕run_metadata = 把本次运行的全部关键配置打包,写进每个输出 JSON,让结果可追溯「是哪套参数跑出来的」。
        run_metadata = {
            "model": model,
            "output_dir": output_dir,
            "dataset": dataset,
            "roll_out_count": roll_out_count,
            "max_workers": args.max_workers,
            "max_instances": max_instances,
            "total_splits": total_splits,
            "worker_split": worker_split,
            "api_base": args.api_base,
            "max_llm_call_per_run": args.max_llm_call_per_run,
            "max_tokens": args.max_tokens,
        }

        # 〔讲解〕同步写 JSON 的小工具。ensure_ascii=False 保留中文原文,indent=2 便于人工查看。
        def write_json_file(filepath, data):
            """Write JSON data to file synchronously"""
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        # 〔讲解〕get_score:把一条轨迹的判分结果(auto_judge)折算成统一的 0~1 标量分,供打印该 rollout 的平均分用。
        # 〔讲解〕不同 benchmark 判分口径不同:deepsearchqa 看 all_correct 布尔;healthbench/researchrubrics 取连续的 overall_score;其余看 correctness 是否 "correct"。
        def get_score(data: dict) -> float:
            auto_judge = data.get("auto_judge", {})
            if dataset == "deepsearchqa":
                return 1.0 if auto_judge.get("all_correct") == True else 0.0
            elif dataset in ("healthbench", "researchrubrics"):
                return float(auto_judge.get("metrics", {}).get("overall_score", 0.0))
            else:
                return 1.0 if auto_judge.get("correctness") == "correct" else 0.0

        # 〔讲解〕process_tasks:整个并发执行的协程总调度。内部定义三件套(限流执行 / 单任务落盘 / 分 rollout 推进)后逐 rollout 跑完。
        async def process_tasks():
            # 〔讲解〕Semaphore(max_workers) = 并发闸门:最多同时放 max_workers 个任务进去跑,其余排队,防止压垮模型服务。
            semaphore = asyncio.Semaphore(args.max_workers)
            rollout_scores = {i: [] for i in range(1, roll_out_count + 1)}

            # 〔讲解〕run_task_with_semaphore:单个任务的「带限流 + 超时 + 异常捕获」包装器。
            # 〔讲解〕返回四元组 (task, result, error, 耗时):正常时 error=None;超时返回 "timeout";其它异常把 exc 原样带回。
            async def run_task_with_semaphore(task):
                # 〔讲解〕async with semaphore:进入即占一个并发名额,退出(无论成败)自动释放,保证名额不泄漏。
                async with semaphore:
                    t_start = asyncio.get_event_loop().time()
                    try:
                        # Use asyncio.wait_for for timeout handling
                        # 〔讲解〕核心调用:test_agent._run(...) 真正去跑这一条 ReAct 轨迹;wait_for 给它套 3600 秒硬超时。
                        result = await asyncio.wait_for(
                            test_agent._run(task, model),
                            timeout=3600.0  # 60 minutes timeout
                        )
                        return task, result, None, asyncio.get_event_loop().time() - t_start
                    # 〔讲解〕超时单独成一类错误(返回 "timeout" 字符串),下面 handle 时会区别对待并打超时日志。
                    except asyncio.TimeoutError:
                        return task, None, "timeout", asyncio.get_event_loop().time() - t_start
                    # 〔讲解〕其它任何异常都吞掉、带回 exc 对象,确保单个任务崩溃不会拖垮整批运行(健壮性关键)。
                    except Exception as exc:
                        return task, None, exc, asyncio.get_event_loop().time() - t_start

            # 〔讲解〕handle_single_task:拿到一个已完成任务的结果后,负责「构造完整记录 + 写文件 + 累计分数」。
            async def handle_single_task(task_info, result, error, time):
                """Handle a single completed task: write result to file."""
                rollout_idx = task_info["rollout_idx"]
                rollout_dir = rollout_dirs[rollout_idx]

                # Generate unique timestamp-based filename with UUID suffix to prevent collisions
                # 〔讲解〕文件名 = UTC 时间戳(到微秒)+ UUID 前 8 位。两层去重,确保高并发下多个任务同刻写入也不会撞文件名。
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
                unique_id = str(uuid.uuid4())[:8]  # First 8 chars of UUID for uniqueness
                output_file = os.path.join(rollout_dir, f"run_{ts}_{unique_id}.json")

                # 〔讲解〕分支一:超时。即便失败也要写一条带 error 标记的完整记录(并调判分器给 0 分),保证统计口径不缺项、续跑时知道它失败过。
                if error == "timeout":
                    print(f'Timeout (>3600s): "{task_info["item"]["question"]}" (Rollout {rollout_idx})')
                    evaluator = get_evaluator(dataset)
                    error_result = {
                        "metadata": run_metadata,
                        "question": task_info["item"].get("question", ""),
                        "instance": task_info["item"],
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": "Timeout (>3600s)",
                        "messages": [],
                        "prediction": "[Failed]",
                        "time": time,
                        "debug_data": empty_debug_data,
                        "cost": {"rollout": 0.0, "tool": 0.0},
                        # 〔讲解〕error=True 让判分器走「失败兜底」分支,直接产出一份 0 分的 auto_judge,而不是真去判一个空答案。
                        "auto_judge": await evaluator.compute_score(prediction="", item=task_info["item"], error=True, err_msg="Timeout (>3600s)"),
                        "termination": "error"
                    }
                    # 〔讲解〕用 asyncio.to_thread 把同步的写文件丢到线程池执行,避免阻塞事件循环、卡住其它并发任务。
                    await asyncio.to_thread(write_json_file, output_file, error_result)
                    rollout_scores[rollout_idx].append(0.0)
                # 〔讲解〕分支二:其它异常。与超时几乎同构,只是 error 文案换成 "Future resolution failed: ..."。
                elif error is not None:
                    print(f'Task for question "{task_info["item"]["question"]}" (Rollout {rollout_idx}) generated an exception: {error}')
                    evaluator = get_evaluator(dataset)
                    error_result = {
                        "metadata": run_metadata,
                        "question": task_info["item"].get("question", ""),
                        "instance": task_info["item"],
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": f"Future resolution failed: {error}",
                        "messages": [],
                        "prediction": "[Failed]",
                        "time": time,
                        "debug_data": empty_debug_data,
                        "cost": {"rollout": 0.0, "tool": 0.0},
                        "auto_judge": await evaluator.compute_score(prediction="", item=task_info["item"], error=True, err_msg=f"Future resolution failed: {error}"),
                        "termination": "error"
                    }
                    await asyncio.to_thread(write_json_file, output_file, error_result)
                    rollout_scores[rollout_idx].append(0.0)
                # 〔讲解〕分支三:成功。Agent 返回的 result 已含 question/messages/prediction/auto_judge 等,这里只补一层 metadata 再落盘。
                else:
                    # 〔讲解〕{"metadata": ..., **result} = 把 metadata 与 result 字段合并成一个 dict(result 的键平铺进来)。
                    result_with_metadata = {"metadata": run_metadata, **result}
                    await asyncio.to_thread(write_json_file, output_file, result_with_metadata)
                    rollout_scores[rollout_idx].append(get_score(result))

            # Group tasks by rollout_idx
            # 〔讲解〕按 rollout_idx 把任务重新分桶:目的是后面「一个 rollout 跑完再跑下一个」,而非全部 N×题 混在一起跑。
            tasks_by_rollout = {}
            for task in tasks_to_run_all:
                rollout_idx = task["rollout_idx"]
                if rollout_idx not in tasks_by_rollout:
                    tasks_by_rollout[rollout_idx] = []
                tasks_by_rollout[rollout_idx].append(task)

            # Process each rollout sequentially
            # 〔讲解〕逐 rollout 顺序推进:每个 rollout 内部是并发的,rollout 之间是串行的,便于每跑完一轮就报一次该轮平均分。
            for rollout_idx in sorted(tasks_by_rollout.keys()):
                rollout_tasks = tasks_by_rollout[rollout_idx]
                print(f"\n=== Starting Rollout {rollout_idx} ({len(rollout_tasks)} tasks) ===")

                # Create tasks for this rollout
                # 〔讲解〕为本 rollout 的每道题创建一个协程(此时还没真正开始执行,只是 coroutine 对象列表)。
                tasks = [run_task_with_semaphore(task) for task in rollout_tasks]

                # Process tasks as they complete for this rollout
                # 〔讲解〕as_completed = 谁先跑完就先拿谁,配 tqdm 实时显示进度条;拿到一个就立刻 handle 落盘,不必等全部完成。
                for coro in tqdm(asyncio.as_completed(tasks), total=len(rollout_tasks), desc=f"Processing Rollout {rollout_idx}"):
                    task_info, result, error, time = await coro
                    await handle_single_task(task_info, result, error, time)

                # 〔讲解〕本 rollout 全部完成后,算并打印这一轮的平均分(×100 转百分比),给实验者即时反馈。
                scores = rollout_scores[rollout_idx]
                mean = sum(scores) / len(scores) if scores else 0.0
                print(f"=== Completed Rollout {rollout_idx} | Score: {mean * 100:.2f}% ({len(scores)} items) ===\n")

        # Run the async processing
        # 〔讲解〕asyncio.run 启动事件循环,真正开跑上面定义的整个并发流程,跑完才返回。
        asyncio.run(process_tasks())
        print("\nAll tasks completed!")

    print(f"\nAll {roll_out_count} rollouts completed!")
