# 〔讲解〕本文件是 evaluation/ 包的「门面/总入口」(package facade)。
# 〔讲解〕在 AggAgent pipeline 中处于「判分」这一环:rollout 跑出轨迹、aggregation/aggagent 聚合出最终答案后,
# 〔讲解〕由本包负责用大模型(LLM-as-judge)给答案打分。它把 4 个具体评测器(各对应一个 benchmark)统一收口,
# 〔讲解〕对外只暴露两个工厂函数:get_evaluator(按任务名拿评测器)、get_task_instruction(按任务名拿作答指令)。
# 〔讲解〕上游:实验主流程按数据集名调用这里取评测器/指令;下游:各 *Evaluator 类(browsecomp/deepsearchqa/healthbench/researchrubrics)。
from __future__ import annotations

# 〔讲解〕从同包的 4 个子模块各引入「一个评测器类 + (部分)一个作答指令常量」。
# 〔讲解〕约定:Evaluator 类负责判分;XXX_INSTRUCTION 是给被测模型的「答案该怎么写」的格式说明(prompt 片段)。
# 〔讲解〕注意 healthbench / researchrubrics 只导入了评测器、没有对应 INSTRUCTION,因为这两个任务不需要统一作答格式约束。
from .browsecomp import BrowseCompEvaluator, BROWSECOMP_INSTRUCTION
from .deepsearchqa import DeepSearchQAEvaluator, DEEPSEARCHQA_INSTRUCTION
from .healthbench import HealthBenchEvaluator
from .researchrubrics import ResearchRubricsEvaluator


# 〔讲解〕在模块导入时就各实例化一份评测器,做成「单例」(模块级全局对象)。
# 〔讲解〕这样下面的工厂函数每次返回的是同一个共享实例,避免反复 new、也便于复用其内部状态/客户端,属于轻量缓存手法。
_BROWSECOMP_EVAL = BrowseCompEvaluator()
_DEEPSEARCHQA_EVAL = DeepSearchQAEvaluator()
_HEALTHBENCH_EVAL = HealthBenchEvaluator()
_RESEARCHRUBRICS_EVAL = ResearchRubricsEvaluator()


# 〔讲解〕工厂函数:按任务名(数据集名)分发,返回对应的评测器单例。
# 〔讲解〕入参 task 是字符串任务标识;返回值是某个 *Evaluator 实例(有统一的判分接口)。
# 〔讲解〕角色:让上游调用方无需 import 具体类,只靠一个 task 字符串就能拿到正确的判分器,解耦「任务名」与「实现类」。
def get_evaluator(task: str):
    if task == "deepsearchqa":
        return _DEEPSEARCHQA_EVAL
    if task == "healthbench":
        return _HEALTHBENCH_EVAL
    if task == "researchrubrics":
        return _RESEARCHRUBRICS_EVAL
    # 〔讲解〕兜底分支:其余所有任务名(含 browsecomp / browsecomp-plus / hle 等)统一走 BrowseComp 评测器,即默认判分器。
    return _BROWSECOMP_EVAL


# 〔讲解〕工厂函数:按任务名返回「给被测模型的作答指令」(answer-format 提示词片段)。
# 〔讲解〕入参 task 是任务名;返回 str。这段指令会拼进 rollout 阶段的 prompt,约束模型按可被自动判分的格式输出答案。
def get_task_instruction(task: str) -> str:
    # 〔讲解〕browsecomp 系列(含 plus)与 hle 三个任务共用同一套 BrowseComp 作答指令,故合并判断。
    if task in ("browsecomp", "browsecomp-plus", "hle"):
        return BROWSECOMP_INSTRUCTION
    if task == "deepsearchqa":
        return DEEPSEARCHQA_INSTRUCTION
    # 〔讲解〕其余任务(如 healthbench / researchrubrics)无统一作答格式要求,返回空串表示「不附加指令」。
    return ""


# 〔讲解〕__all__ 显式声明 `from evaluation import *` 时只导出这两个工厂函数,
# 〔讲解〕把上面的具体类与单例视作内部实现而隐藏,对外维持「只用 task 字符串」的简洁契约。
__all__ = ["get_evaluator", "get_task_instruction"]
