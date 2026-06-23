# 〔讲解〕本文件是 aggagent 子包的入口(package __init__)。它本身不实现逻辑,
# 〔讲解〕只负责把核心类 AggAgent 从内部模块 agent.py "提到包顶层",对外暴露统一入口。
# 〔讲解〕在 pipeline 里属于「聚合」一环:rollout/ 跑出多条 ReAct 轨迹后,AggAgent 这个
# 〔讲解〕会调工具读轨迹的聚合智能体把它们融合成一份答案,再交给 evaluation/ 判分。
# 〔讲解〕上游 = rollout 产出的多条轨迹;下游 = evaluation 模块。
"""AggAgent: agentic aggregation for parallel scaling of long-horizon agentic tasks."""

# 〔讲解〕从同包的 agent 模块(aggagent/agent.py)导入真正的实现类 AggAgent。
# 〔讲解〕这一行的意义:让外部代码可以写 `from aggagent import AggAgent`,
# 〔讲解〕而不必关心它实际定义在哪个子文件里(隐藏内部目录结构)。
from .agent import AggAgent

# 〔讲解〕__all__ 声明 `from aggagent import *` 时只导出 AggAgent,
# 〔讲解〕同时也起到「本包公开 API 清单」的文档作用——告诉读者本包对外只有这一个类。
__all__ = ["AggAgent"]
