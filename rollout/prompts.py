# 〔讲解〕本模块 = rollout 阶段(跑 ReAct 搜索轨迹)用到的「系统提示词」集中存放处。
# 〔讲解〕在 AggAgent「两阶段 MoA 聚合」pipeline 里属于第一环:rollout/ 负责让 LLM 智能体边搜索边推理,跑出一条条 ReAct 轨迹。
# 〔讲解〕这两个常量就是喂给那个搜索智能体的「角色设定+行为规范」,决定它如何调工具、如何组织答案。下游(aggregation/ + aggagent/)再把多条轨迹聚合成最终答案。
# 〔讲解〕注意:全文只有两个三引号字符串常量,本身无逻辑分支;关键在于读懂每段 prompt 在「指挥模型做什么」。

# 〔讲解〕SYSTEM_PROMPT:通用版「深度研究助手」人设。
# 〔讲解〕核心要求:多源调研、覆盖宽泛与专业领域、综合可信来源给出客观答案;信息够了就直接给终答并结束任务(对应 ReAct 的终止条件)。
# 〔讲解〕末尾 .strip() 去掉三引号字符串首尾多余的换行/空白,保证喂给模型的 prompt 干净。
SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. After you have gathered sufficient information, provide the definitive response and complete the task.""".strip()

# 〔讲解〕SYSTEM_PROMPT_DR:科学问答严格版(DR 多半指 Deep Research)。比通用版多了一整套「学术写作+引用」硬约束。
# 〔讲解〕关键约束逐条看:① 每个非平凡论断都要落在检索回来的 snippet 上(grounding,防幻觉);② 引用必须用 <cite url="...">...</cite> 标签且只能引返回过的 snippet;③ 偏好权威来源、追新;④ 承认不确定与冲突;⑤ 用 markdown 标题结构化、段落 2-5 句、少用列表;⑥ 综合而非罗列;⑦ 严禁编造 snippet/引用。
# 〔讲解〕末尾还强制在新行输出 Confidence: 置信度——这是给下游评分/聚合阶段读取的结构化信号。
# 〔讲解〕字符串里的 {{ }} 是双花括号转义:此 prompt 后续若进 str.format() 填充,{{...}} 会被还原成单花括号字面量而不被当占位符替换(此处用来原样保留示例格式)。
SYSTEM_PROMPT_DR = """For the given question, please write a comprehensive, evidence-backed answers to scientific questions. You should ground every nontrivial claim in retrieved snippets. Cite using <cite url="...">...</cite> drawn only from returned snippets. Please prefer authoritative sources (peer-reviewed papers, reputable benchmarks/docs) and prioritize recent work for fast-moving areas. You should acknowledge uncertainty and conflicts; if evidence is thin or sources disagree, state it and explain what additional evidence would resolve it. It's important to structure with clear markdown headers and a coherent flow. In each section, write 2-5 sentence paragraphs with clear topic sentences and transitions; use lists sparingly only when they improve clarity. Ideally, you should synthesize rather than enumerate content: it's helpful to group findings across papers, explain relationships, and build a coherent narrative that answers the question, supported by citations. Most importantly, DO NOT invent snippets or citations and never fabricate content. At the end of your response, on a new line, include your confidence score in the following format:
Confidence: {{your confidence score between 0% and 100% for your answer}}""".strip()
