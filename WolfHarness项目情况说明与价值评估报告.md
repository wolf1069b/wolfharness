# WolfHarness(AgentPool)项目情况说明与价值评估报告

- **状态**:草稿 v3.1(新增 §2.8 框架对比、§4.6 开源论证)
- **日期**:2026-08-25
- **用途**:上级领导汇报材料 — 就 WolfHarness(原 AgentPool)项目上传 GitHub 一事作情况说明,同时评估项目价值,供领导决策参考
- **数据说明**:业界数据截至 2026 年中,来源为公开报道/官方公告;项目侧数据来自代码库盘点

---

## 一、情况说明

### 1.1 做了什么

本项目 fork 自 MIT 许可的开源项目 **phil65/agentpool**(上游本就在 GitHub 公开),团队在其基础上进行了大规模拓展——累计 9,415 次提交、17.5 万行 Python、662 个测试文件,并已更名为 WolfHarness v4.0。项目实现了完整的 ACP 协议(与官方 Rust 参考逐文件同步,哈希锁定),配套双层协议测试。

### 1.2 为什么在 GitHub 上开发

选型 agentpool 作为技术底座后,需要快速验证其能否满足业务需求——GitHub 是开源项目的原生协作平台,fork、提交、CI/CD、Issue 管理等工具链开箱即用,**在验证阶段优先考虑的是开发效率和迭代速度**。这是工程团队在技术选型验证期的常见做法:先跑通价值,再做合规化。如今项目已积累了核心价值和工程资产(17.5 万行、五协议、ACP 哈希同步),正是从"快速验证"转入"正当信息安全管理"的时机。

### 1.3 内容性质

仓库内容主要为**上游开源代码(MIT 许可)与在此基础上的个人拓展**,不涉及公司专有业务代码、密钥或敏感数据。MIT 许可明确允许 fork、修改与再分发,不存在知识产权侵权问题;唯一需要检视的是"上传外部平台是否符合公司内部流程"这一管理事项。

### 1.4 当前处理与承诺

- **当前处理**:已就仓库状态配合公司流程处理中(设为私有或删除),相关流程正在推进。
- **态度**:上传行为初衷是在验证阶段追求效率,认识到应更早同步公司流程,愿诚恳接受规范并立即整改。
- **承诺**:无论评估结果如何,完全服从公司决定;今后所有代码托管、对外发布行为,一律先走公司审批流程。

### 1.5 整改行动计划

| 阶段 | 行动 | 状态 |
|------|------|------|
| 立即 | 仓库转为私有或删除,GitHub 停止公开访问 | 流程推进中 |
| 立即 | 全仓库自查:确认无密钥、内部数据、专有业务代码(需安全/法务复核) | 待执行 |
| 1 周内 | 完成自查报告,提交安全/法务确认"无敏感内容" | 待执行 |
| 2 周内 | 完成评估,明确:以公司名义重新托管 / 转内部平台 / 终止公开,三选一 | 待决策 |

---

## 二、项目价值

### 2.1 项目定位

WolfHarness 是基于 PydanticAI 的多 Agent 编排框架,核心理念"**One YAML, every protocol**"——用一份 YAML 定义 Agent,同时通过 ACP、OpenCode、MCP、AG-UI、OpenAI API 五种协议对外暴露。

| 维度 | 业界主流做法 | WolfHarness 做法 |
|------|------------|-----------------|
| Agent 定义 | 每个协议/平台各写一套 | YAML 定义一次,五协议自动暴露 |
| Agent 异构性 | 通常同框架内协作 | native + ACP + Claude Code + AG-UI 四类异构 Agent 同池编排 |
| 协议覆盖 | 通常实现 1-2 种 | 五协议全覆盖 |
| 模型绑定 | 多数绑定单一厂商 | 8 类模型配置,跨厂商 fallback 链,零硬编码 |

### 2.2 已验证的工程实力

- **17.5 万行 Python 源码**,154 个文件深度集成 PydanticAI
- **662 个测试文件**,4 层金字塔(Unit → Integration → VCR → E2E)
- **mypy --strict 全绿**,import-linter 强制架构合同
- **OpenSpec 变更管控**:49 个 specs,结构性变更全流程可追溯
- **Logfire 全链路插桩**(RunLoop、Turn、delegation、protocol entry points)

### 2.3 已在框架上验证的业务场景(不是纸面提案,是已跑通的实践)

以下场景均在 WolfHarness 上一一验证了其价值:

| 场景 | 验证内容 | 价值证明 |
|------|---------|---------|
| **翻译场景** | 多语种翻译 Agent,支撑三一全球化(180+ 国家/地区)业务 | 替代人工翻译,降低跨国协作沟通成本 |
| **Wiki 生成场景** | 自动从技术文档/维修手册生成结构化知识库 | 知识沉淀自动化,降低文档维护人力 |
| **故障诊断场景** | 知识库故障诊断 Agent(`kb_diag_agent.yaml`),MCP 接入知识库 + 工具描述重写 | 已落地运行,验证了"知识库+Agent"闭环诊断能力 |
| **知识闭环场景** | 从知识检索→诊断推理→结果标注→回流知识库的完整闭环 | 知识越用越准,实现"知识自生长"而非静态库 |
| **焊接场景** | 焊接工艺参数 Agent,验证工业工艺场景的 Agent 化可行性 | 从通用框架向工业垂直场景延伸的实证 |
| **专家标注场景** | 领域专家标注工作流,支持知识结构化与质量管控 | 解决"老师傅经验"如何数字化、可复用的问题 |

### 2.4 标准化体系:预判对齐成本,减少重复沟通

更重要的是,我们在实践中预判到了一个行业级问题:**Agent 生态充满不确定性——协议在变、模型在变、工具在变,如果每遇到一个新需求就从头对齐,开发沟通成本将无底洞化。**

为此,我们正在 WolfHarness 上建立一套**标准的工作流体系与统一对象模型**:

| 标准化对象 | 解决的问题 | 效果 |
|-----------|-----------|------|
| **Resources 统一模型** | 知识库/文件/数据的访问接口各不相同 | 一套 URI 体系(viking://、kb://),Agent 无需关心底层存储 |
| **MCP 统一接入** | 各工具/数据源的接入方式碎片化 | 一个配置接入所有 MCP server,Agent 自动发现工具 |
| **ACP 统一编辑器协议** | 不同 IDE/客户端各自对接 Agent | 一份 Agent 定义,Zed/JetBrains/OpenCode 等全端接入 |
| **Capability 统一能力注册** | Agent 能力定义不统一,跨团队复用困难 | entry-point 插件发现机制,能力可跨项目复用 |
| **标准工作流体系** | 多 Agent 协作流程缺乏规范,每次重新设计 | teams(并行/串行)+ graph(DAG)统一编排模型 |

**这套标准化体系是闭门造车做不到的**——它来源于对 MCP/ACP/AG-UI 等国际协议的深度跟踪与逐文件同步,来源于对 PydanticAI V2 路线的对齐,来源于多个业务场景验证中暴露的真实问题。**提前建立标准,是为了让后续每一条业务线的 Agent 开发都不用重新对齐、重新沟通、重新造轮子。**

### 2.5 能为三一带来什么(核心价值)

| 价值项 | 说明 | 参照 |
|--------|------|------|
| **省重复建设** | 各业务线 Agent 各自开发 → 1 套统一平台,节省 N× 人力 | 西门子:非差异化软件降本 40% |
| **提售后效率** | Agent 自主诊断→维修方案→配件→工单闭环 | 易维讯已有 50% 预诊断基础,Agent 化后再提升 |
| **知识不随人走** | "老师傅"经验固化为数字员工 | 130 万条工业知识已有基础 |
| **后市场竞争** | 智能运维作为增值服务;设备数据变现 | 后市场 3000 亿;卡特彼勒服务营收 190 亿美元 |
| **开发提效** | YAML 声明式定义,业务线只写配置不写框架 | 三一自述 AI 数字员工使交付效率 +50% |

### 2.6 三一后市场 Agent 化机遇

三一已有丰富的后市场数字化基础,WolfHarness 可直接赋能:

| 现有能力 | Agent 化升级 |
|---------|-------------|
| 易维讯(EVI):98% 服务线上化、50% 故障预诊断 | Agent 自主诊断→维修方案→配件识别→工单闭环 |
| ECC 智能调度:出发前收到故障预测 | Agent 驱动预测性维护 + 智能排程 |
| AI 维修助手(2025 已落地) | 从单点助手升级为多 Agent 协同(诊断 + 知识库 + 配件) |
| 130 万条工业知识 | 以 MCP server 接入,Agent 可直接调用 |
| 110 万台联网设备 | 设备数据以 MCP server 接入,Agent 实时感知设备状态 |

### 2.7 可拓展的业务场景

| 场景 | 落地方式 | 市场参照 |
|------|---------|---------|
| 工业设备智能运维 | 知识库 + 诊断 Agent(MCP 接入工单/传感器) | Agentic AI 市场 $7.8B→$52B |
| 企业内部统一 Agent 平台 | 一份 YAML,各业务线共用,多端接入 | LangGraph 400+ 企业 |
| AI 编程助手/IDE 集成 | ACP 协议(JetBrains/Zed) | Claude Code ARR $25 亿 |
| 智能客服/知识问答 | 多 Agent 团队 + MCP 工具 + 会话存储 | Klarna 年省 $60M |

### 2.8 与主流 Agent 框架的架构对比(2026 年中)

**先澄清一个关键认知:WolfHarness 不是在和 LangGraph/CrewAI "卷同一个位置"。** 框架层(编排引擎)已收敛、竞争充分;WolfHarness 占据的是它们之上的**"多协议桥接中间件"层**——把异构 Agent 用一份 YAML 定义、以五协议对外暴露。下表按真实架构能力逐项对比,数据来源为 RFC-0051 八框架横评、RFC-0050 架构对比与本报告的业界对标(§4.4)。

| 维度 | LangGraph | CrewAI | AutoGen/MS Agent Framework | OpenAI Agents SDK | Claude Agent SDK / Claude Code | Google ADK | PydanticAI | **WolfHarness** |
|------|-----------|--------|---------------------------|-------------------|------------------------------|-----------|------------|------------------|
| **定位** | 图编排引擎 | 角色协作编排 | 多 Agent 协作研究 | 官方 Agent SDK | 官方 Agent SDK(闭源 CLI) | 官方 Agent SDK | Agent 构建基座 | **多协议编排中间件** |
| **Agent 定义** | 代码为主 | 代码为主(+部分 YAML) | 代码为主 | 代码为主 | 代码为主 | 代码为主 | 代码为主 | **YAML 声明式,一次定义** |
| **对外协议暴露** | 无(仅内部 API) | 无 | 无 | 无(仅 SDK) | 无(仅 Anthropic 生态) | 无 | 无 | **五协议:ACP/MCP/AG-UI/OpenCode/OpenAI API** |
| **异构 Agent 同池** | ✗ 仅自身 Native | ✗ | ✗ | ✗ | ✗ 仅 Anthropic | ✗ | ✗ | **✓ native+ACP+Claude Code+AG-UI 四类同池编排** |
| **模型中立** | ✓ | ✓ | ✓ | ✗ OpenAI 中心 | ✗ 仅 Anthropic | ✗ Google 中心 | ✓ | **✓ 8 类模型配置 + 跨厂商 fallback 链,零硬编码** |
| **统一扩展抽象** | ✗ 无插件系统 | ✗ | ✗ | ✗ | ✗ | ✗ | △ Capability 单一层 | **✓ Capability 统一注册(tools/skills/commands/MCP)** |
| **开源/许可** | Apache-2.0 | MIT | MIT | MIT | **闭源/专有** | Apache-2.0 | MIT | MIT(上游)+ 自有拓展 |
| **企业生产验证** | 400+ 企业 | 有 | 研究为主 | 有 | 有(Anthropic 全家桶) | 有 | 有 | 已在三一 6 类业务场景落地 |

**架构优势解读(逐条对应表格):**

1. **唯一"五协议同时对外暴露"的框架。** 其余框架要么只做协议客户端(消费 MCP 工具),要么只服务自家生态。WolfHarness 是服务端思维:一份 YAML,ACP/MCP/AG-UI/OpenCode/OpenAI API 五端同时对外。这是"中间件"与"编排库"的本质区别——**LangGraph 解决的是"Agent 内部怎么编排",WolfHarness 解决的是"Agent 怎么被全世界访问"。**

2. **唯一支持异构 Agent 同池编排。** 其他框架只能编排自己的 Native Agent;WolfHarness 能把 native、ACP(Goose/第三方 ACP server)、Claude Code、AG-UI 四类异构 Agent 放进同一个池里互相 delegation。**这是企业落地最稀缺的能力:不要求你替换已有 Agent 投资,而是把它们统一纳管。**

3. **声明式配置 + 多协议同时打通,业内只有我们。** Codex/Claude Code/CrewAI/MS Agent Framework/PydanticAI V2 都支持 YAML,但只覆盖"定义 Agent"这一步;WolfHarness 的 YAML 从 Agent、工具、团队、协议暴露到会话存储全链路声明式。RFC-0051 八框架横评结论:所有被调研框架均缺乏统一扩展抽象,AgentPool(WolfHarness)的 Capability 统一注册领先于全部八家。

4. **零厂商锁定,适配三一数据主权要求。** 巨头 SDK 全部绑定自家模型生态;WolfHarness 模型层中立、可私有部署(接 vLLM/Ollama,数据不出企业),这直接对应三一 110 万台联网设备的数据合规底线。

5. **工程化程度对标商业产品。** mypy --strict 全绿、662 测试文件四层金字塔、OpenSpec 变更管控、Logfire 全链路插桩——不是实验室原型,而是可维护、可交接、可审计的工程资产。

> **一句话:框架层拼"算法优雅度"(已收敛),中间件层拼"协议覆盖 × 异构能力 × 声明式全链路"。WolfHarness 在中间件层没有直接对标物——这既是风险(无人验证),也是最大的生态位机会(先发定义标准)。**

---

## 三、安全与风险

### 3.1 信息安全风险与缓解

| 风险类别 | 具体风险 | 等级 | 缓解措施 |
|----------|---------|------|---------|
| 供应链 | 依赖被投毒 | 中 | uv.lock 全量锁定、48 个全主流成熟依赖 |
| 密钥泄露 | API key 意外提交 | 中 | 集中 auth 模块、pre-commit 扫描、开源前 gitleaks 扫描 |
| 提示注入 | 恶意输入操纵 Agent | 高 | 工具参数净化、人工审批、delegation 深度上限 |
| 工具越权 | Agent 调用未授权工具 | 高 | 白名单裁剪、URI 前缀白名单、权限协商、沙箱执行 |
| 数据外泄 | 敏感数据发送给模型供应商 | 高 | 私有部署(接 vLLM/Ollama)、模型路由可控、日志可脱敏 |
| 许可/专利 | 开源后专利纠纷 | 低 | MIT 许可(最宽松)、fork 上游已获许可 |

### 3.2 项目已有的安全能力(工程化默认安全)

- **人工审批(HITL)**:工具层 ApprovalRequired 异常,敏感操作需人工确认
- **权限协商**:ACP request_permission 完整实现
- **输入净化**:tool_arg_sanitize 清洗非法 JSON 参数
- **失控防护**:loop_detection(深度上限)、tool_output_budget
- **沙箱执行**:ExecuteCodeTool 用 exxec 隔离 Python 执行
- **URI 白名单**:viking capability 限制知识库访问范围
- **SSRF 防护**:图片规范化器规避 file:// 直通
- **审计追溯**:28 个文件 Logfire 插桩,工具调用参数写入 span
- **会话隔离**:每 session 独立 + journal/snapshot 双写崩溃恢复
- **测试验证**:36 个 security marker 测试 + ALLOW_MODEL_REQUESTS 双重门闩

### 3.3 数据/隐私合规

- WolfHarness 支持**私有部署**(OpenAI-compatible endpoint 接 vLLM/Ollama)——可做到数据不出企业
- 会话存储可配置加密与留存策略

### 3.4 "未来框架会成熟"的应对

- 框架层会收敛,但**统一编排基建的需求不变**——协议越多越需要桥接
- YAML 声明式定义是**跨框架可移植的**(业界已有 Agent Spec 标准)
- 协议实现(ACP/MCP)独立可迁移;即使更换底座,迁移成本远低于从零开发
- 团队积累的是"对 Agent 协议生态的深度理解",不是一份代码

---

## 四、成本与对比

### 4.1 已投入成本

| 维度 | 数据 |
|------|------|
| 提交规模 | **9,415 次提交**(2024-12 至 2026-08,约 20 个月) |
| 团队贡献 | 上游 Philipp(7,000)+ 团队 Leoyzen(1,857)+ iroot-llm 成员 500+ |
| 代码体量 | **17.5 万行 Python**,662 个测试文件 |
| 工程资产 | mypy --strict 全绿、49 个 specs、import-linter 架构合同 |

**这不是"要不要投入"的问题,而是"已投入的成本如何在战略上兑现"的问题。**

### 4.2 不统一管理的成本

- **Klarna** 基于 agent 自动化,公开报道每年节约约 **$60M**
- **OSSA 报告**:"每花在非标准 Agent 配置上的一美元都是复利技术债"
- Agent 市场 2026 年 $7.8B → 2030 年 $52B(CAGR 46%),碎片化基座越大,迁移纠正成本越高

### 4.3 开源是降低维护成本的手段(不是目的)

| 维护方式 | 成本结构 | 适合三一吗 |
|---------|---------|-----------|
| 纯内部维护 | 全部成本独自承担,随协议演进递增 | 短期可行,长期负担重 |
| **开源共建**(推荐) | 前期投入后社区分担维护 | 降低长期 TCO;附带 AI 人才招聘效应 |
| 购买商业框架 | 持续 license 费 + 厂商锁定 | 三一已有 17.5 万行自有资产,无需从零买 |

### 4.4 业界进展对标

**协议层已分层定型**(2026 年共识):

| 协议层 | 标准 | 规模 |
|--------|------|------|
| Agent → 工具 | MCP(Anthropic) | 月下载超 4 亿;10,000+ 公共 server;2025-12 捐赠 Linux Foundation |
| Agent → Agent | A2A(Google) | 25.3k stars;150+ 支持组织 |
| Agent → 用户 | AG-UI(CopilotKit) | 已集成 LangGraph/CrewAI/PydanticAI 等 |
| Editor ↔ Agent | ACP(Google+Zed) | 4,004 stars;Zed/JetBrains/Neovim 已集成 |

**框架层已收敛**:LangGraph(33.9k stars)、OpenAI SDK(26.9k)、CrewAI(52.8k)——但**没有主流框架覆盖"多协议桥接中间件"层**,这正是 WolfHarness 的位置。

**声明式配置已成共识**:Codex、Claude Code、CrewAI、Microsoft Agent Framework、PydanticAI V2 均支持 YAML 定义 Agent——WolfHarness 是少数把声明式与多协议同时打通的框架。

### 4.5 开源的成本结构

| 成本项 | 闭源 | 开源后 | 净变化 |
|--------|------|--------|--------|
| 维护成本 | 团队独自承担 | 社区分担(参考 OpenCode 950+ 贡献者) | ↓ |
| 协议跟进 | 团队独自跟踪 | 社区 + 生态反馈驱动 | ↓ |
| 获客/分发 | 需自建渠道 | GitHub/生态目录自带流量 | ↓ |
| 商业变现 | 可售 license | 改用企业版模式(托管/认证/SLA) | 等价,延迟兑现 |
| 标准地位 | 无 | 可能获得生态位先发 | ↑ 战略性收益 |

### 4.6 为什么开源:这不是"情怀",是中间件生态位的结构性要求

领导核心关切:**"为什么要开源?从开源能获取什么?闭源不行吗?"** 本节正面回答。核心结论先行:**对"多协议桥接中间件"这一类软件,开源不是可选项,而是生态位的入场券**——闭源等于主动退出这场竞争,而不是保守。

#### 4.6.1 为什么开源:四个底层逻辑

**① 开源是对抗"生态碎片化"的唯一筹码。** 2026 年行业共识(AAIF 成立公告原文):"若缺乏共同约定与中立治理,Agent 开发将分化成互不兼容的孤岛。" 巨头们(Anthropic/OpenAI/Google/微软)已把 MCP、AGENTS.md、A2A 全部捐给 Linux Foundation——**他们放弃了私有控制,换取整个产业的采用**。一个 Agent 中间件若闭源,恰好站到了这个趋势的对立面:既无法参与标准治理,也无法被生态集成。

**② "放弃控制权换采用率"已被验证是赢家策略。** Anthropic 把月下载 9,700 万次的 MCP 捐给 Linux Foundation 旗下的 AAIF:官方表述是"放弃对标准的私有控制,换取大幅加速的采用率""企业不会在单一厂商控制的协议上建构关键基础设施"。结果:AAIF 成立 2 个月内新增 97 家会员(18 家黄金 + 79 家白银)。**这是"中间件靠采用率活,不靠控制权活"的最强证据。**

**③ 开源是基础设施类软件的信任货币。** 企业采购决策者越来越警惕厂商锁定。一个可审计、可自托管、可 fork 的中间件,才能让外部企业放心采用;闭源中间件在采购流程里直接出局(参照 OpenCode 企业客户 Cloudflare 明确以"无供应商锁定"为采购理由)。§3 已论证安全能力不因开源而降低(参照 MCP 开源后安全性反而提升)。

**④ 时机窗口:生态位先发者定义标准。** 框架层已收敛(§4.4),中间件层尚无主流框架占据——这个窗口正在收窄(2028 年前后 ACP v2/MCP 成熟)。MCP 在 2024-11 开源时也"不完善",但先发让它成了事实标准。**现在开源,定义的是"多协议中间件"的标准;晚两年开源,只是在追赶别人的标准。**

#### 4.6.2 从开源能获取什么(收益清单)

| 收益 | 机制 | 硬证据/参照 |
|------|------|------------|
| **① 维护成本社区化** | 代码、文档、协议移植由社区分担 | OpenCode 900+ 贡献者、13,000+ commits;950+ 贡献者持续维护 |
| **② 生态位先发与标准话语权** | 成为"多协议中间件"的第一个开源实现,参与 ACP/MCP 演进 | MCP 捐赠后 2 个月 97 家企业加入 AAIF;标准=咽喉要道 |
| **③ 获客/分发成本趋零** | GitHub/生态目录自带流量,无需销售团队 | OpenCode 一年 172k★、8M MAU,预估年收入 $25M |
| **④ AI 人才招聘效应** | 开源项目是 AI 工程师的首选标签 | OpenCode/Anthropic/OpenAI 均以开源项目吸引顶尖人才 |
| **⑤ 质量与安全反馈回路** | 真实用户场景驱动 bug 修复、安全研究、边缘用例覆盖 | 662 测试 → 社区补充测试;开源审计提升可信度 |
| **⑥ 企业信任与合规背书** | 可审计、可自托管、无锁定 → 外部企业敢用 | Cloudflare 以"无供应商锁定"采用 OpenCode;air-gapped 部署被支持 |
| **⑦ 商业变现基础**(不冲突) | 开源核心 + 企业版(托管/认证/SLA/私有部署) | OpenCode MIT 核心 + Go/Black 付费档;Codex CLI 开源引流 API |

**关键点:开源与商业化不矛盾,反而是商业化杠杆。** 参照 OpenCode(MIT 核心,一年做到预估 $25M ARR)、Databricks(Apache Spark 开源,商业版收入百亿级)。中间件的钱不在"卖代码",在"卖服务/卖标准地位/卖生态集成"。

#### 4.6.3 为什么闭源不行:三个结构性原因

| 闭源的理由 | 为什么在"中间件"上不成立 |
|-----------|--------------------------|
| "代码是核心资产,怕被抄" | 中间件的护城河不在代码,在**协议保真度(ACP 哈希锁定)、生态集成、标准话语权、演进速度**。代码可被重写,生态位不能。 |
| "闭源可以卖 license" | 中间件 license 天花板极低:客户不会为"桥接层"付大钱(参照 §5.4 结论);而企业版服务/生态地位的价值大一个量级。 |
| "闭源更安全" | 恰恰相反:安全靠工程化默认安全(§3.2 十二项)+ 开源审计 + 漏洞披露流程。MCP 开源后安全性提升、OpenCode 开源后由社区修复高危漏洞,均为反例。 |
| "开源被白嫖" | 中间件的收益模型是"被采用才值钱"。被 fork 不等于失去价值——上游 phil65/agentpool 被我们 fork 后,我们贡献的回流与生态地位反而证明了开源的正和博弈。 |

**闭源在三种情况下成立**:终端产品(Claude Code/Codex CLI——绑定自家模型卖用量)、独占数据(工业知识库)、核心算法专利。**WolfHarness 三者都不是**:它是纯工程资产,上游本就 MIT 开源,不含工业数据(§1.3)。**闭源唯一的结果是:17.5 万行工程资产退回纯内部工具,失去生态位窗口——这是对已投入 20 个月成本的最大浪费。**

---

## 五、后续建议

### 5.1 建议考虑将 WolfHarness 纳入公司正式管理

供领导决策参考:

1. 将项目纳入公司软件平台建设规划,按公司制度完成代码托管与发布审批
2. 以公司名义继续投入,既确保全程合规,又能让存量投入发挥更大价值
3. 开源作为降低维护成本的手段(如获批准),附带 AI 人才招聘效应

### 5.2 实施路径

| 阶段 | 周期 | 交付物 | 检查点 |
|------|------|--------|--------|
| **转正与合规** | 0-1 月 | 仓库处理完成;自查报告;安全/法务确认;评估通过 | 评审通过 |
| **内部平台化** | 1-3 月 | 统一 Agent 平台 v1;2-3 个业务线试点(诊断/客服/研发) | 试点验证 |
| **能力增强** | 3-6 月 | ACP v2 支持;安全测试扩充(OWASP Agentic Top 10 覆盖) | 协议同步 |
| **开源/商业化**(获批前提下) | 6-12 月 | 以公司名义发布;企业版商业模式验证 | 生态反馈 |

### 5.3 资源配置

- **核心团队**:2-3 名专职(框架核心 + 协议同步 + 内部平台运营)
- **投入量级**:远低于从零搭建同等框架的成本(存量 17.5 万行资产已就绪)
- **退出机制**:若评估不适合开源,转入纯内部平台,存量投入持续产生内部价值,无沉没风险

### 5.4 领导可能关心的问题(精选)

| 问题 | 应对要点 |
|------|---------|
| 和 LangGraph/CrewAI 比优势在哪? | 不在同一层竞争:它们拼框架层编排,我们拼中间件层协议覆盖(§2.8)。五协议服务端 + 异构同池 + 声明式全链路,目前无直接对标物 |
| 不闭源卖吗? | 中间件卖 license 天花板极低;开源→生态位先发 + 获客趋零 + 企业版变现(参照 OpenCode MIT 核心一年 $25M ARR) |
| 别人白嫖怎么办? | 竞争力不在代码,在协议保真度(哈希锁定的工程壁垒)+ 生态集成 + 标准话语权 + 持续演进能力(§4.6.3) |
| 现在开源太早? | 标准窗口 2028 年收窄;MCP 在 2024-11 就开源了,那时也"不完善"。先发者定义标准(§4.6.1④) |
| 未来框架成熟了怎么办? | 框架层会收敛,但中间件/桥接层价值反升;YAML 资产可迁移;团队沉淀的是能力(§4.6.3) |
| 开源后安全谁负责? | 已有工程化默认安全(12 项);开源后增设 SECURITY.md + 漏洞披露流程;参照 MCP 开源后安全性反而提升(§4.6.2⑤) |

---

## 结论

三一已全面智能化(700+ 场景、年创效 2 亿+),但各业务线 Agent 各自开发、缺统一基建。WolfHarness 是现成的 Agent 编排底座(17.5 万行、五协议、不含工业数据),能直接省重复建设成本、提售后效率、固化管理经验。**核心价值是省钱、提效、攒竞争力。**

**关于开源**:开源不是"让维护更便宜"的锦上添花,而是"多协议中间件"这一生态位**必须走的路**(§4.6)——它是唯一可能让三一从"Agent 框架使用者"变成"Agent 生态标准定义者"的路径:社区分担维护、生态位先发、人才吸附、企业信任,而商业价值可通过企业版兑现(参照 OpenCode/MCP 先例)。**闭源的唯一后果是让 20 个月的工程投入退回内部工具,并永久错过 2028 年前的标准窗口。** 建议:内部平台化先行,开源作为公司级战略选项在获批后推进,两条路都不浪费存量投入。

---

## 附录:数据来源

- MCP 官方博客(2026-07-28):https://blog.modelcontextprotocol.io/posts/2026-07-28/
- Anthropic MCP 捐赠公告(2025-12):https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation
- Linux Foundation AAIF 成立公告:https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation
- Agentic AI Foundation 深度解读(为什么巨头联手开放标准,2026-04):https://faq.com.tw/zh/developer-tools/2026-04-06-agentic-ai-foundation-zh/
- The Verge(MCP 标准战争):https://www.theverge.com/ai-artificial-intelligence/841156/ai-companies-aaif-anthropic-mcp-model-context-protocol
- ACP 官网:https://agentclientprotocol.com/
- The Register(Google/Zed ACP 报道):https://www.theregister.com/2025/08/28/google_zed_acp/
- JetBrains ACP 支持公告:https://blog.jetbrains.com/ai/2025/12/bring-your-own-ai-agent-to-jetbrains-ides/
- AG-UI 发布博客:https://www.copilotkit.ai/blog/introducing-ag-ui-the-protocol-where-agents-meet-users/
- A2A GitHub:https://github.com/a2aproject/A2A/
- PydanticAI GitHub:https://github.com/pydantic/pydantic-ai/
- OpenCode GitHub:https://github.com/anomalyco/opencode
- OpenCode 增长数据(172k★/8M MAU/$25M ARR 预估,2026-06):https://awesomeagents.ai/news/opencode-8m-users-one-year/
- openai/codex:https://github.com/openai/codex
- Sacra(Claude Code ARR):https://sacra.com/c/anthropic/
- OSSA Agentic AI 标准缺口研究:https://openstandardagents.org/research/agentic-ai-market-standards-gap/
- The Agent Report(框架对比 2026-07):https://the-agent-report.com/2026/07/ai-agent-frameworks-comparison-2026-langgraph-crewai-autogen/
- 企业开源捐赠背后的商业逻辑(生态战略分析):http://www.cf-1.com/news/2304542/
- 三一集团官网(人工智能+/灯塔工厂):https://m.sanygroup.com/news/14887.html
- 三一重工 2024 年报(研发投入 54.88 亿):http://static.cninfo.com.cn/finalpage/2025-04-18/1223129214.PDF
- 全国工商联(三一数字化投入超 200 亿):https://www.acfic.org.cn/ztzlhz/szhzxxd/szhzxxd_al/202401/t20240125_315665.html
- 三一 CIO 许国强访谈:https://cio.zhiding.cn/cio/2026/0612/3190359.shtml
- 铁甲工程机械网(三一 AI 大赛):https://www.cehome.com/news/20251017/359349.shtml
- WEF 2025 全球灯塔网络公报:https://cn.weforum.org/press/2025/01/global-lighthouse-network-2025-world-economic-forum-recognizes-companies-transforming-manufacturing-through-innovation-cn/
- 海尔卡奥斯开源:https://www.haier.com/press-events/news/20221116_203257.shtml
- 西门子开源战略:https://opensource.siemens.com/
- Eclipse SDV(非差异化软件降本 40%):https://newsroom.eclipse.org/news/announcements/automotive-innovation-through-open-collaboration-momentum-builds-around-open
- Global Market Insights(建筑设备诊断市场):https://www.gminsights.com/zh/industry-analysis/construction-equipment-diagnostics-market

> 注:Klarna"每年节约约 $60M"为公开报道口径;"生产级 agent 自研成本为多团队人年"为行业内通用认知估算,汇报前可进一步核定引用。框架对比中 LangGraph/CrewAI/OpenAI SDK 星标与协议支持情况以 2026 年中公开资料为准。