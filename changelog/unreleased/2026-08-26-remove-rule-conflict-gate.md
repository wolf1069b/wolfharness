# 移除物化规则冲突门禁，冲突判定交由 LLM

`_mark_merge_conflict` 不再在物化合并时自动建 OPA 或设置 conflict_pending
阻塞发布；原正则数值槽提取(`_PARAMETER_RE`)将故障代码编号(H-1~H-24)、章节号、
型号编号、线号等标识符误判为参数数值冲突（本轮构建 65/65 误报）。冲突判定改由
物化 LLM worker 承担——其已有 create_opa 权限且 prompt 已指示对真实事实矛盾建
fact_conflict OPA。
