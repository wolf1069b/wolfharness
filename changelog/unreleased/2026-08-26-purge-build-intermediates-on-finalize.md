# finalize 成功后自动清理构建中间态

`finalize_wiki` 成功后自动删除 `source_packets/`、`index/chapter_plans/`、
`index/relation_work/`、`index/relation_manifests/` 四类构建中间态。
`materialization_receipts/`（审计回溯）和 `build_checkpoint.json`（完成凭证）保留。

此前中间态无限堆积，增量构建时旧状态干扰新构建（旧 chapter_plans 的
build_id 不匹配、source_packets 全量扫描回退等）。现在 finalize 成功即清，
下次入库自然干净。远程上传失败（`finalized_local`）时不清理，保留中间态
供重试。
