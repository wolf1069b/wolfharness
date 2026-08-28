# Expose batched wiki entity materialization

Added `write_entities_batch` to the in-process wiki build capability's formal
write surface so extraction workers can validate and commit a bounded group of
new entities in one host call. Role filtering and write-tool classification
remain unchanged for existing tools.
