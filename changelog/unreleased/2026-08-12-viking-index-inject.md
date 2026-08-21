# Dynamic resource-namespace index injection

Added a dynamic `<openviking-index>` injection to `VikingCapability`,
cloning the existing profile-inject pattern. Four new config fields
(`index_enabled`, `index_max_tokens`, `index_limit`, `index_uri`) drive a
first-turn handler that lists live `viking://resources/<namespace>`
namespaces via the Viking SDK and injects them into the system prompt —
eliminating hard-coded namespace knowledge. The block format is a contract
consumed by the xeno-adp-agentic prompt layer; it is disabled by default.
