# Use the stable wiki build service entry point

Updated `WikiBuildCapability` to lazily import the host writer from the stable
`xeno_adp_agentic.wiki.serve.build_tools` facade. The production capability
continues to execute in process and no longer couples AgentPool to the host's
legacy `mcp_server` module name. Public configuration and tool behavior are
unchanged.
