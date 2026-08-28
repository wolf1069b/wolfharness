# Fix silent serve-acp startup failures

`serve-acp` could fail to start (e.g. port already in use) with no terminal
output and exit code 0, because `ACPServer._start_async` swallowed `serve()`
exceptions and the CLI's reconfiguration of logging dropped the stderr handler.
Startup errors now propagate to the CLI, which prints the error to stderr and
exits non-zero; details also still go to `acp.log`.