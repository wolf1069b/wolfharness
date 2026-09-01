# ACP server startup failures now surface on stderr

`wolfharness serve-acp` redirected all logging to `~/Library/Logs/wolfharness/acp.log` (or `~/.local/state/wolfharness/acp.log`) and swallowed startup exceptions: serve-loop errors in `ACPServer._start_async` were never re-raised (contradicting its `raise_exceptions=True` setup), and the CLI handler exited with a silent `typer.Exit(1)`. A configuration that passed early validation but failed at agent or transport startup therefore exited with no output on the terminal.

- `ACPServer._start_async` now re-raises serve errors when `raise_exceptions` is set, consistent with `BaseServer.start`.
- `serve_acp` now prints the exception type/message and the log file path to stderr before exiting with code 1.