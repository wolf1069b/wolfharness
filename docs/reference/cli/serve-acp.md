---
title: serve-acp
description: Start ACP server
icon: material/server
---

# serve-acp

Start the Agent Communication Protocol (ACP) server to expose agents to ACP-compatible clients.

```bash
wolfharness serve-acp config.yml
```

For basic usage:

```bash
# Start with default config
wolfharness serve-acp agents.yml

# Listen on specific host and port
wolfharness serve-acp agents.yml --host 0.0.0.0 --port 8321
```

For a full list of options, run:

```bash
wolfharness serve-acp --help
```
