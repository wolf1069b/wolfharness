"""OpenCode-compatible API server.

This module provides a FastAPI-based server that implements the OpenCode API,
allowing OpenCode SDK clients to interact with AgentPool agents.

Example usage:

    from wolfharness import AgentPool
    from wolfharness_server.opencode_server import OpenCodeServer

    async with AgentPool("config.yml") as pool:
        assert pool.session_pool is not None
        agent = await pool.session_pool.sessions.get_or_create_session_agent(
            "opencode-main", pool.main_agent_name
        )
        server = OpenCodeServer(agent, port=4096)
        await server.run_async()

Or programmatically:

    from wolfharness_server.opencode_server import create_app

    app = create_app(agent=my_agent, working_dir="/path/to/project")
    # Use with uvicorn or other ASGI server
"""

__all__ = []
