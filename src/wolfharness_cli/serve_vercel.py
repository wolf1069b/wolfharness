"""Command for serving agents via Vercel AI protocol."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import TYPE_CHECKING, Annotated, Any

import anyenv
import typer as t

from wolfharness_cli import log, resolve_agent_config


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness import ChatMessage


logger = log.get_logger(__name__)


def vercel_command(  # noqa: PLR0915
    ctx: t.Context,
    config: Annotated[str | None, t.Argument(help="Path to agent configuration")] = None,
    agent_name: Annotated[
        str | None, t.Option("--agent", "-a", help="Specific agent to serve")
    ] = None,
    host: Annotated[str, t.Option(help="Host to bind server to")] = "localhost",
    port: Annotated[int, t.Option(help="Port to listen on")] = 8000,
    cors: Annotated[bool, t.Option(help="Enable CORS")] = True,
    show_messages: Annotated[
        bool, t.Option("--show-messages", help="Show message activity")
    ] = False,
) -> None:
    """Serve agents via Vercel AI Data Stream Protocol.

    This creates a server compatible with Vercel AI SDK frontends,
    allowing you to use your agents with Vercel AI UI components.

    The server exposes a POST /chat endpoint that accepts Vercel AI
    protocol requests and streams responses back.

    If --agent is specified, only that agent is served. Otherwise,
    the endpoint accepts an 'agent' field in the request to select
    which agent to use.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response, StreamingResponse
    import uvicorn

    from wolfharness import AgentPool, AgentsManifest
    from wolfharness.agents.events import StreamCompleteEvent, UserMessageInsertedEvent
    from wolfharness_config.context import ConfigContextManager

    logger.info("Server PID", pid=os.getpid())

    def on_message(message: ChatMessage[Any]) -> None:
        print(message.format(style="simple"))

    try:
        config_path = resolve_agent_config(config)
    except ValueError as e:
        msg = str(e)
        raise t.BadParameter(msg) from e

    with ConfigContextManager(config_path):
        manifest = AgentsManifest.from_file(config_path)
    pool = AgentPool(manifest, main_agent_name=agent_name)

    # show_messages is disabled: agent instances are no longer created at pool level.
    # Session-level event monitoring is available via EventBus instead.

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await pool.__aenter__()
        logger.info("Agent pool initialized")
        try:
            yield
        finally:
            await pool.__aexit__(None, None, None)
            logger.info("Agent pool shut down")

    # Create FastAPI app
    app = FastAPI(
        title="AgentPool - Vercel AI Server",
        description="Vercel AI Data Stream Protocol server for AgentPool",
        lifespan=lifespan,
    )

    if cors:
        app.add_middleware(
            CORSMiddleware,  # ty: ignore[invalid-argument-type]
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.post("/chat")
    async def chat(request: Request) -> Response:  # noqa: PLR0915
        """Handle Vercel AI protocol chat requests.

        Implements the Vercel AI Data Stream Protocol:
        https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol#data-stream-protocol
        """
        body = await request.body()
        try:
            data = anyenv.load_json(body, return_type=dict)
        except anyenv.JsonLoadError as e:
            return JSONResponse({"error": f"Invalid JSON: {e}"}, status_code=400)

        # Extract messages from the request
        messages = data.get("messages", [])
        if not messages:
            return JSONResponse({"error": "No messages provided"}, status_code=400)

        # Get the last user message
        last_message = messages[-1]
        user_text = ""
        if last_message.get("role") == "user":
            parts = last_message.get("parts", [])
            for part in parts:
                if part.get("type") == "text":
                    user_text = part.get("text", "")
                    break

        if not user_text:
            return JSONResponse({"error": "No user text found"}, status_code=400)

        # Determine which agent to use and create a per-request session
        # Vercel protocol is stateless — new session per HTTP request
        effective_agent_name = agent_name or pool.main_agent_name
        from wolfharness.utils.identifiers import generate_session_id

        session_id = generate_session_id()
        session_pool = pool.session_pool
        assert session_pool is not None, "SessionPool must be initialized"
        await session_pool.create_session(session_id, agent_name=effective_agent_name)

        async def generate_stream() -> AsyncIterator[str]:
            """Generate Vercel AI Data Stream Protocol events.

            Protocol format:
            - Text: 0:"text content"
            - Finish: e:{"finishReason":"stop",...}
            - Done: d:{"finishReason":"stop"}
            """
            try:
                async for event in session_pool.run_stream(session_id, user_text):
                    # Handle pydantic-ai streaming events
                    match event:
                        case PartStartEvent(part=TextPart() as part):
                            # New part started - if it's text, emit it
                            if part.content:
                                text = part.content
                                escaped = anyenv.dump_json(text)
                                yield f"0:{escaped}\n"
                        case PartDeltaEvent(delta=TextPartDelta(content_delta=content_delta)):
                            # Delta update - emit text deltas
                            if content_delta:
                                escaped = anyenv.dump_json(content_delta)
                                yield f"0:{escaped}\n"
                        case StreamCompleteEvent():
                            # Stream complete - we've received the final message
                            # The content has already been streamed via deltas
                            pass
                        case UserMessageInsertedEvent():
                            pass  # User message insertions not streamed via Vercel protocol

                # Send finish event
                usage = {"promptTokens": 0, "completionTokens": 0}
                finish_data = {"finishReason": "stop", "usage": usage}
                yield f"e:{anyenv.dump_json(finish_data)}\n"

                # Send done marker
                done_data = {"finishReason": "stop"}
                yield f"d:{anyenv.dump_json(done_data)}\n"

            except Exception as e:
                logger.exception("Error during streaming")
                # Send error as text
                error_msg = f"Error: {e}"
                escaped = anyenv.dump_json(error_msg)
                yield f"0:{escaped}\n"
                # Still send finish
                finish_data = {"finishReason": "error"}
                yield f"e:{anyenv.dump_json(finish_data)}\n"
                done_data = {"finishReason": "error"}
                yield f"d:{anyenv.dump_json(done_data)}\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain; charset=utf-8",
            headers={
                "X-Vercel-AI-Data-Stream": "v1",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.get("/agents")
    async def list_agents() -> dict[str, Any]:
        """List available agents."""
        return {
            "agents": [
                {"name": name, "description": config.description}
                for name, config in pool.manifest.agents.items()
            ]
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    # Get log level from the global context
    log_level = ctx.obj.get("log_level", "info") if ctx.obj else "info"

    print(f"Starting Vercel AI server on http://{host}:{port}")
    print(f"Chat endpoint: POST http://{host}:{port}/chat")
    print(f"Available agents: {list(pool.manifest.agents.keys())}")

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


if __name__ == "__main__":
    import typer

    typer.run(vercel_command)
