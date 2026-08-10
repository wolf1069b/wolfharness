"""Storage manager for handling multiple providers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Any, Self

from anyenv import method_spawner
from anyenv.signals import Signal
import anyio
from pydantic import BaseModel, TypeAdapter

from wolfharness.log import get_logger
from wolfharness.messaging import ChatMessage
from wolfharness.sessions.models import PendingDeferredCall
from wolfharness.storage.serialization import deserialize_messages
from wolfharness.utils.identifiers import generate_session_id
from wolfharness_config.session import SessionQuery
from wolfharness_config.storage import StorageConfig


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from types import TracebackType

    from pydantic_ai.messages import ModelMessage

    from wolfharness.common_types import JsonValue
    from wolfharness.sessions.models import ProjectData, SessionData
    from wolfharness_config.storage import BaseStorageProviderConfig
    from wolfharness_storage.base import StorageProvider

logger = get_logger(__name__)


class SessionMetadata(BaseModel):
    """Generated metadata for a conversation."""

    title: str
    """Short descriptive title (3-7 words)."""

    emoji: str
    """Single emoji representing the topic."""

    icon: str
    """Iconify icon name (e.g., 'mdi:code-braces')."""


@dataclass(frozen=True, slots=True)
class SessionMetadataGeneratedEvent:
    """Event emitted when session metadata is generated.

    Attributes:
        session_id: ID of the conversation
        metadata: Generated metadata including title, emoji, and icon
    """

    session_id: str
    metadata: SessionMetadata


class StorageManager:
    """Manages multiple storage providers.

    Handles:
    - Provider initialization and cleanup
    - Message distribution to providers
    - History loading from capable providers
    - Global logging filters

    Signals:
    - metadata_generated: Emitted when session metadata is generated.
      Subscribers receive SessionMetadataGeneratedEvent with session_id and metadata.

    Example:
        manager.metadata_generated.connect(my_handler)
        # Handler will be called with SessionMetadataGeneratedEvent
    """

    # Signal emitted when session metadata is generated
    metadata_generated: Signal[SessionMetadataGeneratedEvent] = Signal()

    def __init__(self, config: StorageConfig | None = None) -> None:
        """Initialize storage manager.

        Args:
            config: Storage configuration including providers and filters
        """
        self.config = config or StorageConfig()
        self.providers = [self._create_provider(cfg) for cfg in self.config.effective_providers]
        self._session_logged: set[str] = set()  # Track logged conversations for idempotency
        self._model_variants: dict[str, Any] = {}  # Set by AgentPool after init

    @staticmethod
    def generate_session_id() -> str:
        """Generate a unique session ID.

        Returns:
            A unique session ID string.
        """
        return generate_session_id()

    async def __aenter__(self) -> Self:
        """Initialize all providers."""
        for provider in self.providers:
            await provider.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up all providers."""
        errors = []
        for provider in self.providers:
            try:
                await provider.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                # Create a more informative error message with provider context
                provider_name = provider.__class__.__name__
                error_msg = f"Error in {provider_name}: {type(e).__name__}: {e}"
                logger.exception(
                    "Error cleaning up provider",
                    provider=provider_name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Wrap error with context for better debugging
                chained_error = RuntimeError(error_msg)
                chained_error.__cause__ = e
                errors.append(chained_error)

        if errors:
            raise ExceptionGroup("Provider cleanup errors", errors)

    def cleanup(self) -> None:
        """Clean up all providers."""
        for provider in self.providers:
            try:
                provider.cleanup()
            except Exception:
                logger.exception("Error cleaning up provider", provider=provider)
        self.providers.clear()

    def _create_provider(self, config: BaseStorageProviderConfig) -> StorageProvider:
        """Create provider instance from configuration."""
        # Extract common settings from BaseStorageProviderConfig
        match self.config.filter_mode:
            case "and" if self.config.agents and config.agents:
                logged_agents: set[str] | None = self.config.agents & config.agents
            case "and":
                # If either is None, use the other; if both None, use None (log all)
                if self.config.agents is None and config.agents is None:
                    logged_agents = None
                else:
                    logged_agents = self.config.agents or config.agents or set()
            case "override":
                logged_agents = config.agents if config.agents is not None else self.config.agents

        provider_config = config.model_copy(
            update={
                "log_messages": config.log_messages and self.config.log_messages,
                "log_sessions": config.log_sessions and self.config.log_sessions,
                "log_commands": config.log_commands and self.config.log_commands,
                "agents": logged_agents,
            }
        )

        return provider_config.get_provider()

    def get_history_provider(self, preferred: str | None = None) -> StorageProvider:
        """Get provider for loading history.

        Args:
            preferred: Optional preferred provider name

        Returns:
            First capable provider based on priority:
            1. Preferred provider if specified and capable
            2. Default provider if specified and capable
            3. First capable provider
            4. Raises error if no capable provider found
        """

        # Function to find capable provider by name
        def find_provider(name: str) -> StorageProvider | None:
            for p in self.providers:
                if p.can_load_history and p.__class__.__name__.lower() == name.lower():
                    return p
            return None

        # Try preferred provider
        if preferred and (provider := find_provider(preferred)):
            return provider

        # Try default provider
        if self.config.default_provider:
            if provider := find_provider(self.config.default_provider):
                return provider
            msg = "Default provider not found or not capable of loading history"
            logger.warning(msg, provider=self.config.default_provider)

        # Find first capable provider
        for provider in self.providers:
            if provider.can_load_history:
                return provider

        raise RuntimeError("No capable provider found for loading history")

    @method_spawner
    async def filter_messages(
        self,
        query: SessionQuery,
        preferred_provider: str | None = None,
    ) -> list[ChatMessage[Any]]:
        """Get messages matching query.

        Args:
            query: Filter criteria
            preferred_provider: Optional preferred provider to use
        """
        provider = self.get_history_provider(preferred_provider)
        return await provider.filter_messages(query)

    @method_spawner
    async def log_message(self, message: ChatMessage[Any]) -> None:
        """Log message to all providers."""
        if not self.config.log_messages:
            return

        for provider in self.providers:
            if provider.should_log_agent(message.name or "no name"):
                await provider.log_message(message=message)

    @method_spawner
    async def log_session(
        self,
        *,
        session_id: str,
        node_name: str,
        start_time: datetime | None = None,
        model: str | None = None,
        initial_prompt: str | None = None,
        parent_session_id: str | None = None,
    ) -> None:
        """Log session to all providers (idempotent).

        If session was already logged, skips provider calls but still
        triggers title generation if initial_prompt is provided.

        Args:
            session_id: Unique session identifier
            node_name: Name of the node/agent
            start_time: Optional start time
            model: Requested model identifier for this session
            initial_prompt: Optional initial prompt to trigger title generation
            parent_session_id: Optional parent session ID
        """
        if not self.config.log_sessions:
            return

        # Check if already logged (idempotent behavior)
        if session_id not in self._session_logged:
            # Mark as logged before calling providers
            self._session_logged.add(session_id)

            # Log to all providers
            for provider in self.providers:
                await provider.log_session(
                    session_id=session_id,
                    node_name=node_name,
                    start_time=start_time,
                    model=model,
                    parent_session_id=parent_session_id,
                )

        # Handle title generation based on prompt length
        # Skip during tests to avoid external API calls
        if not initial_prompt or os.environ.get("PYTEST_CURRENT_TEST"):
            return

        # Generate title from initial prompt
        await self._generate_title_from_prompt(
            session_id=session_id,
            prompt=initial_prompt,
        )

    @method_spawner
    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        """Log command to all providers."""
        if not self.config.log_commands:
            return

        for provider in self.providers:
            await provider.log_command(
                agent_name=agent_name,
                session_id=session_id,
                command=command,
                context_type=context_type,
                metadata=metadata,
            )

    @method_spawner
    async def reset(
        self,
        *,
        agent_name: str | None = None,
        hard: bool = False,
    ) -> tuple[int, int]:
        """Reset storage in all providers concurrently."""

        async def reset_provider(provider: StorageProvider) -> tuple[int, int]:
            try:
                return await provider.reset(agent_name=agent_name, hard=hard)
            except Exception as e:
                cls_name = provider.__class__.__name__
                logger.exception(
                    "Error resetting provider",
                    provider=cls_name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return (0, 0)

        results = await asyncio.gather(*(reset_provider(provider) for provider in self.providers))
        # Return the counts from the last provider (maintaining existing behavior)
        return results[-1] if results else (0, 0)

    @method_spawner
    async def get_session_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        """Get counts from primary provider."""
        provider = self.get_history_provider()
        return await provider.get_session_counts(agent_name=agent_name)

    @method_spawner
    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
        preferred_provider: str | None = None,
    ) -> list[str]:
        """Get command history."""
        if not self.config.log_commands:
            return []

        provider = self.get_history_provider(preferred_provider)
        return await provider.get_commands(
            agent_name=agent_name,
            session_id=session_id,
            limit=limit,
            current_session_only=current_session_only,
        )

    async def save_session(self, data: SessionData) -> None:
        """Save or update session data in the primary provider.

        Args:
            data: Session data to persist
        """
        provider = self.get_project_provider()
        with anyio.CancelScope(shield=True):
            await provider.save_session(data)
        self._session_logged.add(data.session_id)

    @method_spawner
    async def load_session(self, session_id: str) -> SessionData | None:
        """Load session data by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data if found, None otherwise
        """
        provider = self.get_project_provider()
        return await provider.load_session(session_id)

    @method_spawner
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from all providers.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted from at least one provider
        """
        deleted = False
        for provider in self.providers:
            try:
                with anyio.CancelScope(shield=True):
                    if await provider.delete_session(session_id):
                        deleted = True
            except Exception:
                logger.exception(
                    "Error deleting session",
                    provider=provider.__class__.__name__,
                    session_id=session_id,
                )
        # Also clean up any checkpoint data
        await self.delete_checkpoint(session_id)
        return deleted

    @method_spawner
    async def list_session_ids(
        self,
        pool_id: str | None = None,
        agent_name: str | None = None,
        cwd: str | None = None,
    ) -> list[str]:
        """List session IDs, optionally filtered.

        Args:
            pool_id: Filter by pool/manifest ID
            agent_name: Filter by agent name
            cwd: Filter by working directory

        Returns:
            List of session IDs
        """
        provider = self.get_project_provider()
        return await provider.list_session_ids(pool_id=pool_id, agent_name=agent_name, cwd=cwd)

    @method_spawner
    async def load_sessions_batch(
        self,
        session_ids: list[str],
        *,
        agent_name: str | None = None,
    ) -> list[SessionData]:
        """Load multiple sessions by IDs in a single query.

        Delegates to the project provider's batch method to avoid N+1 queries.

        Args:
            session_ids: List of session identifiers to load
            agent_name: Optional filter to return only sessions for this agent

        Returns:
            List of found SessionData objects
        """
        provider = self.get_project_provider()
        return await provider.load_sessions_batch(session_ids, agent_name=agent_name)

    async def update_sdk_session_id(self, session_id: str, sdk_session_id: str) -> None:
        """Update the external SDK session ID for a session.

        Args:
            session_id: Internal session identifier
            sdk_session_id: External SDK session ID
        """
        for provider in self.providers:
            with anyio.CancelScope(shield=True):
                await provider.update_sdk_session_id(session_id, sdk_session_id)

    async def update_session_title(self, session_id: str, title: str) -> None:
        """Update conversation title in all providers.

        Args:
            session_id: ID of conversation to update
            title: New title for conversation
        """
        for provider in self.providers:
            with anyio.CancelScope(shield=True):
                await provider.update_session_title(session_id, title)

    async def get_session_title(self, session_id: str) -> str | None:
        """Get the title of a conversation.

        Args:
            session_id: ID of the conversation

        Returns:
            The conversation title, or None if not set.
        """
        provider = self.get_history_provider()
        return await provider.get_session_title(session_id)

    async def get_session_titles(self, session_ids: list[str]) -> dict[str, str | None]:
        """Get titles for multiple conversations.

        Args:
            session_ids: List of conversation IDs

        Returns:
            Dict mapping session_id to title (or None if not set)
        """
        if not session_ids:
            return {}

        provider = self.get_history_provider()
        titles: dict[str, str | None] = {}
        for conv_id in session_ids:
            try:
                titles[conv_id] = await provider.get_session_title(conv_id)
            except Exception:  # noqa: BLE001
                titles[conv_id] = None
        return titles

    async def get_message_counts(self, session_ids: list[str]) -> dict[str, int]:
        """Get message counts for multiple conversations.

        Args:
            session_ids: List of conversation IDs

        Returns:
            Dict mapping session_id to message count
        """
        if not session_ids:
            return {}

        counts: dict[str, int] = {}
        for conv_id in session_ids:
            try:
                query = SessionQuery(name=conv_id)
                messages = await self.filter_messages(query)
                counts[conv_id] = len(messages) if messages else 0
            except Exception:  # noqa: BLE001
                counts[conv_id] = 0
        return counts

    @method_spawner
    async def get_session_messages(
        self,
        session_id: str,
        *,
        include_ancestors: bool = False,
    ) -> list[ChatMessage[Any]]:
        """Get all messages for a session.

        Args:
            session_id: ID of the conversation
            include_ancestors: If True, also include messages from ancestor
                conversations by following the parent_id chain. Useful for
                forked conversations.

        Returns:
            List of messages ordered by timestamp.
        """
        provider = self.get_history_provider()
        return await provider.get_session_messages(session_id, include_ancestors=include_ancestors)

    @method_spawner
    async def get_message(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> ChatMessage[Any] | None:
        """Get a single message by ID.

        Args:
            message_id: ID of the message
            session_id: When set, only return the message if it belongs to this session.

        Returns:
            The message if found, None otherwise.
        """
        provider = self.get_history_provider()
        return await provider.get_message(message_id, session_id=session_id)

    @method_spawner
    async def get_message_ancestry(
        self,
        message_id: str,
        *,
        session_id: str | None = None,
    ) -> list[ChatMessage[Any]]:
        """Get the ancestry chain of a message.

        Traverses the parent_id chain to build full history leading to this message.

        Args:
            message_id: ID of the message
            session_id: Optional session ID hint for faster lookup

        Returns:
            List of messages from oldest ancestor to the specified message.
        """
        provider = self.get_history_provider()
        return await provider.get_message_ancestry(message_id, session_id=session_id)

    @method_spawner
    async def fork_conversation(
        self,
        *,
        source_session_id: str,
        new_session_id: str,
        fork_from_message_id: str | None = None,
        new_agent_name: str | None = None,
    ) -> str | None:
        """Fork a conversation at a specific point.

        Creates a new conversation that branches from the source. New messages
        in the forked conversation should use the returned fork_point_id as
        their parent_id to maintain the history chain.

        Args:
            source_session_id: ID of the conversation to fork from
            new_session_id: ID for the new forked conversation
            fork_from_message_id: Message ID to fork from. If None, forks from
                the last message.
            new_agent_name: Agent name for the new conversation.

        Returns:
            The message_id of the fork point (use as parent_id for new messages),
            or None if the source conversation is empty.
        """
        provider = self.get_history_provider()
        return await provider.fork_conversation(
            source_session_id=source_session_id,
            new_session_id=new_session_id,
            fork_from_message_id=fork_from_message_id,
            new_agent_name=new_agent_name,
        )

    @method_spawner
    async def truncate_messages(
        self,
        session_id: str,
        up_to_message_id: str,
    ) -> int:
        """Remove all messages after the given message ID.

        Keeps messages up to and including up_to_message_id,
        removes everything after it.

        Args:
            session_id: ID of the conversation to truncate
            up_to_message_id: Keep messages up to and including this ID

        Returns:
            The count of removed messages
        """
        provider = self.get_history_provider()
        return await provider.truncate_messages(session_id, up_to_message_id)

    @method_spawner
    async def delete_session_messages(
        self,
        session_id: str,
    ) -> int:
        """Delete all messages for a session in all providers.

        Used for compaction - removes existing messages so they can be
        replaced with compacted versions.

        Args:
            session_id: ID of the conversation to clear

        Returns:
            Total number of messages deleted across all providers
        """
        total_deleted = 0
        for provider in self.providers:
            try:
                deleted = await provider.delete_session_messages(session_id)
                total_deleted += deleted
            except NotImplementedError:
                # Provider doesn't support deletion (e.g., write-only log)
                pass
            except Exception:
                logger.exception(
                    "Error deleting messages from provider",
                    provider=provider.__class__.__name__,
                    session_id=session_id,
                )
        return total_deleted

    @method_spawner
    async def replace_conversation_messages(
        self,
        session_id: str,
        messages: Sequence[ChatMessage[Any]],
    ) -> tuple[int, int]:
        """Replace all messages for a session with new ones.

        Deletes existing messages and logs new ones. Used for compaction
        where the full history is replaced with a compacted version.

        Args:
            session_id: ID of the conversation
            messages: New messages to store

        Returns:
            Tuple of (deleted_count, added_count)
        """
        # First delete existing messages
        deleted = await self.delete_session_messages(session_id)

        # Then log new messages
        added = 0
        for message in messages:
            # Ensure session_id is set on the message
            msg_to_log: ChatMessage[Any] = message
            if not message.session_id:
                msg_to_log = ChatMessage(
                    content=message.content,
                    role=message.role,
                    name=message.name,
                    session_id=session_id,
                    message_id=message.message_id,
                    parent_id=message.parent_id,
                    model_name=message.model_name,
                    cost_info=message.cost_info,
                    response_time=message.response_time,
                    timestamp=message.timestamp,
                    provider_name=message.provider_name,
                    provider_response_id=message.provider_response_id,
                    messages=message.messages,
                    finish_reason=message.finish_reason,
                )
            await self.log_message(msg_to_log)
            added += 1

        return deleted, added

    async def _generate_title_core(
        self,
        session_id: str,
        prompt_text: str,
    ) -> SessionMetadata | None:
        """Core title generation logic using LLM with structured output.

        Args:
            session_id: ID of the conversation to title
            prompt_text: Formatted prompt text to send to the LLM

        Returns:
            SessionMetadata with title, emoji, and icon, or None if generation fails.
        """
        from wolfharness import Agent
        from wolfharness.utils.model_helpers import infer_model

        logger.info("_generate_title_core called", session_id=session_id)
        if not self.config.title_generation_model:
            logger.info("No title_generation_model configured, skipping")
            return None

        try:
            # Resolve model_variants reference if applicable
            model_str = self.config.title_generation_model
            if model_str in self._model_variants:
                model = self._model_variants[model_str].get_model()
            else:
                model = infer_model(model_str)
            agent = Agent(
                model=model,
                system_prompt=self.config.title_generation_prompt,
                output_type=SessionMetadata,
                retries=3,
            )
            logger.debug("Title generation prompt", prompt_text=prompt_text)
            result = await agent.run(prompt_text)
            metadata = result.data
            # Store the title
            await self.update_session_title(session_id, metadata.title)
            logger.debug("Generated session metadata", session_id=session_id, metadata=metadata)
            # Emit signal for subscribers (e.g., OpenCode UI updates)
            event = SessionMetadataGeneratedEvent(session_id=session_id, metadata=metadata)
            logger.info(
                "Emitting metadata_generated signal",
                session_id=session_id,
                title=metadata.title,
            )
            await self.metadata_generated.emit(event)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to generate session title",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None
        else:
            return metadata

    async def _generate_title_from_prompt(
        self,
        session_id: str,
        prompt: str,
    ) -> str | None:
        """Generate title from initial prompt (internal, fire-and-forget).

        Called automatically by log_session when initial_prompt is provided.
        Persisting the title to storage is handled inside `_generate_title_core`
        (via ``update_session_title``), not by this wrapper.

        Args:
            session_id: ID of the conversation to title
            prompt: The initial user prompt

        Returns:
            The generated title, or None if generation fails/disabled.
        """
        # Check if title already exists and is not the default placeholder
        existing = await self.get_session_title(session_id)
        if existing and existing != "New Session":
            return existing
        # Generate using core logic, with timeout to prevent zombie tasks.
        # 15 seconds is generous for a short title-generation prompt.
        try:
            metadata = await asyncio.wait_for(
                self._generate_title_core(session_id, f"user: {prompt[:500]}"),
                timeout=15.0,
            )
        except TimeoutError:
            logger.info(
                "Session title generation exceeded background timeout",
                session_id=session_id,
            )
            return None
        if metadata:
            title = metadata.title
            # Persist the title — _generate_title_core also calls update_session_title
            # internally, but we call it here too so that the title is stored even if
            # the core method is mocked in tests (where the side-effect is lost).
            # The duplicate call is harmless: update_session_title is idempotent.
            await self.update_session_title(session_id, title)
            return title
        return None

    async def generate_session_title(
        self,
        session_id: str,
        messages: Sequence[ChatMessage[Any]],
    ) -> str | None:
        """Generate and store a title for a conversation.

        Uses the configured title generation model to create a short,
        descriptive title based on the conversation content.

        Args:
            session_id: ID of the conversation to title
            messages: Messages to use for title generation

        Returns:
            The generated title, or None if title generation is disabled.
        """
        # Check if title already exists
        existing = await self.get_session_title(session_id)
        if existing:
            return existing

        # Format messages for the prompt
        formatted = "\n".join(f"{i.role}: {i.content[:500]}" for i in messages[:4])

        # Generate using core logic
        metadata = await self._generate_title_core(session_id, formatted)

        return metadata.title if metadata else None

    # Checkpoint methods

    async def save_checkpoint(
        self,
        session_id: str,
        messages_json: str,
        pending_calls: list[PendingDeferredCall],
    ) -> bool:
        """Save checkpoint data to all capable providers.

        Serializes pending_calls and distributes messages_json and
        pending_calls_json together atomically.

        Args:
            session_id: Session identifier.
            messages_json: Pre-serialized JSON of ModelMessage list.
            pending_calls: List of unresolved deferred tool calls.

        Returns:
            True if at least one provider successfully saved the checkpoint,
            False if all providers failed or no providers are configured.
        """
        calls_adapter = TypeAdapter(list[PendingDeferredCall])
        pending_calls_json = calls_adapter.dump_json(pending_calls).decode()

        any_success = False
        for provider in self.providers:
            try:
                with anyio.CancelScope(shield=True):
                    await provider.save_checkpoint(session_id, messages_json, pending_calls_json)
                any_success = True
            except NotImplementedError:
                pass
            except Exception:
                logger.exception(
                    "Error saving checkpoint",
                    provider=provider.__class__.__name__,
                    session_id=session_id,
                )
        if not any_success:
            logger.error(
                "Checkpoint save failed — no provider succeeded",
                session_id=session_id,
                provider_count=len(self.providers),
            )
        return any_success

    async def load_checkpoint(
        self,
        session_id: str,
    ) -> tuple[list[ModelMessage], list[PendingDeferredCall]] | None:
        """Load and deserialize checkpoint from first capable provider.

        Returns:
            Tuple of (messages, pending_calls) or None if no checkpoint found.
        """
        for provider in self.providers:
            try:
                result = await provider.load_checkpoint(session_id)
                if result is not None:
                    messages_json, pending_calls_json = result
                    messages = deserialize_messages(messages_json) if messages_json else []
                    calls = (
                        TypeAdapter(list[PendingDeferredCall]).validate_json(
                            pending_calls_json.encode()
                        )
                        if pending_calls_json
                        else []
                    )
                    return messages, calls
            except NotImplementedError:
                continue
            except Exception:
                logger.exception(
                    "Error loading checkpoint",
                    provider=provider.__class__.__name__,
                    session_id=session_id,
                )
        return None

    async def delete_checkpoint(self, session_id: str) -> None:
        """Delete checkpoint from all providers.

        Args:
            session_id: Session identifier.
        """
        for provider in self.providers:
            try:
                with anyio.CancelScope(shield=True):
                    await provider.delete_checkpoint(session_id)
            except NotImplementedError:
                pass
            except Exception:
                logger.exception(
                    "Error deleting checkpoint",
                    provider=provider.__class__.__name__,
                    session_id=session_id,
                )

    # Project methods

    def get_project_provider(self) -> StorageProvider:
        """Get provider capable of storing projects.

        Returns:
            First provider that supports project storage.

        Raises:
            RuntimeError: If no capable provider found.
        """
        for provider in self.providers:
            if provider.can_store_projects:
                return provider

        raise RuntimeError("No storage provider supports project storage")

    @method_spawner
    async def save_project(self, project: ProjectData) -> None:
        """Save or update a project in all capable providers.

        Args:
            project: Project data to persist
        """
        for provider in self.providers:
            try:
                with anyio.CancelScope(shield=True):
                    await provider.save_project(project)
            except NotImplementedError:
                pass
            except Exception:
                logger.exception(
                    "Error saving project",
                    provider=provider.__class__.__name__,
                    project_id=project.project_id,
                )

    @method_spawner
    async def get_project(self, project_id: str) -> ProjectData | None:
        """Get a project by ID.

        Args:
            project_id: Project identifier

        Returns:
            Project data if found, None otherwise
        """
        provider = self.get_project_provider()
        return await provider.get_project(project_id)

    @method_spawner
    async def get_project_by_worktree(self, worktree: str) -> ProjectData | None:
        """Get a project by worktree path.

        Args:
            worktree: Absolute path to the project worktree

        Returns:
            Project data if found, None otherwise
        """
        provider = self.get_project_provider()
        return await provider.get_project_by_worktree(worktree)

    @method_spawner
    async def get_project_by_name(self, name: str) -> ProjectData | None:
        """Get a project by friendly name.

        Args:
            name: Project name

        Returns:
            Project data if found, None otherwise
        """
        provider = self.get_project_provider()
        return await provider.get_project_by_name(name)

    @method_spawner
    async def list_projects(self, limit: int | None = None) -> list[ProjectData]:
        """List all projects, ordered by last_active descending.

        Args:
            limit: Maximum number of projects to return

        Returns:
            List of project data objects
        """
        provider = self.get_project_provider()
        return await provider.list_projects(limit=limit)

    @method_spawner
    async def delete_project(self, project_id: str) -> bool:
        """Delete a project from all providers.

        Args:
            project_id: Project identifier

        Returns:
            True if project was deleted from at least one provider
        """
        deleted = False
        for provider in self.providers:
            try:
                with anyio.CancelScope(shield=True):
                    if await provider.delete_project(project_id):
                        deleted = True
            except NotImplementedError:
                pass
            except Exception:
                logger.exception(
                    "Error deleting project",
                    provider=provider.__class__.__name__,
                    project_id=project_id,
                )
        return deleted

    @method_spawner
    async def touch_project(self, project_id: str) -> None:
        """Update project's last_active timestamp in all providers.

        Args:
            project_id: Project identifier
        """
        for provider in self.providers:
            try:
                await provider.touch_project(project_id)
            except NotImplementedError:
                pass
            except Exception:
                logger.exception(
                    "Error touching project",
                    provider=provider.__class__.__name__,
                    project_id=project_id,
                )
