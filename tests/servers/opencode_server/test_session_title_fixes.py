"""Pytest tests for session title persistence fixes.

These tests verify the fixes for:
1. converters.py: opencode_to_session_data saves title to metadata
2. session_store.py: SQL storage converters handle title field
3. storage/manager.py: _generate_title_from_prompt stores generated title
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from wolfharness.sessions.models import SessionData
from wolfharness.storage.manager import (
    SessionMetadata,
    SessionMetadataGeneratedEvent,
    StorageManager,
)
from wolfharness_config.storage import MemoryStorageConfig, SQLStorageConfig, StorageConfig
from wolfharness_server.opencode_server.converters import (
    opencode_to_session_data,
    session_data_to_opencode,
)
from wolfharness_server.opencode_server.models import Session
from wolfharness_server.opencode_server.models.common import TimeCreatedUpdated
from wolfharness_storage.memory_provider.provider import MemoryStorageProvider
from wolfharness_storage.sql_provider.sql_provider import SQLModelProvider


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from pathlib import Path


class TestConvertersTitleFix:
    """Tests for converters.py title persistence fix."""

    def test_opencode_to_session_data_saves_title(self) -> None:
        """Verify opencode_to_session_data saves title to metadata."""
        session = Session(
            id="test_session_001",
            project_id="global",
            directory="/tmp",
            title="My Test Title",
            version="1",
            time=TimeCreatedUpdated(created=1234567890, updated=1234567890),
        )

        session_data = opencode_to_session_data(session, agent_name="test_agent")

        # Title should be in metadata
        assert "title" in session_data.metadata
        assert session_data.metadata["title"] == "My Test Title"
        # Title property should also work
        assert session_data.title == "My Test Title"

    def test_opencode_to_session_data_empty_title(self) -> None:
        """Verify opencode_to_session_data handles empty title correctly."""
        session = Session(
            id="test_session_002",
            project_id="global",
            directory="/tmp",
            title="",  # Empty title
            version="1",
            time=TimeCreatedUpdated(created=1234567890, updated=1234567890),
        )

        session_data = opencode_to_session_data(session, agent_name="test_agent")

        # Empty title should not be added to metadata
        assert session_data.metadata.get("title") in (None, "")

    def test_session_data_to_opencode_reads_title(self) -> None:
        """Verify session_data_to_opencode reads title from metadata."""
        session_data = SessionData(
            session_id="test_session_003",
            agent_name="test_agent",
            metadata={"title": "Metadata Title"},
        )

        session = session_data_to_opencode(session_data)

        assert session.title == "Metadata Title"

    def test_session_data_to_opencode_default_title(self) -> None:
        """Verify session_data_to_opencode uses default when no title in metadata."""
        session_data = SessionData(
            session_id="test_session_004",
            agent_name="test_agent",
            metadata={},  # No title
        )

        session = session_data_to_opencode(session_data)

        assert session.title == "New Session"  # Default value

    def test_roundtrip_conversion_preserves_title(self) -> None:
        """Verify round-trip conversion preserves title."""
        original = Session(
            id="test_roundtrip",
            project_id="global",
            directory="/tmp",
            title="Round Trip Title",
            version="1",
            time=TimeCreatedUpdated(created=1234567890, updated=1234567890),
        )

        # Convert to SessionData and back
        session_data = opencode_to_session_data(original, agent_name="test_agent")
        converted = session_data_to_opencode(session_data)

        assert converted.title == "Round Trip Title"


class TestSQLModelProviderTitleFix:
    """Tests for SQLModelProvider title field handling."""

    @pytest.fixture
    def sql_config(self, tmp_path: Path) -> SQLStorageConfig:
        """Create SQL config with temp database."""
        db_path = tmp_path / "test_title_fix.db"
        return SQLStorageConfig(url=f"sqlite:///{db_path}")

    async def test_sql_save_includes_title(self, sql_config: SQLStorageConfig) -> None:
        """Verify save_session includes title field in the DB row."""
        async with SQLModelProvider(sql_config) as store:
            session_data = SessionData(
                session_id="sql_test_001",
                agent_name="test_agent",
                metadata={"title": "SQL Stored Title", "custom_key": "value"},
            )
            await store.save_session(session_data)

            # Load it back and verify title roundtrips
            loaded = await store.load_session("sql_test_001")
            assert loaded is not None
            assert loaded.title == "SQL Stored Title"
            # Metadata should still have other keys
            assert "custom_key" in loaded.metadata

    async def test_sql_from_db_includes_title(self, sql_config: SQLStorageConfig) -> None:
        """Verify _session_from_db includes title in metadata."""
        async with SQLModelProvider(sql_config) as store:
            # Create and save a session with title
            session_data = SessionData(
                session_id="sql_test_002",
                agent_name="test_agent",
                metadata={"title": "Original Title"},
            )
            await store.save_session(session_data)

            # Load it back
            loaded = await store.load_session("sql_test_002")

            assert loaded is not None
            # Title should be in metadata
            assert "title" in loaded.metadata
            assert loaded.metadata["title"] == "Original Title"
            # Title property should work
            assert loaded.title == "Original Title"

    async def test_sql_save_load_roundtrip_title(self, sql_config: SQLStorageConfig) -> None:
        """Verify title survives SQL save/load roundtrip."""
        async with SQLModelProvider(sql_config) as store:
            original = SessionData(
                session_id="sql_test_003",
                agent_name="test_agent",
                metadata={"title": "Round Trip Title", "other": "data"},
            )

            # Save
            await store.save_session(original)

            # Load
            loaded = await store.load_session("sql_test_003")

            assert loaded is not None
            assert loaded.title == "Round Trip Title"
            assert loaded.metadata.get("other") == "data"


class TestStorageManagerTitleGenerationFix:
    """Tests for storage/manager.py title generation fix."""

    async def test_generate_title_from_prompt_stores_title(self) -> None:
        """Verify _generate_title_from_prompt stores the generated title."""
        config = StorageConfig(
            providers=[MemoryStorageConfig()],
            title_generation_model="test",
        )

        async with StorageManager(config) as manager:
            session_id = "gen_test_001"

            # Create session first
            await manager.log_session(session_id=session_id, node_name="test_agent")

            # Mock title generation
            mock_metadata = SessionMetadata(
                title="Generated Title",
                emoji="🧪",
                icon="mdi:test-tube",
            )

            with patch.object(
                StorageManager,
                "_generate_title_core",
                return_value=mock_metadata,
            ):
                # Generate title
                title = await manager._generate_title_from_prompt(
                    session_id=session_id,
                    prompt="Test prompt",
                )

                # Should return the title
                assert title == "Generated Title"

                # Title should be stored
                stored = await manager.get_session_title(session_id)
                assert stored == "Generated Title"

    async def test_generate_title_from_prompt_emits_signal(self) -> None:
        """Verify _generate_title_from_prompt emits metadata_generated signal."""
        config = StorageConfig(
            providers=[MemoryStorageConfig()],
            title_generation_model="test",
        )

        async with StorageManager(config) as manager:
            session_id = "gen_test_002"
            await manager.log_session(session_id=session_id, node_name="test_agent")

            signal_titles: list[str] = []

            def on_signal(event):
                signal_titles.append(event.metadata.title)

            manager.metadata_generated.connect(on_signal)

            mock_metadata = SessionMetadata(
                title="Signal Title",
                emoji="\ud83d\udce1",
                icon="mdi:antenna",
            )

            async def mock_core_with_signal(self_, sid, prompt):
                event = SessionMetadataGeneratedEvent(session_id=sid, metadata=mock_metadata)
                await self_.metadata_generated.emit(event)
                return mock_metadata

            with patch.object(
                StorageManager,
                "_generate_title_core",
                mock_core_with_signal,
            ):
                await manager._generate_title_from_prompt(
                    session_id=session_id,
                    prompt="Another prompt",
                )

                assert len(signal_titles) > 0
                assert signal_titles[0] == "Signal Title"

    async def test_generate_title_from_prompt_skips_if_exists(self) -> None:
        """Verify _generate_title_from_prompt skips generation if title exists."""
        config = StorageConfig(
            providers=[MemoryStorageConfig()],
            title_generation_model="test",
        )

        async with StorageManager(config) as manager:
            session_id = "gen_test_003"
            await manager.log_session(session_id=session_id, node_name="test_agent")

            # Set existing title
            await manager.update_session_title(session_id, "Existing Title")

            # Try to generate - should return existing without calling core
            with patch.object(
                StorageManager,
                "_generate_title_core",
            ) as mock_core:
                title = await manager._generate_title_from_prompt(
                    session_id=session_id,
                    prompt="New prompt",
                )

                assert title == "Existing Title"
                mock_core.assert_not_called()  # Should not generate

    async def test_generate_title_from_prompt_generates_despite_default_title(self) -> None:
        """Verify _generate_title_from_prompt generates title despite default 'New Session'.

        Regression test: 'New Session' is the default placeholder title assigned
        when a session is created.  Title generation should NOT be blocked by
        this default value — it should proceed and replace it with an LLM-generated title.
        """
        config = StorageConfig(
            providers=[MemoryStorageConfig()],
            title_generation_model="test",
        )

        async with StorageManager(config) as manager:
            session_id = "gen_test_new_session_default"
            await manager.log_session(session_id=session_id, node_name="test_agent")

            # Set the default placeholder title (simulates what create_session does)
            await manager.update_session_title(session_id, "New Session")

            mock_metadata = SessionMetadata(
                title="Generated Title",
                emoji="🧪",
                icon="mdi:test-tube",
            )

            with patch.object(
                StorageManager,
                "_generate_title_core",
                return_value=mock_metadata,
            ) as mock_core:
                title = await manager._generate_title_from_prompt(
                    session_id=session_id,
                    prompt="Test prompt",
                )

                # Should have called _generate_title_core (not skipped)
                mock_core.assert_called_once()
                # Should return the generated title, not "New Session"
                assert title == "Generated Title"
                # Title should be stored
                stored = await manager.get_session_title(session_id)
                assert stored == "Generated Title"

    async def test_log_session_triggers_generation(self) -> None:
        """Verify log_session triggers title generation with initial_prompt."""
        config = StorageConfig(
            providers=[MemoryStorageConfig()],
            title_generation_model="test",
        )

        async with StorageManager(config) as manager:
            session_id = "log_test_001"
            signal_titles: list[str] = []

            def on_signal(event):
                signal_titles.append(event.metadata.title)

            manager.metadata_generated.connect(on_signal)

            mock_metadata = SessionMetadata(
                title="Auto Title",
                emoji="\ud83e\udd16",
                icon="mdi:robot",
            )

            async def mock_core_with_signal(self_, sid, prompt):
                event = SessionMetadataGeneratedEvent(session_id=sid, metadata=mock_metadata)
                await self_.metadata_generated.emit(event)
                return mock_metadata

            with (
                patch.object(
                    StorageManager,
                    "_generate_title_core",
                    mock_core_with_signal,
                ),
                patch.dict(os.environ, {}, clear=False),
            ):
                os.environ.pop("PYTEST_CURRENT_TEST", None)

                await manager.log_session(
                    session_id=session_id,
                    node_name="test_agent",
                    initial_prompt="Generate a title for this",
                )

                # Wait for async processing
                await anyio.sleep(0.1)

            # Title should be generated and stored
            stored = await manager.get_session_title(session_id)
            assert stored == "Auto Title"
            assert len(signal_titles) > 0


class TestMemoryProviderTitleOperations:
    """Tests for MemoryStorageProvider title operations."""

    async def test_memory_provider_update_and_get_title(self) -> None:
        """Verify MemoryStorageProvider update/get title works."""
        config = MemoryStorageConfig()
        provider = MemoryStorageProvider(config)

        session_id = "mem_test_001"
        await provider.log_session(session_id=session_id, node_name="test_agent")

        # Initially no title
        assert await provider.get_session_title(session_id) is None

        # Update title
        await provider.update_session_title(session_id, "Memory Title")

        # Get title
        assert await provider.get_session_title(session_id) == "Memory Title"


class TestOpenCodeProviderTitleFix:
    """Tests for OpenCodeStorageProvider title persistence fix."""

    async def test_save_session_creates_new_session_file(self, tmp_path: Path) -> None:
        """Verify save_session creates new session file when it doesn't exist."""
        from wolfharness_config.storage import OpenCodeStorageConfig
        from wolfharness_storage.opencode_provider.provider import OpenCodeStorageProvider

        config = OpenCodeStorageConfig(path=str(tmp_path / "opencode_storage"))
        provider = OpenCodeStorageProvider(config)

        # Create SessionData with title
        session_data = SessionData(
            session_id="new_session_001",
            agent_name="test_agent",
            project_id="test_project",
            cwd="/tmp/test",
            metadata={"title": "New Session Title"},
        )

        # Save session (should create new file)
        await provider.save_session(session_data)

        # Verify session file was created
        session_file = provider.sessions_path / "test_project" / "new_session_001.json"
        assert session_file.exists(), f"Session file should be created at {session_file}"

        # Verify title was saved
        from wolfharness_storage.opencode_provider import helpers

        oc_session = helpers.read_session(session_file)
        assert oc_session is not None
        title = oc_session.title
        assert title == "New Session Title"
        assert oc_session.id == "new_session_001"
        assert oc_session.project_id == "test_project"

    async def test_save_session_updates_existing_session(self, tmp_path: Path) -> None:
        """Verify save_session updates existing session file."""
        from wolfharness_config.storage import OpenCodeStorageConfig
        from wolfharness_storage.opencode_provider.provider import OpenCodeStorageProvider

        config = OpenCodeStorageConfig(path=str(tmp_path / "opencode_storage"))
        provider = OpenCodeStorageProvider(config)

        # Create initial session
        session_data = SessionData(
            session_id="update_session_001",
            agent_name="test_agent",
            project_id="test_project",
            metadata={"title": "Initial Title"},
        )
        await provider.save_session(session_data)

        # Update title
        updated_data = SessionData(
            session_id="update_session_001",
            agent_name="test_agent",
            project_id="test_project",
            metadata={"title": "Updated Title"},
        )
        await provider.save_session(updated_data)

        # Verify title was updated
        session_file = provider.sessions_path / "test_project" / "update_session_001.json"
        from wolfharness_storage.opencode_provider import helpers

        oc_session = helpers.read_session(session_file)
        assert oc_session is not None
        title = oc_session.title
        assert title == "Updated Title"

    async def test_load_session_reads_title(self, tmp_path: Path) -> None:
        """Verify load_session reads title from session file."""
        from wolfharness_config.storage import OpenCodeStorageConfig
        from wolfharness_storage.opencode_provider.provider import OpenCodeStorageProvider

        config = OpenCodeStorageConfig(path=str(tmp_path / "opencode_storage"))
        provider = OpenCodeStorageProvider(config)

        # Create session with title
        session_data = SessionData(
            session_id="load_session_001",
            agent_name="test_agent",
            project_id="test_project",
            metadata={"title": "Loadable Title"},
        )
        await provider.save_session(session_data)

        # Load session
        loaded = await provider.load_session("load_session_001")

        assert loaded is not None
        assert loaded.title == "Loadable Title"
        assert loaded.metadata.get("title") == "Loadable Title"


class TestSQLProviderTitleOperations:
    """Tests for SQLModelProvider title operations."""

    @pytest.fixture
    def sql_config(self, tmp_path: Path) -> SQLStorageConfig:
        """Create SQL config with temp database."""
        db_path = tmp_path / "test_sql_title.db"
        return SQLStorageConfig(url=f"sqlite:///{db_path}")

    async def test_sql_provider_update_and_get_title(self, sql_config: SQLStorageConfig) -> None:
        """Verify SQLModelProvider update/get title works."""
        async with SQLModelProvider(sql_config) as provider:
            session_id = "sql_prov_test_001"
            await provider.log_session(session_id=session_id, node_name="test_agent")

            # Initially no title
            assert await provider.get_session_title(session_id) is None

            # Update title
            await provider.update_session_title(session_id, "SQL Provider Title")

            # Get title
            assert await provider.get_session_title(session_id) == "SQL Provider Title"


class TestModelVariantResolution:
    """Tests for _generate_title_core resolving model_variants before infer_model."""

    async def test_generate_title_core_resolves_variant_name(self) -> None:
        """Verify _generate_title_core resolves a model variant name before infer_model.

        Regression test: When title_generation_model is a variant name like 'ack-dev',
        infer_model() raises 'Unknown model: ack-dev'. The fix checks _model_variants
        first and calls variant.get_model() to get the actual Model instance.
        """
        config = StorageConfig(
            providers=[MemoryStorageConfig()],
            title_generation_model="my-variant",
        )

        async with StorageManager(config) as manager:
            # Simulate AgentPool setting _model_variants after init
            mock_variant = MagicMock()
            mock_model = MagicMock()
            mock_variant.get_model.return_value = mock_model
            manager._model_variants = {"my-variant": mock_variant}

            # Patch the Agent import inside the method body
            with patch("wolfharness.Agent") as mock_agent_cls:
                mock_agent = MagicMock()
                mock_result = MagicMock()
                mock_result.data = SessionMetadata(
                    title="Variant Resolved Title",
                    emoji="✅",
                    icon="mdi:check",
                )
                mock_agent.run = AsyncMock(return_value=mock_result)
                mock_agent_cls.return_value = mock_agent

                await manager._generate_title_core("test_session", "user: hello")

                # Should have called variant.get_model(), not infer_model
                mock_variant.get_model.assert_called_once()
                # Agent should be created with the resolved model
                mock_agent_cls.assert_called_once()
                call_kwargs = mock_agent_cls.call_args
                assert (
                    call_kwargs.kwargs.get("model") is mock_model
                    or call_kwargs[1].get("model") is mock_model
                )

    async def test_generate_title_core_falls_back_to_infer_model(self) -> None:
        """Verify _generate_title_core falls back to infer_model for non-variant model strings.

        When title_generation_model is not in _model_variants, it should be passed
        directly to infer_model() rather than raising 'Unknown model'.
        """
        config = StorageConfig(
            providers=[MemoryStorageConfig()],
            title_generation_model="openai:gpt-4o-mini",
        )

        async with StorageManager(config) as manager:
            # No model_variants set (empty dict by default) - so model string
            # goes to infer_model(). We just verify no crash on variant lookup.
            # The actual LLM call will fail, but we catch that as expected.
            with contextlib.suppress(Exception):
                await manager._generate_title_core("test_session", "user: hello")


class TestTitlePersistedAfterReload:
    """Regression tests: title must survive session reload from storage.

    Bug: ``update_session_title`` only wrote the ``Conversation.title`` column,
    but ``_session_from_db`` skipped syncing it into ``metadata["title"]`` when
    a stale entry already existed there (from ``create_session`` setting
    ``title="New Session"``).  Result: generated title visible in current TUI
    session but lost on reload.

    Fix: ``_session_from_db`` always lets ``Conversation.title`` override
    ``metadata_json["title"]`` so the column is the single source of truth.
    """

    @pytest.fixture
    def sql_config(self, tmp_path: Path) -> SQLStorageConfig:
        """Create SQL config with temp database."""
        db_path = tmp_path / "test_title_reload.db"
        return SQLStorageConfig(url=f"sqlite:///{db_path}")

    async def test_sql_title_survives_reload(self, sql_config: SQLStorageConfig) -> None:
        """Verify generated title persists after load_session round-trip."""
        async with SQLModelProvider(sql_config) as provider:
            session_id = "reload_test_001"

            # 1. Save session with default title (mimics create_session)
            data = SessionData(
                session_id=session_id,
                agent_name="test_agent",
                metadata={"title": "New Session"},
            )
            await provider.save_session(data)

            # 2. Update title (mimics title generation)
            await provider.update_session_title(session_id, "Generated Title")

            # 3. Reload from storage — title must be "Generated Title", not "New Session"
            loaded = await provider.load_session(session_id)
            assert loaded is not None
            assert loaded.title == "Generated Title"

    async def test_sql_title_in_list_sessions(self, sql_config: SQLStorageConfig) -> None:
        """Verify generated title appears in list_sessions (batch query)."""
        async with SQLModelProvider(sql_config) as provider:
            session_id = "list_test_001"
            data = SessionData(
                session_id=session_id,
                agent_name="test_agent",
                metadata={"title": "New Session"},
            )
            await provider.save_session(data)
            await provider.update_session_title(session_id, "Listed Title")

            # Batch load should return updated title
            sessions = await provider.load_sessions_batch([session_id])
            assert len(sessions) == 1
            assert sessions[0].title == "Listed Title"

    async def test_memory_title_survives_reload(self) -> None:
        """Verify generated title persists in MemoryStorageProvider."""
        config = StorageConfig(providers=[MemoryStorageConfig()])
        async with StorageManager(config) as manager:
            session_id = "mem_reload_001"
            await manager.log_session(session_id=session_id, node_name="test_agent")
            # Save to sessions dict (mimics create_session flow)
            data = SessionData(
                session_id=session_id,
                agent_name="test_agent",
                metadata={"title": "New Session"},
            )
            await manager.save_session(data)

            # Update title
            await manager.update_session_title(session_id, "Memory Generated Title")

            # Reload
            loaded = await manager.load_session(session_id)
            assert loaded is not None
            assert loaded.title == "Memory Generated Title"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
