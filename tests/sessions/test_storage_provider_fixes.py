"""Tests for storage provider fixes.

Covers:
- SQLProvider.log_session() duplicate session handling
- OpenCodeStorageProvider.load_session() implementation
- Serialization/deserialization of ModelMessage objects
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import pytest

from wolfharness.messaging import ChatMessage
from wolfharness.sessions import SessionData
from wolfharness.utils.identifiers import ascending
from wolfharness_config.storage import OpenCodeStorageConfig, SQLStorageConfig
from wolfharness_storage.opencode_provider import OpenCodeStorageProvider
from wolfharness_storage.sql_provider import SQLModelProvider


pytestmark = pytest.mark.integration


class TestSQLProviderLogSession:
    """Tests for SQLProvider.log_session() duplicate handling."""

    @pytest.fixture
    async def provider(self):
        """Create a SQL provider with temp database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            config = SQLStorageConfig(url=f"sqlite:///{db_path}")
            provider = SQLModelProvider(config)
            async with provider:
                yield provider

    async def test_log_session_duplicate_idempotent(self, provider: SQLModelProvider) -> None:
        """Test that log_session is idempotent.

        Calling twice with same session_id should not raise.
        """
        session_id = "test_session_001"
        node_name = "test_agent"

        # First call should succeed
        await provider.log_session(
            session_id=session_id,
            node_name=node_name,
        )

        # Second call with same session_id should not raise UNIQUE constraint error
        # It should silently skip the insertion
        await provider.log_session(
            session_id=session_id,
            node_name=node_name,
        )

        # Verify session exists
        sessions = await provider.list_session_ids()
        assert session_id in sessions

    async def test_log_session_different_sessions(self, provider: SQLModelProvider) -> None:
        """Test that different sessions can be logged."""
        session_id_1 = "test_session_001"
        session_id_2 = "test_session_002"

        await provider.log_session(session_id=session_id_1, node_name="agent1")
        await provider.log_session(session_id=session_id_2, node_name="agent2")

        sessions = await provider.list_session_ids()
        assert session_id_1 in sessions
        assert session_id_2 in sessions

    async def test_log_message_duplicate_id_updates_existing_row(
        self,
        provider: SQLModelProvider,
    ) -> None:
        """Test that repeated message persistence updates instead of failing."""
        session_id = "test_session_001"
        message_id = "msg_duplicate_001"
        await provider.log_session(session_id=session_id, node_name="test_agent")

        await provider.log_message(
            message=ChatMessage(
                content="partial response",
                role="assistant",
                name="test_agent",
                session_id=session_id,
                message_id=message_id,
                model_name="openai:test-model",
            ),
        )
        await provider.log_message(
            message=ChatMessage(
                content="final response",
                role="assistant",
                name="test_agent",
                session_id=session_id,
                message_id=message_id,
                parent_id="msg_parent_001",
                model_name="openai:test-model",
                finish_reason="stop",
            ),
        )

        stored = await provider.get_message(message_id, session_id=session_id)
        assert stored is not None
        assert stored.content == "final response"
        assert stored.parent_id == "msg_parent_001"
        assert stored.finish_reason == "stop"


class TestOpenCodeStorageProviderPathHandling:
    """Tests for OpenCodeStorageProvider path handling."""

    async def test_db_path_converted_to_storage_dir(self) -> None:
        """Test that .db file path is converted to storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure with .db file path (legacy config style)
            db_path = Path(tmpdir) / "opencode.db"
            config = OpenCodeStorageConfig(path=str(db_path))
            provider = OpenCodeStorageProvider(config)

            # Provider should use storage directory, not the .db file path
            expected_base = Path(tmpdir) / "storage"
            assert provider.base_path == expected_base
            assert provider.sessions_path == expected_base / "session"
            assert provider.messages_path == expected_base / "message"
            assert provider.parts_path == expected_base / "part"

    async def test_directory_path_used_directly(self) -> None:
        """Test that directory path is used as-is."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "storage"
            config = OpenCodeStorageConfig(path=str(storage_path))
            provider = OpenCodeStorageProvider(config)

            # Provider should use the directory path directly
            assert provider.base_path == storage_path
            assert provider.sessions_path == storage_path / "session"
            assert provider.messages_path == storage_path / "message"
            assert provider.parts_path == storage_path / "part"

    """Tests for OpenCodeStorageProvider.load_session() implementation."""

    @pytest.fixture
    async def provider(self):
        """Create an OpenCode provider with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OpenCodeStorageConfig(path=tmpdir)
            provider = OpenCodeStorageProvider(config)
            async with provider:
                yield provider

    async def test_load_session_not_found(self, provider: OpenCodeStorageProvider) -> None:
        """Test loading a non-existent session returns None."""
        result = await provider.load_session("non_existent_session")
        assert result is None

    async def test_load_session_returns_session_data(
        self, provider: OpenCodeStorageProvider
    ) -> None:
        """Test that load_session returns SessionData for existing session."""
        # Use ascending() to generate unique IDs
        from wolfharness_server.opencode_server.models import (
            Session,
            SessionSummary,
            TimeCreatedUpdated,
        )

        # Create a session manually
        session_id = ascending("session")
        project_id = "test_project"

        # Create session directory and file
        session_dir = provider.sessions_path / project_id
        session_dir.mkdir(parents=True, exist_ok=True)

        now_ms = 1234567890000
        oc_session = Session(
            id=session_id,
            project_id=project_id,
            directory="/test/dir",
            title="Test Session",
            version="1.1.7",
            time=TimeCreatedUpdated(created=now_ms, updated=now_ms),
            summary=SessionSummary(files=0, additions=0, deletions=0),
        )

        import anyenv

        session_file = session_dir / f"{session_id}.json"
        session_file.write_text(anyenv.dump_json(oc_session.model_dump(by_alias=True), indent=True))

        # Create message directory (empty)
        (provider.messages_path / session_id).mkdir(parents=True, exist_ok=True)

        # Load the session
        result = await provider.load_session(session_id)

        # Verify result
        assert result is not None
        assert isinstance(result, SessionData)
        assert result.session_id == session_id
        assert result.project_id == project_id
        assert result.cwd == "/test/dir"

    async def test_delete_session(self, provider: OpenCodeStorageProvider) -> None:
        """Test that delete_session removes session and all data."""
        # Use ascending() to generate unique IDs
        from wolfharness_server.opencode_server.models import (
            Session,
            SessionSummary,
            TimeCreatedUpdated,
        )

        session_id = ascending("session")
        project_id = "test_project"

        # Create session
        session_dir = provider.sessions_path / project_id
        session_dir.mkdir(parents=True, exist_ok=True)

        oc_session = Session(
            id=session_id,
            project_id=project_id,
            directory="/test/dir",
            title="Test Session",
            version="1.1.7",
            time=TimeCreatedUpdated(created=1234567890000, updated=1234567890000),
            summary=SessionSummary(files=0, additions=0, deletions=0),
        )

        import anyenv

        session_file = session_dir / f"{session_id}.json"
        session_file.write_text(anyenv.dump_json(oc_session.model_dump(by_alias=True), indent=True))
        (provider.messages_path / session_id).mkdir(parents=True, exist_ok=True)
        (provider.parts_path / session_id).mkdir(parents=True, exist_ok=True)

        # Verify session exists
        assert (provider.messages_path / session_id).exists()

        # Delete session
        result = await provider.delete_session(session_id)
        assert result is True

        # Verify session is deleted
        result = await provider.load_session(session_id)
        assert result is None
        assert not (provider.messages_path / session_id).exists()

    async def test_list_session_ids(self, provider: OpenCodeStorageProvider) -> None:
        """Test that list_session_ids returns all session IDs."""
        # Use ascending() to generate unique IDs
        import anyenv

        from wolfharness_server.opencode_server.models import (
            Session,
            SessionSummary,
            TimeCreatedUpdated,
        )

        session_ids = []
        for i in range(3):
            session_id = ascending("session")
            session_ids.append(session_id)
            project_id = "test_project"

            session_dir = provider.sessions_path / project_id
            session_dir.mkdir(parents=True, exist_ok=True)

            oc_session = Session(
                id=session_id,
                project_id=project_id,
                directory=f"/test/dir{i}",
                title=f"Test Session {i}",
                version="1.1.7",
                time=TimeCreatedUpdated(created=1234567890000 + i, updated=1234567890000 + i),
                summary=SessionSummary(files=0, additions=0, deletions=0),
            )

            session_file = session_dir / f"{session_id}.json"
            session_file.write_text(
                anyenv.dump_json(oc_session.model_dump(by_alias=True), indent=True)
            )
            (provider.messages_path / session_id).mkdir(parents=True, exist_ok=True)

        # List sessions
        result = await provider.list_session_ids()

        # Verify all sessions are listed
        for session_id in session_ids:
            assert session_id in result


def _init_git_repo(directory: str) -> None:
    """Initialize a minimal git repo so compute_project_id returns a commit SHA."""
    subprocess.run(["git", "init"], cwd=directory, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=directory,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=directory,
        capture_output=True,
        check=True,
    )
    dummy = Path(directory) / "README.md"
    dummy.write_text("init", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=directory, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=directory,
        capture_output=True,
        check=True,
    )


class TestOpenCodeListSessionIdsCwdParameter:
    """Regression tests for OpenCodeStorageProvider.list_session_ids() missing 'cwd' parameter.

    Bug: When the TUI calls GET /session?directory=<dir>, the call chain is:
        session_routes.list_sessions()
          → NativeAgent.list_sessions(cwd=effective_cwd)
            → StorageManager.list_session_ids(cwd=cwd)
              → OpenCodeStorageProvider.list_session_ids(cwd=cwd)
                → TypeError: unexpected keyword argument 'cwd'

    The base class StorageProvider.list_session_ids() and all other providers
    (SQLModelProvider, MemoryProvider) accept 'cwd', but OpenCodeStorageProvider
    overrides the method WITHOUT the 'cwd' parameter, causing a TypeError that
    is silently caught by NativeAgent.list_sessions() (line 1220), returning [].
    """

    @pytest.fixture
    async def provider(self):
        """Create an OpenCode provider with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OpenCodeStorageConfig(path=tmpdir)
            prov = OpenCodeStorageProvider(config)
            async with prov:
                yield prov

    async def test_list_session_ids_accepts_cwd_parameter(
        self, provider: OpenCodeStorageProvider
    ) -> None:
        """list_session_ids must accept 'cwd' keyword argument (base class contract).

        The base class StorageProvider.list_session_ids() defines cwd: str | None = None.
        OpenCodeStorageProvider overrides this method but omits the cwd parameter,
        causing TypeError when called with cwd= via StorageManager.
        """
        # This should NOT raise TypeError — it must accept cwd like the base class
        result = await provider.list_session_ids(cwd="/some/path")
        assert isinstance(result, list)

    async def test_list_session_ids_cwd_filters_by_directory(
        self, provider: OpenCodeStorageProvider
    ) -> None:
        """list_session_ids(cwd=...) should filter sessions by their directory field.

        Uses real git repos so compute_project_id() returns a valid project_id
        that matches the session storage layout, mirroring production behavior.
        """
        import anyenv

        from wolfharness_server.opencode_server.models import (
            Session,
            TimeCreatedUpdated,
        )
        from wolfharness_storage.opencode_provider.helpers import compute_project_id

        # Create two separate git repos (simulating two different projects)
        alpha_dir = provider.base_path / "alpha_project"
        alpha_dir.mkdir(parents=True, exist_ok=True)
        _init_git_repo(str(alpha_dir))

        beta_dir = provider.base_path / "beta_project"
        beta_dir.mkdir(parents=True, exist_ok=True)
        _init_git_repo(str(beta_dir))

        alpha_project_id = compute_project_id(str(alpha_dir))
        beta_project_id = compute_project_id(str(beta_dir))

        # Create session in alpha project
        alpha_id = ascending("session")
        alpha_project_dir = provider.sessions_path / alpha_project_id
        alpha_project_dir.mkdir(parents=True, exist_ok=True)
        alpha_session = Session(
            id=alpha_id,
            project_id=alpha_project_id,
            directory=str(alpha_dir),
            title="Alpha Session",
            version="1.1.7",
            time=TimeCreatedUpdated(created=1000, updated=1000),
        )
        (alpha_project_dir / f"{alpha_id}.json").write_text(
            anyenv.dump_json(alpha_session.model_dump(by_alias=True), indent=True)
        )
        (provider.messages_path / alpha_id).mkdir(parents=True, exist_ok=True)

        # Create session in beta project
        beta_id = ascending("session")
        beta_project_dir = provider.sessions_path / beta_project_id
        beta_project_dir.mkdir(parents=True, exist_ok=True)
        beta_session = Session(
            id=beta_id,
            project_id=beta_project_id,
            directory=str(beta_dir),
            title="Beta Session",
            version="1.1.7",
            time=TimeCreatedUpdated(created=2000, updated=2000),
        )
        (beta_project_dir / f"{beta_id}.json").write_text(
            anyenv.dump_json(beta_session.model_dump(by_alias=True), indent=True)
        )
        (provider.messages_path / beta_id).mkdir(parents=True, exist_ok=True)

        # Filter by cwd — should only return matching project's session
        alpha_only = await provider.list_session_ids(cwd=str(alpha_dir))
        assert alpha_id in alpha_only
        assert beta_id not in alpha_only

        beta_only = await provider.list_session_ids(cwd=str(beta_dir))
        assert beta_id in beta_only
        assert alpha_id not in beta_only

    async def test_list_session_ids_cwd_none_returns_all(
        self, provider: OpenCodeStorageProvider
    ) -> None:
        """list_session_ids(cwd=None) should return all sessions (no filtering)."""
        import anyenv

        from wolfharness_server.opencode_server.models import (
            Session,
            TimeCreatedUpdated,
        )

        project_id = "test_project"
        project_dir = provider.sessions_path / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        session_id = ascending("session")
        session = Session(
            id=session_id,
            project_id=project_id,
            directory="/some/dir",
            title="Test Session",
            version="1.1.7",
            time=TimeCreatedUpdated(created=1000, updated=1000),
        )
        (project_dir / f"{session_id}.json").write_text(
            anyenv.dump_json(session.model_dump(by_alias=True), indent=True)
        )
        (provider.messages_path / session_id).mkdir(parents=True, exist_ok=True)

        # No cwd filter — should return all sessions
        all_sessions = await provider.list_session_ids(cwd=None)
        assert session_id in all_sessions

    async def test_list_session_ids_cwd_excludes_corrupted_session(
        self, provider: OpenCodeStorageProvider
    ) -> None:
        """Corrupted session files (read_session returns None) must be excluded.

        This applies when cwd filter is active.

        Regression test: previously, if read_session returned None (corrupted JSON / I/O error),
        the cwd filter was bypassed and the session was incorrectly included in results.
        """
        from wolfharness_storage.opencode_provider.helpers import compute_project_id

        # Create a real git repo so compute_project_id returns a valid project_id
        workdir = provider.base_path / "corrupt_project"
        workdir.mkdir(parents=True, exist_ok=True)
        _init_git_repo(str(workdir))

        project_id = compute_project_id(str(workdir))
        project_dir = provider.sessions_path / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create a valid session
        valid_id = ascending("session")
        import anyenv

        from wolfharness_server.opencode_server.models import (
            Session,
            TimeCreatedUpdated,
        )

        valid_session = Session(
            id=valid_id,
            project_id=project_id,
            directory=str(workdir),
            title="Valid Session",
            version="1.1.7",
            time=TimeCreatedUpdated(created=1000, updated=1000),
        )
        (project_dir / f"{valid_id}.json").write_text(
            anyenv.dump_json(valid_session.model_dump(by_alias=True), indent=True)
        )
        (provider.messages_path / valid_id).mkdir(parents=True, exist_ok=True)

        # Create a corrupted session (invalid JSON)
        corrupt_id = ascending("session")
        (project_dir / f"{corrupt_id}.json").write_text("{invalid json!!!", encoding="utf-8")
        (provider.messages_path / corrupt_id).mkdir(parents=True, exist_ok=True)

        # Filter by cwd — corrupted session must be excluded
        result = await provider.list_session_ids(cwd=str(workdir))
        assert valid_id in result
        assert corrupt_id not in result

    async def test_list_session_ids_signature_matches_base_class(
        self, provider: OpenCodeStorageProvider
    ) -> None:
        """OpenCodeStorageProvider.list_session_ids must have the same signature as base class.

        This is a static check: the override's signature must include all keyword
        parameters defined in StorageProvider.list_session_ids().
        """
        import inspect

        from wolfharness_storage.base import StorageProvider

        base_params = inspect.signature(StorageProvider.list_session_ids).parameters
        override_params = inspect.signature(OpenCodeStorageProvider.list_session_ids).parameters

        for param_name in base_params:
            if param_name == "self":
                continue
            assert param_name in override_params, (
                f"OpenCodeStorageProvider.list_session_ids() missing parameter "
                f"'{param_name}' that exists in base class StorageProvider.list_session_ids(). "
                f"Base has: {list(base_params.keys())}, "
                f"Override has: {list(override_params.keys())}"
            )


class TestSerialization:
    """Tests for ModelMessage serialization/deserialization."""

    def test_deserialize_messages_returns_model_message_objects(self) -> None:
        """Test that deserialize_messages returns ModelMessage objects, not dicts.

        This is a regression test for a bug where TypeAdapter(list) was used
        instead of TypeAdapter(list[ModelMessage]), causing deserialization
        to return dicts instead of ModelMessage objects. This caused pydantic-ai
        to ignore the message history when loading sessions.
        """
        from pydantic_ai import ModelRequest, ModelResponse, TextPart, UserPromptPart

        from wolfharness.storage.serialization import (
            deserialize_messages,
            serialize_messages,
        )

        # Create sample messages
        request = ModelRequest(
            parts=[UserPromptPart(content="Hello")],
            instructions=None,
        )
        response = ModelResponse(
            parts=[TextPart(content="Hi there")],
            model_name="gpt-4",
        )
        original_messages = [request, response]

        # Serialize
        serialized = serialize_messages(original_messages)
        assert serialized is not None

        # Deserialize
        deserialized = deserialize_messages(serialized)

        # Verify we got ModelMessage objects, not dicts
        assert len(deserialized) == 2
        assert isinstance(deserialized[0], ModelRequest)
        assert isinstance(deserialized[1], ModelResponse)

        # Verify content is preserved
        assert isinstance(deserialized[0].parts[0], UserPromptPart)
        assert deserialized[0].parts[0].content == "Hello"
        assert isinstance(deserialized[1].parts[0], TextPart)
        assert deserialized[1].parts[0].content == "Hi there"

    def test_deserialize_messages_empty_json(self) -> None:
        """Test that deserialize_messages handles empty/None input."""
        from wolfharness.storage.serialization import deserialize_messages

        assert deserialize_messages(None) == []
        assert deserialize_messages("") == []

    def test_serialize_messages_empty_list(self) -> None:
        """Test that serialize_messages returns None for empty list."""
        from wolfharness.storage.serialization import serialize_messages

        assert serialize_messages([]) is None
