"""SQLModel-based storage provider."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Self

from sqlmodel import SQLModel

from wolfharness.log import get_logger
from wolfharness_config.storage import SQLStorageConfig
from wolfharness_storage.base import StorageProvider
from wolfharness_storage.sql_provider.sql_messages import SQLMessagesMixin
from wolfharness_storage.sql_provider.sql_projects import SQLProjectsMixin
from wolfharness_storage.sql_provider.sql_sessions import SQLSessionsMixin


if TYPE_CHECKING:
    from types import TracebackType

    from sqlalchemy.ext.asyncio import AsyncSession


logger = get_logger(__name__)


class SQLModelProvider(
    SQLMessagesMixin,
    SQLSessionsMixin,
    SQLProjectsMixin,
    StorageProvider,
):
    """Storage provider using SQLModel.

    Can work with any database supported by SQLAlchemy/SQLModel.
    Provides efficient SQL-based storage.
    """

    can_load_history = True
    can_store_projects = True

    def __init__(self, config: SQLStorageConfig | None = None) -> None:
        """Initialize provider with async database engine.

        Args:
            config: Configuration for provider
        """
        config = config or SQLStorageConfig()
        super().__init__(config)
        self.engine = config.get_engine()
        self.auto_migrate = config.auto_migration
        self.session: AsyncSession | None = None

    async def _init_database(self, auto_migrate: bool = True) -> None:
        """Initialize database tables and run migrations.

        Args:
            auto_migrate: Whether to automatically run Alembic migrations
        """
        from wolfharness_storage.sql_provider.utils import run_alembic_migrations

        if auto_migrate:
            # Run migrations first (handles schema changes for existing DBs)
            async with self.engine.begin() as conn:
                await conn.run_sync(run_alembic_migrations)

        # Always ensure all tables exist (creates new tables, no-op for existing)
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def __aenter__(self) -> Self:
        """Initialize async database resources."""
        await self._init_database(auto_migrate=self.auto_migrate)
        await super().__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up async database resources properly.

        Handles CancelledError gracefully — during shutdown, asyncio may cancel
        the main task while engine.dispose() is still running. We shield the
        dispose call so it continues in the background even if the current task
        is cancelled, and suppress the CancelledError to avoid noisy logs.
        SQLite's WAL mode ensures data integrity even if connections are not
        explicitly closed.
        """
        dispose_task = asyncio.create_task(self.engine.dispose())
        try:
            await asyncio.shield(dispose_task)
        except asyncio.CancelledError:
            logger.warning(
                "engine.dispose() was cancelled during shutdown; "
                "SQLite WAL mode ensures data integrity"
            )
        except Exception:
            logger.exception("Error during engine.dispose()")
        return await super().__aexit__(exc_type, exc_val, exc_tb)

    def cleanup(self) -> None:
        """Clean up database resources."""
        # For sync cleanup, just pass - proper cleanup happens in __aexit__
