"""Project store mixin for SQLModelProvider.

Extracted from sql_provider.py as part of the session-debt-cleanup file split.
Contains project-related database operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import desc, select

from wolfharness.log import get_logger
from wolfharness.utils.time_utils import get_now
from wolfharness_storage.sql_provider.models import Project


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from wolfharness.sessions.models import ProjectData


logger = get_logger(__name__)


class SQLProjectsMixin:
    """Mixin providing project store methods for SQLModelProvider.

    Attributes:
        engine: Async database engine (provided by SQLModelProvider).
    """

    engine: AsyncEngine

    def _to_project_data(self, row: Project) -> ProjectData:
        """Convert database model to ProjectData."""
        from wolfharness.sessions.models import ProjectData

        return ProjectData(
            project_id=row.project_id,
            worktree=row.worktree,
            name=row.name,
            vcs=row.vcs,
            config_path=row.config_path,
            created_at=row.created_at,
            last_active=row.last_active,
            settings=row.settings_json or {},
        )

    def _to_project_model(self, data: ProjectData) -> Project:
        """Convert ProjectData to database model."""
        return Project(
            project_id=data.project_id,
            worktree=data.worktree,
            name=data.name,
            vcs=data.vcs,
            config_path=data.config_path,
            created_at=data.created_at,
            last_active=data.last_active,
            settings_json=data.settings,
        )

    async def save_project(self, project: ProjectData) -> None:
        """Save or update a project."""
        from sqlalchemy import delete

        async with AsyncSession(self.engine) as session:
            # Delete existing if present (upsert via delete+insert)
            stmt = delete(Project).where(Project.project_id == project.project_id)  # type: ignore[arg-type]
            await session.execute(stmt)
            # Insert new/updated
            db_project = self._to_project_model(project)
            session.add(db_project)
            await session.commit()
            logger.debug("Saved project", project_id=project.project_id)

    async def get_project(self, project_id: str) -> ProjectData | None:
        """Get a project by ID."""
        async with AsyncSession(self.engine) as session:
            stmt = select(Project).where(Project.project_id == project_id)
            result = await session.execute(stmt)
            row = result.scalars().first()
            return self._to_project_data(row) if row else None

    async def get_project_by_worktree(self, worktree: str) -> ProjectData | None:
        """Get a project by worktree path."""
        async with AsyncSession(self.engine) as session:
            stmt = select(Project).where(Project.worktree == worktree)
            result = await session.execute(stmt)
            row = result.scalars().first()
            return self._to_project_data(row) if row else None

    async def get_project_by_name(self, name: str) -> ProjectData | None:
        """Get a project by friendly name."""
        async with AsyncSession(self.engine) as session:
            stmt = select(Project).where(Project.name == name)
            result = await session.execute(stmt)
            row = result.scalars().first()
            return self._to_project_data(row) if row else None

    async def list_projects(self, limit: int | None = None) -> list[ProjectData]:
        """List all projects, ordered by last_active descending."""
        async with AsyncSession(self.engine) as session:
            stmt = select(Project).order_by(desc(Project.last_active))
            if limit is not None:
                stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return [self._to_project_data(row) for row in result.scalars().all()]

    async def delete_project(self, project_id: str) -> bool:
        """Delete a project."""
        from sqlalchemy import delete

        async with AsyncSession(self.engine) as session:
            stmt = delete(Project).where(Project.project_id == project_id)  # type: ignore[arg-type]
            result = await session.execute(stmt)
            await session.commit()
            deleted: bool = result.rowcount > 0  # type: ignore[attr-defined]
            if deleted:
                logger.debug("Deleted project", project_id=project_id)
            return deleted

    async def touch_project(self, project_id: str) -> None:
        """Update project's last_active timestamp."""
        from sqlalchemy import update

        async with AsyncSession(self.engine) as session:
            stmt = (
                update(Project)
                .where(Project.project_id == project_id)  # type: ignore[arg-type]
                .values(last_active=get_now())
            )
            await session.execute(stmt)
            await session.commit()
