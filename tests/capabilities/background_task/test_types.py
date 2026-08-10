"""Tests for background task types."""

from __future__ import annotations

import asyncio
from datetime import UTC

import pytest

from wolfharness.capabilities.background_task import BackgroundTask, TaskHandle, TaskStatus
from wolfharness.capabilities.background_task.types import TaskStatus as TaskStatusFromTypes


def test_background_task_defaults() -> None:
    """BackgroundTask instantiates with required fields and status == pending."""
    task = BackgroundTask(
        id="t1",
        description="test task",
        agent_or_team="agent_a",
        prompt="do something",
        parent_session_id="s1",
        child_session_id=None,
    )
    assert task.status == "pending"
    assert task.load_skills == []
    assert task.result is None
    assert task.error is None
    assert task.output_file is None
    assert task.started_at is None
    assert task.completed_at is None


def test_background_task_created_at_is_utc_aware() -> None:
    """BackgroundTask.created_at is timezone-aware UTC."""
    task = BackgroundTask(
        id="t2",
        description="utc check",
        agent_or_team="agent_b",
        prompt="check tz",
        parent_session_id=None,
        child_session_id=None,
    )
    assert task.created_at.tzinfo is not None
    assert task.created_at.tzinfo == UTC


def test_background_task_load_skills_default_empty() -> None:
    """BackgroundTask.load_skills defaults to empty list."""
    task = BackgroundTask(
        id="t3",
        description="skills check",
        agent_or_team="agent_c",
        prompt="check skills",
        parent_session_id=None,
        child_session_id=None,
    )
    assert task.load_skills == []
    assert isinstance(task.load_skills, list)


def test_background_task_no_asyncio_attrs() -> None:
    """BackgroundTask does NOT have asyncio.Task or asyncio.Event attributes."""
    task = BackgroundTask(
        id="t4",
        description="serializable check",
        agent_or_team="agent_d",
        prompt="check serializable",
        parent_session_id=None,
        child_session_id=None,
    )
    slot_names = set(task.__slots__)
    assert "task" not in slot_names
    assert "completion_event" not in slot_names
    assert not hasattr(task, "task")
    assert not hasattr(task, "completion_event")


async def test_task_handle_wraps_asyncio_task() -> None:
    """TaskHandle wraps an asyncio.Task[None]."""

    async def dummy() -> None:
        pass

    coro = dummy()
    atask = asyncio.create_task(coro)
    handle = TaskHandle(task=atask)
    assert isinstance(handle.task, asyncio.Task)
    await atask


async def test_task_handle_completion_event_is_asyncio_event() -> None:
    """TaskHandle.completion_event is an asyncio.Event."""

    async def dummy() -> None:
        pass

    atask = asyncio.create_task(dummy())
    handle = TaskHandle(task=atask)
    assert isinstance(handle.completion_event, asyncio.Event)
    assert not handle.completion_event.is_set()
    await atask


@pytest.mark.parametrize(
    "status",
    ["pending", "running", "cancelling", "completed", "error", "cancelled", "timed_out"],
)
def test_task_status_accepts_all_literal_values(status: TaskStatus) -> None:
    """TaskStatus accepts all 7 valid literal values."""
    task = BackgroundTask(
        id="t5",
        description=f"status {status}",
        agent_or_team="agent_e",
        prompt="check status",
        parent_session_id=None,
        child_session_id=None,
        status=status,
    )
    assert task.status == status


def test_task_status_importable_from_both_modules() -> None:
    """TaskStatus is importable from both __init__ and types."""
    assert TaskStatus is TaskStatusFromTypes
