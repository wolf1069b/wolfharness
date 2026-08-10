"""Task definition and registry for agents."""

from __future__ import annotations

from typing import Any

from wolfharness.tasks.exceptions import JobRegistrationError
from wolfharness.utils.baseregistry import BaseRegistry
from wolfharness_config.task import Job


class TaskRegistry(BaseRegistry[str, Job[Any, Any]]):
    """Registry for managing tasks."""

    @property
    def _error_class(self) -> type[JobRegistrationError]:
        return JobRegistrationError

    def _validate_item(self, item: Any) -> Job[Any, Any]:
        from wolfharness_config.task import Job

        if not isinstance(item, Job):
            raise self._error_class(f"Expected Job, got {type(item)}")
        return item

    def register(self, key: str, item: Job[Any, Any], replace: bool = False) -> None:
        """Register a task with name.

        Creates a copy of the task with the name set.
        """
        task_copy = item.model_copy(update={"name": key})
        super().register(key, task_copy, replace)
