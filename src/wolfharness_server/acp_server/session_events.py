"""Event handling mixin for ACPSession.

Extracted from session.py as part of the session-debt-cleanup file split.
Contains state update event handling and available commands update methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tokonomics.model_discovery.model_info import ModelInfo

from wolfharness.agents.events.events import ToastInfo
from wolfharness.agents.modes import ConfigOptionChanged, ModeInfo
from wolfharness.log import get_logger


if TYPE_CHECKING:
    from wolfharness.agents.base_agent import StateUpdate


logger = get_logger(__name__)


class ACPSessionEventsMixin:
    """Mixin providing event handling methods for ACPSession.

    Contains state update event handling (forwarding mode/model/config changes
    to ACP clients) and available commands update methods.

    All attributes are provided by the main :class:`ACPSession` dataclass.
    Type annotations are declared under ``TYPE_CHECKING`` to avoid being
    treated as dataclass fields.
    """

    if TYPE_CHECKING:
        agent: Any  # BaseAgent[Any, Any]
        log: Any
        _remote_commands: list[Any]  # list[AvailableCommand]
        command_store: Any  # CommandStore
        notifications: Any  # ACPNotifications
        session_id: str

        def _notify_command_update(self) -> None: ...
        def get_acp_commands(self) -> list[Any]: ...

    async def _on_state_updated(self, state: StateUpdate) -> None:
        """Handle state update signal from agent - forward to ACP client."""
        from acp.schema import (
            AvailableCommandsUpdate,
            ConfigOptionUpdate,
            CurrentModelUpdate,
            CurrentModeUpdate,
        )
        from wolfharness_server.acp_server.acp_agent import get_session_config_options

        update: CurrentModeUpdate | CurrentModelUpdate | ConfigOptionUpdate
        match state:
            case ModeInfo(id=mode_id):
                update = CurrentModeUpdate(current_mode_id=mode_id)
                self.log.debug("Forwarding mode change to client", mode_id=mode_id)
            case ModelInfo(id=model_id):
                update = CurrentModelUpdate(current_model_id=model_id)
                self.log.debug("Forwarding model change to client", model_id=model_id)
            case AvailableCommandsUpdate(available_commands=cmds):
                # Store remote commands and send merged list
                self._remote_commands = list(cmds)
                await self.send_available_commands_update()
                self.log.debug("Merged and sent commands update to client")
                return
            case ToastInfo():
                self.log.debug("Received ToastInfo, ignoring")
                return
            case ConfigOptionChanged(config_id=config_id, value_id=value_id):
                # Get full config_options from agent (required by ACP protocol)
                config_options = await get_session_config_options(self.agent)
                # Update the changed option's current_value
                if opt := next((i for i in config_options if i.id == config_id), None):
                    opt.current_value = value_id
                # Convert our core type to ACP type with full config_options
                update = ConfigOptionUpdate(
                    config_id=config_id,
                    value_id=value_id,
                    config_options=config_options,
                )
                self.log.debug("Config option change", config_id=config_id, value_id=value_id)
                # For permissions, also send legacy CurrentModeUpdate (still needed)
                if config_id == "permissions":
                    await self.notifications.update_session_mode(value_id)
                    self.log.debug("Also sent legacy mode update", mode_id=value_id)
        await self.notifications.send_update(update)

    async def send_available_commands_update(self) -> None:
        """Send current available commands to client.

        Merges local commands from command_store with any remote commands
        from nested ACP agents.
        """
        commands = [*self.get_acp_commands(), *self._remote_commands]
        try:
            await self.notifications.update_commands(commands)
        except Exception:
            self.log.exception("Failed to send available commands update")
