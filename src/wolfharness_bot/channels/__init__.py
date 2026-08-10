"""Chat channels module with plugin architecture."""

from __future__ import annotations

from wolfharness_bot.channels.base import BaseChannel
from wolfharness_bot.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
