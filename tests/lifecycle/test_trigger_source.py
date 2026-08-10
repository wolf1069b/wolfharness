"""Tests for TriggerSource implementations: ImmediateTrigger, ProtocolTrigger, stubs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.lifecycle import (
    ChannelTrigger,
    ImmediateTrigger,
    Prompt,
    ProtocolTrigger,
    ScheduledTrigger,
    TriggerSource,
)


if TYPE_CHECKING:
    from typing import Any


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ImmediateTrigger
# ---------------------------------------------------------------------------


def test_immediate_trigger_poll_returns_prompt_once() -> None:
    """Given an ImmediateTrigger with a prompt, When poll() is called,.

    Then a Prompt with the provided content SHALL be returned.
    """
    trigger = ImmediateTrigger("hello world")
    result = trigger.poll()
    assert result is not None
    assert isinstance(result, Prompt)
    assert result.content == "hello world"
    assert result.priority == "normal"


def test_immediate_trigger_poll_returns_none_after_first() -> None:
    """ImmediateTrigger returns None after the first poll.

    Given an ImmediateTrigger whose prompt was already polled, When poll()
    is called again, Then None SHALL be returned.
    """
    trigger = ImmediateTrigger("test")
    first = trigger.poll()
    assert first is not None
    second = trigger.poll()
    assert second is None


def test_immediate_trigger_poll_returns_none_on_third_call() -> None:
    """Third poll call still returns None.

    Given an ImmediateTrigger polled twice, When poll() is called a third
    time, Then None SHALL still be returned.
    """
    trigger = ImmediateTrigger("test")
    trigger.poll()
    trigger.poll()
    assert trigger.poll() is None


def test_immediate_trigger_subscribe_is_noop() -> None:
    """Given an ImmediateTrigger, When subscribe(run_loop) is called,.

    Then no action SHALL be taken (no error raised).
    """
    trigger = ImmediateTrigger("test")
    trigger.subscribe(object())  # Should not raise


def test_immediate_trigger_close_is_noop() -> None:
    """Given an ImmediateTrigger, When close() is called,.

    Then no action SHALL be taken (no error raised).
    """
    trigger = ImmediateTrigger("test")
    trigger.close()  # Should not raise


def test_immediate_trigger_satisfies_trigger_source_protocol() -> None:
    """Given an ImmediateTrigger instance, When checked against TriggerSource,.

    Then isinstance SHALL return True.
    """
    trigger = ImmediateTrigger("test")
    assert isinstance(trigger, TriggerSource)


# ---------------------------------------------------------------------------
# ProtocolTrigger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_protocol_trigger_deliver_then_poll_round_trip() -> None:
    r"""Given a ProtocolTrigger, When deliver(\"hello\") is called then poll(),.

    Then a Prompt with content=\"hello\" SHALL be returned.
    """
    trigger = ProtocolTrigger()
    await trigger.deliver("hello")
    result = trigger.poll()
    assert result is not None
    assert isinstance(result, Prompt)
    assert result.content == "hello"
    assert result.priority == "normal"


@pytest.mark.asyncio
async def test_protocol_trigger_deliver_with_asap_priority() -> None:
    r"""Given a ProtocolTrigger, When deliver(\"urgent\", priority=\"asap\") is.

    called then poll(), Then the Prompt SHALL have priority=\"asap\".
    """
    trigger = ProtocolTrigger()
    await trigger.deliver("urgent", priority="asap")
    result = trigger.poll()
    assert result is not None
    assert result.priority == "asap"


@pytest.mark.asyncio
async def test_protocol_trigger_poll_empty_returns_none() -> None:
    """Given a ProtocolTrigger with no delivered prompts, When poll() is.

    called, Then None SHALL be returned without blocking.
    """
    trigger = ProtocolTrigger()
    result = trigger.poll()
    assert result is None


@pytest.mark.asyncio
async def test_protocol_trigger_multiple_deliver_poll_fifo() -> None:
    """Given a ProtocolTrigger with multiple delivered prompts, When poll().

    is called repeatedly, Then prompts SHALL be returned in FIFO order.
    """
    trigger = ProtocolTrigger()
    await trigger.deliver("first")
    await trigger.deliver("second")
    await trigger.deliver("third")

    first = trigger.poll()
    second = trigger.poll()
    third = trigger.poll()
    fourth = trigger.poll()

    assert first is not None
    assert first.content == "first"
    assert second is not None
    assert second.content == "second"
    assert third is not None
    assert third.content == "third"
    assert fourth is None


@pytest.mark.asyncio
async def test_protocol_trigger_subscribe_stores_run_loop_ref() -> None:
    """Given a ProtocolTrigger, When subscribe(run_loop) is called,.

    Then the RunLoop reference SHALL be stored.
    """
    trigger = ProtocolTrigger()
    run_loop: Any = object()
    trigger.subscribe(run_loop)
    assert trigger._run_loop is run_loop


@pytest.mark.asyncio
async def test_protocol_trigger_close_drains_queue() -> None:
    """Given a ProtocolTrigger with pending prompts, When close() is called,.

    Then the queue SHALL be drained and no prompts remain.
    """
    trigger = ProtocolTrigger()
    await trigger.deliver("msg1")
    await trigger.deliver("msg2")
    trigger.close()
    assert trigger.poll() is None


@pytest.mark.asyncio
async def test_protocol_trigger_satisfies_trigger_source_protocol() -> None:
    """Given a ProtocolTrigger instance, When checked against TriggerSource,.

    Then isinstance SHALL return True.
    """
    trigger = ProtocolTrigger()
    assert isinstance(trigger, TriggerSource)


# ---------------------------------------------------------------------------
# ScheduledTrigger (stub)
# ---------------------------------------------------------------------------


def test_scheduled_trigger_poll_raises_not_implemented() -> None:
    """Given a ScheduledTrigger stub, When poll() is called,.

    Then NotImplementedError SHALL be raised.
    """
    trigger = ScheduledTrigger()
    with pytest.raises(NotImplementedError):
        trigger.poll()


def test_scheduled_trigger_subscribe_raises_not_implemented() -> None:
    """Given a ScheduledTrigger stub, When subscribe() is called,.

    Then NotImplementedError SHALL be raised.
    """
    trigger = ScheduledTrigger()
    with pytest.raises(NotImplementedError):
        trigger.subscribe(object())


def test_scheduled_trigger_close_raises_not_implemented() -> None:
    """Given a ScheduledTrigger stub, When close() is called,.

    Then NotImplementedError SHALL be raised.
    """
    trigger = ScheduledTrigger()
    with pytest.raises(NotImplementedError):
        trigger.close()


def test_scheduled_trigger_stores_config() -> None:
    """Given a ScheduledTrigger constructed with a config dict, When accessed,.

    Then the config SHALL be stored as provided.
    """
    config = {"interval": 60, "template": "Run check {{ date }}"}
    trigger = ScheduledTrigger(config=config)
    assert trigger.config == config


def test_scheduled_trigger_default_config_empty() -> None:
    """Given a ScheduledTrigger constructed without config, When accessed,.

    Then config SHALL be an empty dict.
    """
    trigger = ScheduledTrigger()
    assert trigger.config == {}


# ---------------------------------------------------------------------------
# ChannelTrigger (stub)
# ---------------------------------------------------------------------------


def test_channel_trigger_poll_raises_not_implemented() -> None:
    """Given a ChannelTrigger stub, When poll() is called,.

    Then NotImplementedError SHALL be raised.
    """
    trigger = ChannelTrigger()
    with pytest.raises(NotImplementedError):
        trigger.poll()


def test_channel_trigger_subscribe_raises_not_implemented() -> None:
    """Given a ChannelTrigger stub, When subscribe() is called,.

    Then NotImplementedError SHALL be raised.
    """
    trigger = ChannelTrigger()
    with pytest.raises(NotImplementedError):
        trigger.subscribe(object())


def test_channel_trigger_close_raises_not_implemented() -> None:
    """Given a ChannelTrigger stub, When close() is called,.

    Then NotImplementedError SHALL be raised.
    """
    trigger = ChannelTrigger()
    with pytest.raises(NotImplementedError):
        trigger.close()


def test_channel_trigger_stores_config() -> None:
    """Given a ChannelTrigger constructed with a config dict, When accessed,.

    Then the config SHALL be stored as provided.
    """
    config = {"channel": "telegram", "token": "abc123"}
    trigger = ChannelTrigger(config=config)
    assert trigger.config == config


def test_channel_trigger_default_config_empty() -> None:
    """Given a ChannelTrigger constructed without config, When accessed,.

    Then config SHALL be an empty dict.
    """
    trigger = ChannelTrigger()
    assert trigger.config == {}
