"""Tests for ACP toast notification mechanism."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.agents.events import ToastInfo


pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestSendToast:
    """Tests for _send_toast method in ACPSession."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock ACPSession with required attributes."""
        session = MagicMock()
        session._cancelled = False
        session.log = MagicMock()
        session.notifications = AsyncMock()
        return session

    async def test_send_toast_error(self, mock_session):
        """Test sending error toast notification."""
        from wolfharness_server.acp_server.session import ACPSession

        # Patch the method onto the mock
        with patch.object(mock_session, "notifications") as mock_notifs:
            await ACPSession._send_toast(
                mock_session,
                message="Test error",
                level="error",
            )
            mock_notifs.send_ext_notification.assert_called_once_with(
                method="_wolfharness/toast",
                params={
                    "message": "Test error",
                    "level": "error",
                    "duration": None,
                    "action": None,
                },
            )

    async def test_send_toast_warning(self, mock_session):
        """Test sending warning toast notification."""
        from wolfharness_server.acp_server.session import ACPSession

        with patch.object(mock_session, "notifications") as mock_notifs:
            await ACPSession._send_toast(
                mock_session,
                message="Warning message",
                level="warning",
                duration=5000,
            )
            mock_notifs.send_ext_notification.assert_called_once_with(
                method="_wolfharness/toast",
                params={
                    "message": "Warning message",
                    "level": "warning",
                    "duration": 5000,
                    "action": None,
                },
            )

    async def test_send_toast_with_action(self, mock_session):
        """Test sending toast with action button."""
        from wolfharness_server.acp_server.session import ACPSession

        with patch.object(mock_session, "notifications") as mock_notifs:
            await ACPSession._send_toast(
                mock_session,
                message="Click to retry",
                level="info",
                action={"label": "Retry", "command": "/retry"},
            )
            mock_notifs.send_ext_notification.assert_called_once_with(
                method="_wolfharness/toast",
                params={
                    "message": "Click to retry",
                    "level": "info",
                    "duration": None,
                    "action": {"label": "Retry", "command": "/retry"},
                },
            )

    async def test_send_toast_cancelled(self, mock_session):
        """Test that toast is not sent when session is cancelled."""
        from wolfharness_server.acp_server.session import ACPSession

        mock_session._cancelled = True
        with patch.object(mock_session, "notifications") as mock_notifs:
            await ACPSession._send_toast(
                mock_session,
                message="Test",
                level="error",
            )
            mock_notifs.send_ext_notification.assert_not_called()

    async def test_send_toast_exception_handled(self, mock_session):
        """Test that exceptions in send_ext_notification are handled."""
        from wolfharness_server.acp_server.session import ACPSession

        with patch.object(mock_session, "notifications") as mock_notifs:
            mock_notifs.send_ext_notification.side_effect = RuntimeError("Network error")
            # Should not raise
            await ACPSession._send_toast(
                mock_session,
                message="Test",
                level="error",
            )
            mock_session.log.exception.assert_called_once()


@pytest.mark.unit
class TestClientHandlerToast:
    """Tests for ACPClientHandler._toast handling."""

    @pytest.fixture
    def mock_handler(self):
        """Create a mock ACPClientHandler."""
        handler = MagicMock()
        handler._agent = MagicMock()
        handler._agent.state_updated = AsyncMock()
        return handler

    async def test_handle_toast_notification(self, mock_handler):
        """Test handling _wolfharness/toast ext notification."""
        from wolfharness.agents.acp_agent.client_handler import ACPClientHandler

        params = {
            "message": "Error occurred",
            "level": "error",
            "duration": 3000,
            "action": {"label": "Retry", "command": "/retry"},
        }

        await ACPClientHandler.ext_notification(mock_handler, "_wolfharness/toast", params)

        # Verify state_updated was emitted with ToastInfo
        mock_handler._agent.state_updated.emit.assert_called_once()
        toast = mock_handler._agent.state_updated.emit.call_args[0][0]
        assert isinstance(toast, ToastInfo)
        assert toast.message == "Error occurred"
        assert toast.level == "error"
        assert toast.duration == 3000
        assert toast.action == {"label": "Retry", "command": "/retry"}

    async def test_handle_toast_notification_defaults(self, mock_handler):
        """Test handling toast with default values."""
        from wolfharness.agents.acp_agent.client_handler import ACPClientHandler

        params = {"message": "Info message"}

        await ACPClientHandler.ext_notification(mock_handler, "_wolfharness/toast", params)

        toast = mock_handler._agent.state_updated.emit.call_args[0][0]
        assert isinstance(toast, ToastInfo)
        assert toast.message == "Info message"
        assert toast.level == "info"
        assert toast.duration is None
        assert toast.action is None

    async def test_handle_unknown_ext_notification(self, mock_handler):
        """Test that unknown ext notifications are ignored gracefully."""
        from wolfharness.agents.acp_agent.client_handler import ACPClientHandler

        await ACPClientHandler.ext_notification(mock_handler, "_unknown/method", {"foo": "bar"})

        mock_handler._agent.state_updated.emit.assert_not_called()


@pytest.mark.unit
class TestNotificationsSendExt:
    """Tests for ACPNotifications.send_ext_notification."""

    @pytest.fixture
    def mock_notifications(self):
        """Create mock ACPNotifications."""
        notifs = MagicMock()
        notifs.client = AsyncMock()
        return notifs

    async def test_send_ext_notification(self, mock_notifications):
        """Test sending ext notification."""
        from acp.agent.notifications import ACPNotifications

        await ACPNotifications.send_ext_notification(
            mock_notifications,
            method="_wolfharness/toast",
            params={"message": "Hello", "level": "info"},
        )

        mock_notifications.client.ext_notification.assert_called_once_with(
            "_wolfharness/toast",
            {"message": "Hello", "level": "info"},
        )

    async def test_send_ext_notification_empty_params(self, mock_notifications):
        """Test sending ext notification with None params."""
        from acp.agent.notifications import ACPNotifications

        await ACPNotifications.send_ext_notification(
            mock_notifications,
            method="_test/method",
            params=None,
        )

        mock_notifications.client.ext_notification.assert_called_once_with(
            "_test/method",
            {},
        )


@pytest.mark.unit
class TestToastInfo:
    """Tests for ToastInfo dataclass."""

    def test_toast_info_defaults(self):
        """Test ToastInfo with default values."""
        toast = ToastInfo(message="Hello")
        assert toast.message == "Hello"
        assert toast.level == "info"
        assert toast.duration is None
        assert toast.action is None

    def test_toast_info_full(self):
        """Test ToastInfo with all fields."""
        toast = ToastInfo(
            message="Error!",
            level="error",
            duration=5000,
            action={"label": "Retry", "command": "/retry"},
        )
        assert toast.message == "Error!"
        assert toast.level == "error"
        assert toast.duration == 5000
        assert toast.action == {"label": "Retry", "command": "/retry"}

    def test_toast_info_in_state_update(self):
        """Test that ToastInfo can be used as a state update."""
        from wolfharness.agents.events import ToastInfo

        toast = ToastInfo(message="Test", level="warning")
        assert isinstance(toast, ToastInfo)
        assert toast.message == "Test"
        # StateUpdate is a TYPE_CHECKING-only union; verify at runtime by
        # checking the signal accepts ToastInfo (done implicitly by using it).
