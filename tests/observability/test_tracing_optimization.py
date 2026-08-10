"""Unit tests for logfire tracing optimization.

Tests cover:
- _otel_log_processor (no span creation, structured attributes preserved)
- ObservabilityRegistry config fields and API
- Per-run trace context (run.message root span on RunHandle)
- SessionState no longer has trace_context (replaced by per-run spans)
- Conditional auto-instrumentation (instrument_pydantic_ai, instrument_mcp, instrument_fastapi)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestOtelLogProcessor:
    """Tests for _otel_log_processor structlog processor."""

    def test_calls_logfire_log_without_creating_span(self):
        """_otel_log_processor calls logfire.log() but does not create a span."""
        from wolfharness.log import _otel_log_processor

        event_dict: dict[str, object] = {
            "event": "test message",
            "level": "info",
            "session_id": "s123",
        }

        with patch("wolfharness.log.logfire") as mock_logfire:
            _otel_log_processor(None, "info", event_dict)

            mock_logfire.log.assert_called_once()
            call_kwargs = mock_logfire.log.call_args
            assert call_kwargs.kwargs["level"] == "info"
            assert call_kwargs.kwargs["msg_template"] == "test message"
            # session_id should be in attributes but "event" should not
            attrs = call_kwargs.kwargs["attributes"]
            assert attrs["session_id"] == "s123"
            assert "event" not in attrs

    def test_returns_event_dict_unchanged(self):
        """_otel_log_processor returns event_dict unchanged for downstream renderers."""
        from wolfharness.log import _otel_log_processor

        event_dict: dict[str, object] = {
            "event": "test message",
            "level": "info",
        }

        with patch("wolfharness.log.logfire"):
            result = _otel_log_processor(None, "info", event_dict)

        # event_dict must still have "event" key for renderer
        assert "event" in result
        assert result["event"] == "test message"
        assert result["level"] == "info"

    def test_handles_missing_event_key(self):
        """_otel_log_processor handles missing 'event' key gracefully."""
        from wolfharness.log import _otel_log_processor

        event_dict: dict[str, object] = {"level": "debug"}

        with patch("wolfharness.log.logfire") as mock_logfire:
            _otel_log_processor(None, "debug", event_dict)

            # Should fall back to method_name as msg
            call_kwargs = mock_logfire.log.call_args
            assert call_kwargs.kwargs["msg_template"] == "debug"

    def test_does_not_raise_on_logfire_error(self):
        """_otel_log_processor swallows logfire errors to never break logging."""
        from wolfharness.log import _otel_log_processor

        event_dict: dict[str, object] = {"event": "test", "level": "info"}

        with patch("wolfharness.log.logfire") as mock_logfire:
            mock_logfire.log.side_effect = RuntimeError("OTel not configured")
            # Should not raise
            result = _otel_log_processor(None, "info", event_dict)

        assert result == event_dict


@pytest.mark.unit
class TestObservabilityConfig:
    """Tests for new BaseObservabilityConfig fields."""

    def test_new_fields_exist_with_correct_defaults(self):
        """BaseObservabilityConfig has instrument_* fields with correct defaults."""
        from wolfharness_config.observability import CustomObservabilityConfig

        config = CustomObservabilityConfig(type="custom", endpoint="http://localhost:4318")
        assert config.instrument_pydantic_ai is False
        assert config.instrument_mcp is False
        assert config.instrument_fastapi is True

    def test_fields_can_be_overridden(self):
        """instrument_* fields can be overridden via config."""
        from wolfharness_config.observability import CustomObservabilityConfig

        config = CustomObservabilityConfig(
            type="custom",
            endpoint="http://localhost:4318",
            instrument_pydantic_ai=True,
            instrument_mcp=True,
            instrument_fastapi=False,
        )
        assert config.instrument_pydantic_ai is True
        assert config.instrument_mcp is True
        assert config.instrument_fastapi is False


@pytest.mark.unit
class TestObservabilityRegistry:
    """Tests for ObservabilityRegistry API extensions."""

    def test_is_configured_returns_false_before_configuration(self):
        """is_configured() returns False before configure_observability() is called."""
        from wolfharness.observability.observability_registry import (
            ObservabilityRegistry,
        )

        registry = ObservabilityRegistry()
        assert registry.is_configured() is False

    def test_config_returns_none_before_configuration(self):
        """Config property returns None before configuration."""
        from wolfharness.observability.observability_registry import (
            ObservabilityRegistry,
        )

        registry = ObservabilityRegistry()
        assert registry.config is None

    def test_conditional_instrumentation(self):
        """configure_observability respects instrument_* config fields."""
        from wolfharness.observability.observability_registry import (
            ObservabilityRegistry,
        )
        from wolfharness_config.observability import (
            CustomObservabilityConfig,
            ObservabilityConfig,
        )

        provider = CustomObservabilityConfig(
            type="custom",
            endpoint="http://localhost:4318",
            instrument_pydantic_ai=False,
            instrument_mcp=False,
        )
        config = ObservabilityConfig(enabled=True, provider=provider)

        registry = ObservabilityRegistry()
        with (
            patch("wolfharness.observability.observability_registry.logfire") as mock_lf,
            patch("wolfharness.observability.observability_registry._setup_otel_environment"),
        ):
            registry.configure_observability(config)

            # Should NOT call instrument_pydantic_ai or instrument_mcp
            mock_lf.instrument_pydantic_ai.assert_not_called()
            mock_lf.instrument_mcp.assert_not_called()
            # Should call configure
            mock_lf.configure.assert_called_once()
            assert registry.is_configured() is True
            assert registry.config is not None
            assert registry.config.instrument_pydantic_ai is False

    def test_conditional_instrumentation_enabled(self):
        """configure_observability calls instrumentation when enabled."""
        from wolfharness.observability.observability_registry import (
            ObservabilityRegistry,
        )
        from wolfharness_config.observability import (
            CustomObservabilityConfig,
            ObservabilityConfig,
        )

        provider = CustomObservabilityConfig(
            type="custom",
            endpoint="http://localhost:4318",
            instrument_pydantic_ai=True,
            instrument_mcp=True,
        )
        config = ObservabilityConfig(enabled=True, provider=provider)

        registry = ObservabilityRegistry()
        with (
            patch("wolfharness.observability.observability_registry.logfire") as mock_lf,
            patch("wolfharness.observability.observability_registry._setup_otel_environment"),
        ):
            registry.configure_observability(config)

            mock_lf.instrument_pydantic_ai.assert_called_once()
            mock_lf.instrument_mcp.assert_called_once()


@pytest.mark.unit
class TestPerRunTrace:
    """Tests for per-run trace context (run.message root span).

    Each RunHandle gets its own trace via a ``run.message`` root span
    created in ``_start_run_handle``. The span context is attached in
    ``_consume_run`` so all run-internal spans share the same trace_id.
    """

    def test_run_handle_has_run_span_fields(self):
        """RunHandle has _run_span and _run_context fields (default None)."""
        from wolfharness.orchestrator.run import RunHandle

        handle = RunHandle(run_id="test", session_id="s1", agent_type="native")
        assert handle._run_span is None
        assert handle._run_context is None

    def test_session_state_no_longer_has_trace_context(self):
        """SessionState no longer has trace_context field (removed in per-run design)."""
        from wolfharness.orchestrator.session_controller import SessionState

        state = SessionState(session_id="test", agent_name="test")
        assert not hasattr(state, "trace_context")

    def test_session_controller_no_longer_has_get_trace_context(self):
        """SessionController no longer has get_trace_context method."""
        from wolfharness.orchestrator.session_controller import SessionController

        assert not hasattr(SessionController, "get_trace_context")

    def test_run_message_span_creates_new_trace(self):
        """run.message span creates a new trace (not inheriting caller context)."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            SimpleSpanProcessor,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        # Use a real TracerProvider (not the default no-op) so spans get real trace IDs
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        default_provider = trace.get_tracer_provider()
        trace.set_tracer_provider(provider)
        try:
            tracer = trace.get_tracer("test")
            # Create a parent span to verify run.message does NOT inherit it
            parent = tracer.start_span("parent")
            parent_ctx = trace.set_span_in_context(parent)

            from opentelemetry.context import Context, attach, detach

            token = attach(parent_ctx)
            try:
                # Pass empty Context to create a new trace (not inheriting parent)
                run_span = tracer.start_span("run.message", context=Context())

                # run_span should have a different trace_id than parent
                parent_ctx_info = parent.get_span_context()
                run_ctx_info = run_span.get_span_context()
                assert parent_ctx_info.trace_id != run_ctx_info.trace_id
                assert run_ctx_info.trace_id != 0  # valid trace

                run_span.end()
                parent.end()
            finally:
                detach(token)
        finally:
            trace.set_tracer_provider(default_provider)
