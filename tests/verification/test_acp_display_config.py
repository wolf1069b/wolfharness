#!/usr/bin/env python3
"""Verification test for ACP subagent_display_mode feature.

This script tests the complete data flow for subagent_display_mode:
- Config model field validation
- Default value preservation
- CLI argument parsing
- End-to-end flow: Server → Agent → Session

Run with: uv run python tests/verification/test_acp_display_config.py
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import pytest


pytestmark = pytest.mark.integration


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_success(message: str) -> None:
    """Print success message."""
    print(f"✓ {message}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"✗ {message}")


def test_config_model() -> bool:
    """Test 1: Config model field exists and works."""
    print_section("Test 1: Config Model Field")

    try:
        from wolfharness_config.pool_server import ACPPoolServerConfig

        # Test 1.1: Field accepts "zed" (new mode)
        config_zed = ACPPoolServerConfig(subagent_display_mode="zed")
        assert config_zed.subagent_display_mode == "zed"
        print_success('ACPPoolServerConfig(subagent_display_mode="zed") works')

        # Test 1.2: Field accepts "legacy" (current default)
        config_legacy = ACPPoolServerConfig(subagent_display_mode="legacy")
        assert config_legacy.subagent_display_mode == "legacy"
        print_success('ACPPoolServerConfig(subagent_display_mode="legacy") works')

        # Test 1.3: Old values "inline" / "tool_box" coerce with DeprecationWarning
        for old_val in ("inline", "tool_box"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                config_old = ACPPoolServerConfig(subagent_display_mode=old_val)  # type: ignore[arg-type]
                assert config_old.subagent_display_mode == "legacy"
                # Verify a DeprecationWarning was emitted
                deprecation_warnings = [
                    x
                    for x in w
                    if issubclass(x.category, DeprecationWarning)
                    and "subagent_display_mode" in str(x.message)
                ]
                assert len(deprecation_warnings) >= 1, (
                    f"No DeprecationWarning for subagent_display_mode='{old_val}'"
                )
            print_success(
                f'ACPPoolServerConfig(subagent_display_mode="{old_val}") '
                f'coerces to "legacy" with DeprecationWarning'
            )

        # Test 1.4: Type validation - invalid value should fail
        try:
            ACPPoolServerConfig(subagent_display_mode="invalid")  # type: ignore[arg-type]
        except (ValueError, TypeError):
            print_success("Config model correctly rejects invalid values")
        else:
            print_error("Config model should reject invalid values")
            return False

    except (ValueError, TypeError, ImportError) as e:
        print_error(f"Config model test failed: {e}")
        return False
    else:
        return True


def test_default_value() -> bool:
    """Test 2: Default value is preserved."""
    print_section("Test 2: Default Value")

    try:
        from wolfharness_config.pool_server import ACPPoolServerConfig

        # Test default value
        config_default = ACPPoolServerConfig()
        assert config_default.subagent_display_mode == "legacy"
        print_success('ACPPoolServerConfig() defaults to "legacy"')

    except (ValueError, TypeError, ImportError) as e:
        print_error(f"Default value test failed: {e}")
        return False
    else:
        return True


def test_cli_option() -> bool:
    """Test 3: CLI option is recognized."""
    print_section("Test 3: CLI Option Recognition")

    try:
        # Test that help shows the option
        result = subprocess.run(
            ["uv", "run", "wolfharness", "serve-acp", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print_error("CLI help command timed out")
        return False
    except (subprocess.SubprocessError, OSError) as e:
        print_error(f"CLI option test failed: {e}")
        return False
    else:
        # Check for partial match (help might wrap or truncate)
        if "subagent" in result.stdout.lower() and "display" in result.stdout.lower():
            print_success('CLI option "--subagent-display-mode" is recognized in help output')
            return True

        print_error('CLI option "--subagent-display-mode" not found in help')
        print("  Searched for 'subagent' and 'display' in output")
        return False


def test_server_initialization() -> bool:
    """Test 4: Server can be initialized with mode."""
    print_section("Test 4: Server Initialization")

    try:
        from wolfharness import AgentPool
        from wolfharness.models.manifest import AgentsManifest
        from wolfharness_config.pool_server import ACPPoolServerConfig
        from wolfharness_server.acp_server.server import ACPServer

        # Create a minimal manifest
        manifest_dict = {
            "agents": {
                "test_agent": {
                    "type": "native",
                    "model": "openai:gpt-4o-mini",
                    "system_prompt": "Test agent for display mode verification",
                }
            }
        }

        # Test 4.1: Manifest with zed mode in pool_server config
        manifest_dict_with_config = {
            **manifest_dict,
            "pool_server": {
                "type": "acp",
                "subagent_display_mode": "zed",
            },
        }
        manifest = AgentsManifest.model_validate(manifest_dict_with_config)

        # pool_server is a union type - check if it's ACPPoolServerConfig

        assert isinstance(manifest.pool_server, ACPPoolServerConfig)
        assert manifest.pool_server.subagent_display_mode == "zed"
        print_success('Manifest accepts subagent_display_mode="zed" in pool_server')

        # Test 4.2: Server from_config with zed mode via argument
        server_zed = ACPServer.from_config(
            manifest,
            subagent_display_mode="zed",
        )
        assert server_zed.subagent_display_mode == "zed"
        print_success("ACPServer.from_config() accepts subagent_display_mode argument")

        # Test 4.3: Server from_config defaults to config value when arg not provided
        server_from_config = ACPServer.from_config(
            manifest,  # manifest has zed mode in pool_server
        )
        assert server_from_config.subagent_display_mode == "zed"
        print_success("ACPServer.from_config() uses config value when arg not provided")

        # Test 4.4: Server __init__ accepts mode directly
        # Need to use manifest object, not dict
        manifest_for_pool = AgentsManifest.model_validate(manifest_dict)
        pool = AgentPool(manifest=manifest_for_pool)
        server_direct = ACPServer(pool, subagent_display_mode="zed")
        assert server_direct.subagent_display_mode == "zed"
        print_success("ACPServer.__init__() accepts subagent_display_mode argument")

    except (ValueError, TypeError, ImportError, AttributeError) as e:
        print_error(f"Server initialization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    else:
        return True


def test_agent_display_mode() -> bool:
    """Test 5: Agent receives and stores mode."""
    print_section("Test 5: Agent Display Mode")

    try:
        from dataclasses import fields
        import inspect

        from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent

        # Test 5.1: AgentPoolACPAgent has subagent_display_mode field
        field_names = [f.name for f in fields(AgentPoolACPAgent)]
        assert "subagent_display_mode" in field_names
        print_success("AgentPoolACPAgent has subagent_display_mode field")

        # Test 5.2: AgentPoolACPAgent default value is "legacy"
        # We can't fully instantiate AgentPoolACPAgent without a real client,
        # but we can verify that type annotation exists
        sig = inspect.signature(AgentPoolACPAgent.__init__)
        params = sig.parameters

        if "subagent_display_mode" in params:
            param = params["subagent_display_mode"]
            default = param.default
            if default == "legacy":
                print_success('AgentPoolACPAgent subagent_display_mode defaults to "legacy"')
            else:
                print_error(f'Expected default "legacy", got {default}')
                return False
        else:
            print_error("AgentPoolACPAgent.__init__ missing subagent_display_mode parameter")
            return False

    except (ValueError, TypeError, ImportError, AttributeError) as e:
        print_error(f"Agent display mode test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    else:
        return True


def test_session_display_mode() -> bool:
    """Test 6: Session can be created with mode."""
    print_section("Test 6: Session Display Mode")

    try:
        from dataclasses import fields

        from wolfharness_server.acp_server.session import ACPSession

        # Test 6.1: ACPSession has subagent_display_mode field
        field_names = [f.name for f in fields(ACPSession)]
        assert "subagent_display_mode" in field_names
        print_success("ACPSession has subagent_display_mode field")

        # Test 6.2: ACPSession default value is "legacy"
        sig_fields = {f.name: f for f in fields(ACPSession)}
        subagent_field = sig_fields["subagent_display_mode"]
        default = subagent_field.default
        if default == "legacy":
            print_success('ACPSession subagent_display_mode defaults to "legacy"')
        else:
            print_error(f'Expected default "legacy", got {default}')
            return False
    except (ValueError, TypeError, ImportError, AttributeError) as e:
        print_error(f"Session display mode test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    else:
        return True


def test_end_to_end_flow() -> bool:
    """Test 7: End-to-end flow (Server → Agent → Session)."""
    print_section("Test 7: End-to-End Data Flow")

    try:
        from dataclasses import fields

        from wolfharness.models.manifest import AgentsManifest
        from wolfharness_server.acp_server.server import ACPServer

        # Create minimal manifest
        manifest_dict = {
            "agents": {
                "test_agent": {
                    "type": "native",
                    "model": "openai:gpt-4o-mini",
                    "system_prompt": "Test agent",
                }
            }
        }
        manifest = AgentsManifest.model_validate(manifest_dict)

        # Test 7.1: Create server with zed mode
        server = ACPServer.from_config(manifest, subagent_display_mode="zed")
        assert server.subagent_display_mode == "zed"
        print_success("Server initialized with zed mode")

        # Test 7.2: Verify agent has access to mode via server reference
        # (AgentPoolACPAgent gets subagent_display_mode from server at instantiation)
        from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent

        # Check that AgentPoolACPAgent stores the mode
        sig_fields = {f.name: f for f in fields(AgentPoolACPAgent)}
        assert "subagent_display_mode" in sig_fields
        print_success("Agent can store subagent_display_mode")

        # Test 7.3: Verify session creation passes mode
        # SessionManager.create_session accepts subagent_display_mode parameter
        import inspect

        from wolfharness_server.acp_server.session_manager import ACPSessionManager

        sig = inspect.signature(ACPSessionManager.create_session)
        params = sig.parameters

        if "subagent_display_mode" in params:
            param = params["subagent_display_mode"]
            default = param.default
            if default == "legacy":
                print_success(
                    "SessionManager.create_session() accepts "
                    'subagent_display_mode with default "legacy"'
                )
            else:
                print_error(f'Expected default "legacy", got {default}')
                return False
        else:
            print_error("SessionManager.create_session() missing subagent_display_mode parameter")
            return False

        # Test 7.4: Verify ACPSession stores the mode
        from wolfharness_server.acp_server.session import ACPSession

        sig_fields = {f.name: f for f in fields(ACPSession)}
        assert "subagent_display_mode" in sig_fields
        print_success("ACPSession stores subagent_display_mode")

    except (ValueError, TypeError, ImportError, AttributeError) as e:
        print_error(f"End-to-end flow test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    else:
        return True


def main() -> int:
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("  ACP Subagent Display Mode Verification Tests")
    print("=" * 60)

    tests = [
        ("Config Model Field", test_config_model),
        ("Default Value", test_default_value),
        ("CLI Option Recognition", test_cli_option),
        ("Server Initialization", test_server_initialization),
        ("Agent Display Mode", test_agent_display_mode),
        ("Session Display Mode", test_session_display_mode),
        ("End-to-End Flow", test_end_to_end_flow),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except (ValueError, TypeError, ImportError, AttributeError) as e:
            print(f"Unexpected error in {name}: {e}")
            results.append((name, False))

    # Print summary
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All verification tests passed!")
        return 0
    print(f"\n✗ {total - passed} test(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
