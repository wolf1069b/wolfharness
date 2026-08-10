"""Run concurrent safety tests and report baseline results.

Usage:
    python run_concurrent_tests.py
    python run_concurrent_tests.py --fail-fast
    python run_concurrent_tests.py --test-name test_concurrent_calls_complete
"""

import argparse
import asyncio
from pathlib import Path
import sys
import time


# Add wolfharness to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from wolfharness import Agent
from wolfharness.agents.events import StreamCompleteEvent


async def run_baseline_test():  # noqa: PLR0915
    """Run basic baseline test to verify setup."""
    print("=" * 70)
    print("RFC-0021 Pre-Flight Test Suite")
    print("=" * 70)
    print()

    agent = Agent(name="test_agent", model="test")

    # Test 1: Serial execution
    print("Test 1: Serial Execution Baseline")
    print("-" * 70)

    for i in range(3):
        event_count = 0
        async for event in agent.run_stream(f"Serial task {i}"):
            event_count += 1
            if isinstance(event, StreamCompleteEvent):
                break
        print(f"  Task {i}: {event_count} events ✓")

    print()

    # Test 2: Concurrent execution
    print("Test 2: Concurrent Execution (Primary RFC-0021 Test)")
    print("-" * 70)

    async def run_task(task_id: str) -> tuple[str, int, str]:
        """Run a task and return status."""
        start = time.perf_counter()
        event_count = 0

        try:
            async for event in agent.run_stream(f"Concurrent task {task_id}"):
                event_count += 1
                if isinstance(event, StreamCompleteEvent):
                    break
        except asyncio.CancelledError:
            duration = time.perf_counter() - start
            return (task_id, event_count, f"CANCELLED after {duration:.2f}s")
        except Exception as e:  # noqa: BLE001
            duration = time.perf_counter() - start
            return (task_id, event_count, f"ERROR: {e}")
        else:
            duration = time.perf_counter() - start
            return (task_id, event_count, f"completed in {duration:.2f}s")

    # Run 3 tasks concurrently
    results = await asyncio.gather(
        run_task("A"),
        run_task("B"),
        run_task("C"),
    )

    all_completed = True
    for task_id, event_count, status in results:
        ok = "✓" if "completed" in status else "✗"
        print(f"  Task {task_id}: {event_count} events - {status} {ok}")
        if "completed" not in status:
            all_completed = False

    print()

    # Test 3: Check for _cancelled pollution
    print("Test 3: Checking for _cancelled state pollution")
    print("-" * 70)

    agent2 = Agent(name="test2", model="test")
    print(f"  _cancelled initial: {agent2._cancelled}")

    # Run a quick task
    async for event in agent2.run_stream("Quick task"):
        if isinstance(event, StreamCompleteEvent):
            break

    print(f"  _cancelled after task: {agent2._cancelled}")

    if agent2._cancelled:
        print("  ⚠️  WARNING: _cancelled is True after normal completion!")
        print("       This is the 'finally block bug' identified in Pre-Flight Analysis.")
    else:
        print("  ✓ _cancelled correctly False after normal completion")

    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    if all_completed:
        print("✓ All concurrent tasks completed")
        print("  (Note: This test may pass intermittently; race conditions are timing-dependent)")
    else:
        print("✗ Some concurrent tasks failed - RFC-0021 fix needed")

    print()
    print("Next Steps:")
    print("  1. Review RFC-0021-Pre-Flight-Analysis.md for state inventory")
    print("  2. Run full test suite: pytest test_concurrent_safety.py -v")
    print("  3. Implement fix following RFC-0021 implementation plan")

    return all_completed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run RFC-0021 pre-flight tests")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.parse_args()

    try:
        success = asyncio.run(run_baseline_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
