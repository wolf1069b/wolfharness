"""Tests for Team execution."""

from __future__ import annotations

from typing import Any

import pytest

from wolfharness import Agent, ChatMessage
from wolfharness.delegation.base_team import BaseTeam


pytestmark = pytest.mark.unit


async def test_team_parallel_execution():
    """Test that team runs all agents in parallel and collects responses."""
    # Create three agents that append their name to input
    a1 = Agent("a1", system_prompt="Append 'a1'", model="test")
    a2 = Agent("a2", system_prompt="Append 'a2'", model="test")
    a3 = Agent("a3", system_prompt="Append 'a3'", model="test")

    team = BaseTeam([a1, a2, a3])
    result = await team.execute("test")

    # Check that we got responses from all agents
    assert len(result) == 3
    agent_names = {r.agent_name for r in result}
    assert agent_names == {"a1", "a2", "a3"}

    # Check that stats were collected
    assert len(team.execution_stats.messages) == 3
    assert all(isinstance(msg, ChatMessage) for msg in team.execution_stats.messages)


async def test_team_shared_prompt():
    """Test that shared prompt is prepended to individual prompts."""

    # Create agents that echo their input
    def echo(prompt: str) -> str:
        return prompt

    a1 = Agent.from_callback(echo, name="a1")
    a2 = Agent.from_callback(echo, name="a2")

    # Create team with shared prompt
    team = BaseTeam([a1, a2], shared_prompt="Common instruction: ")
    result = await team.execute("specific task")

    # Each agent should get both prompts
    assert len(result) == 2
    for response in result:
        assert response.message
        assert "Common instruction" in str(response.message.content)
        assert "specific task" in str(response.message.content)


async def test_nested_teams():
    """Test nesting BaseTeams inside each other."""
    # Create basic agents
    a1 = Agent("a1", model="test")
    a2 = Agent("a2", model="test")
    a3 = Agent("a3", model="test")

    # Case 1: parallel inside sequential
    team = a1 & a2  # parallel team
    execution = team | a3  # sequential with parallel team + agent
    result = await execution.run("test message")
    assert isinstance(result, ChatMessage)
    assert len(execution.execution_stats.messages) == 2  # Team(a1+a2) + a3


async def test_nested_team_run():
    """Test nesting sequential inside parallel."""
    # Create basic agents
    a1 = Agent("a1", model="test")
    a2 = Agent("a2", model="test")
    a3 = Agent("a3", model="test")
    a4 = Agent("a4", model="test")

    # Case 2: sequential inside parallel
    sequential = a1 | a2  # sequential
    parallel_team = BaseTeam([sequential, a3, a4])  # parallel containing sequential + Agents

    result = await parallel_team.run("test message")
    assert isinstance(result, ChatMessage)
    assert len(parallel_team.execution_stats.messages) == 3  # sequential(a1+a2) + a3 + a4

    # Test iteration with nested sequential
    messages = [msg async for msg in parallel_team.run_iter("test message")]
    assert len(messages) == 3


async def test_simple_team_run_iter():
    """Test run_iter with a simple team of agents."""
    a1 = Agent("a1", model="test")
    a2 = Agent("a2", model="test")

    team = BaseTeam([a1, a2])

    messages = [msg async for msg in team.run_iter("test message")]
    assert len(messages) == 2
    assert {msg.name for msg in messages} == {"a1", "a2"}


async def test_sequential_run_iter():
    """Test run_iter with a sequential execution."""
    a1 = Agent("a1", model="test")
    a2 = Agent("a2", model="test")

    sequential: BaseTeam[None, Any] = BaseTeam([a1, a2], mode="sequential", name="seq")

    messages = [msg async for msg in sequential.run_iter("test message")]
    assert len(messages) == 2
    assert [msg.name for msg in messages] == ["a1", "a2"]


async def test_simple_team_with_teamrun_iter():
    """Test run_iter with a parallel team containing a sequential sub-team."""
    a1 = Agent("a1", model="test")
    a2 = Agent("a2", model="test")
    a3 = Agent("a3", model="test")

    # Sequential execution as team member (using | operator)
    sequential = a1 | a2
    # Parallel team with two members: sequential and a3
    team = BaseTeam([sequential, a3])

    messages = [msg async for msg in team.run_iter("test message")]
    assert len(messages) == 2

    senders = {msg.name for msg in messages}
    assert senders == {sequential.name, "a3"}

    # Verify sequential message has metadata about its internal execution
    seq_msg = next(msg for msg in messages if msg.name == sequential.name)
    assert "execution_order" in seq_msg.metadata


async def test_team_run_iter_execution_order():
    """Test that run_iter preserves execution order within sequential parts."""
    a1 = Agent("a1", model="test")
    a2 = Agent("a2", model="test")
    a3 = Agent("a3", model="test")

    # Sequential execution
    sequential = BaseTeam([a1, a2], mode="sequential", name="sequential")
    # Parallel team with sequential + single agent
    team = BaseTeam([sequential, a3], name="parallel")

    messages = [msg async for msg in team.run_iter("test message")]
    seq_msgs = [msg for msg in messages if msg.name == sequential.name]
    seq_msg = seq_msgs[0]
    assert [msg.name for msg in seq_msg.associated_messages] == ["a1", "a2"]


async def test_team_operators():  # noqa: PLR0915
    """Test team combination operators (& and |)."""
    a1 = Agent("a1", model="test")
    a2 = Agent("a2", model="test")
    a3 = Agent("a3", model="test")
    a4 = Agent("a4", model="test")

    # Test parallel combinations (&)
    team1 = a1 & a2
    assert isinstance(team1, BaseTeam)
    assert team1.mode == "parallel"
    assert len(team1.nodes) == 2
    assert list(team1.nodes) == [a1, a2]

    team2 = team1 & a3
    assert isinstance(team2, BaseTeam)
    assert team2.mode == "parallel"
    assert len(team2.nodes) == 3
    assert list(team2.nodes) == [a1, a2, a3]

    # Combining teams - should flatten
    other_team = a3 & a4
    combined = team1 & other_team
    assert isinstance(combined, BaseTeam)
    assert combined.mode == "parallel"
    assert len(combined.nodes) == 4
    assert list(combined.nodes) == [a1, a2, a3, a4]

    # Test sequential combinations (|)
    seq1 = a1 | a2
    assert isinstance(seq1, BaseTeam)
    assert seq1.mode == "sequential"
    assert len(seq1.nodes) == 2
    assert list(seq1.nodes) == [a1, a2]

    # Adding to sequential - should extend
    seq2 = seq1 | a3
    assert seq2 is seq1  # Same instance
    assert len(seq1.nodes) == 3
    assert list(seq1.nodes) == [a1, a2, a3]

    # Complex combinations
    team3 = a1 & a2  # parallel
    seq3 = a3 | a4  # sequential

    # Sequential with parallel member
    combined_1 = team3 | seq3
    assert isinstance(combined_1, BaseTeam)
    assert combined_1.mode == "sequential"
    assert len(combined_1.nodes) == 2
    assert combined_1.nodes[0] is team3
    assert isinstance(combined_1.nodes[1], BaseTeam)
    assert combined_1.nodes[1].mode == "sequential"

    # Parallel with sequential member
    combined_2 = BaseTeam([team3, seq3])
    assert isinstance(combined_2, BaseTeam)
    assert combined_2.mode == "parallel"
    assert len(combined_2.nodes) == 2
    assert combined_2.nodes[0] is team3
    assert isinstance(combined_2.nodes[1], BaseTeam)
    assert combined_2.nodes[1].mode == "sequential"

    # Test actual execution
    result = await combined_1.run("test")
    assert isinstance(result, ChatMessage)
    assert len(combined_1.execution_stats.messages) == len(combined_1.nodes)

    result = await combined_2.run("test")
    assert isinstance(result, ChatMessage)
    assert len(combined_2.execution_stats.messages) == len(combined_2.nodes)


if __name__ == "__main__":
    pytest.main([__file__])
