## ADDED Requirements

### Requirement: ACP prompt processing SHALL use SessionPool exclusively
The ACP server SHALL route all prompt processing through `SessionPool.run_stream()`. There SHALL be no fallback path that calls `agent.run_stream()` directly. If `SessionPool` is unavailable, the system SHALL raise an error rather than silently falling back.

#### Scenario: SessionPool available
- **WHEN** `ACPSession.process_prompt()` is called and `SessionPool` is available on the agent's pool
- **THEN** the system SHALL route the prompt through `session_pool.run_stream()` and NOT fall back to direct agent invocation

#### Scenario: SessionPool unavailable
- **WHEN** `ACPSession.process_prompt()` is called and `SessionPool` is NOT available
- **THEN** the system SHALL raise a clear error indicating that SessionPool is required for ACP prompt processing

### Requirement: Legacy acp_agent.prompt() dead code SHALL be removed
The dead code path in `acp_agent.py` (lines 656-698) that calls `session.process_prompt()` when `_protocol_handler.handle_prompt()` returns `None` SHALL be removed. `handle_prompt()` always returns a `PromptResponse`, making this path unreachable.

#### Scenario: Prompt routing
- **WHEN** `acp_agent.prompt()` receives a prompt
- **THEN** it SHALL route exclusively through `_protocol_handler.handle_prompt()` and NOT fall through to the legacy `session.process_prompt()` path

### Requirement: ACPSessionManager SHALL separate lifecycle from protocol state
`ACPSessionManager._active` SHALL be renamed to `_acp_sessions: dict[str, ACPSession]`. Session lifecycle queries (existence, agent name, run status) SHALL be delegated to `SessionController.get_session()`. The `_acp_sessions` dict SHALL only store `ACPSession` runtime objects with protocol-specific state.

#### Scenario: Session lookup
- **WHEN** `ACPSessionManager.get_session(session_id)` is called
- **THEN** the system SHALL first check `SessionController.get_session(session_id)` for lifecycle state, then look up the `ACPSession` from `_acp_sessions` if the session is alive

#### Scenario: Session listing
- **WHEN** `ACPSessionManager.list_sessions()` is called
- **THEN** the system SHALL delegate to `SessionController.list_sessions()` for the session ID list, resolving `ACPSession` objects from `_acp_sessions` as needed

#### Scenario: Pool swap cleanup
- **WHEN** a pool swap occurs (`acp_agent.py:1110-1111`)
- **THEN** `_acp_sessions` SHALL be iterated for `ACPSession` cleanup, while lifecycle clearing SHALL be delegated to `SessionController`

### Requirement: RunFailedEvent SHALL have a single type definition
There SHALL be exactly one `RunFailedEvent` type, defined in `wolfharness/agents/events/events.py` and published via `RunHandle.fail()` to the EventBus. The signal-based `BaseAgent.RunFailedEvent` inner class SHALL be removed.

#### Scenario: Run failure reporting
- **WHEN** a run fails during execution
- **THEN** `RunHandle.fail()` SHALL publish `events.py:RunFailedEvent` to the EventBus, and all consumers SHALL receive this single event type
