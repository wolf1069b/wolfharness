## ADDED Requirements

### Requirement: YAML team definitions generate Mermaid diagrams
AgentPool SHALL enable Mermaid diagram generation for YAML-defined team and workflow definitions via `pydantic_graph`'s visualization support.

#### Scenario: Parallel team diagram generation
- **WHEN** a YAML parallel team is defined
- **THEN** `GraphBuilder` generates a Mermaid diagram showing the Fork/Join structure

#### Scenario: Sequential team diagram generation
- **WHEN** a YAML sequential team is defined
- **THEN** `GraphBuilder` generates a Mermaid diagram showing the node chain

#### Scenario: CLI diagram access
- **WHEN** user runs `wolfharness visualize <team_name>`
- **THEN** a Mermaid diagram is printed for the specified YAML team/workflow
