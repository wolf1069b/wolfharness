# Common Patterns

## Creating an AgentPool

```python
async with AgentPool("config.yml") as pool:
    agent = pool.get_agent("agent_name")
    result = await agent.run("prompt")
```

## Running Tests on Modified Code

```bash
# Find relevant tests
pytest tests/path/to/test_file.py -k "test_pattern"

# Quick sanity check (unit tests only)
pytest -m unit --no-cov

# Full validation
pytest && mypy src/ && ruff check src/
```

## Debugging Agent Issues

1. Enable verbose logging (set `OBSERVABILITY_ENABLED=true`)
2. Check storage database for interaction history
3. Use `TestModel` for isolated testing
4. Add `--log-cli-level=DEBUG` to pytest

## Working with YAML Configs

- Examples in `site/examples/*/config.yml`
- Schema reference auto-generated from Pydantic models
- Validate with: `python -m wolfharness_config.manifest config.yml`
