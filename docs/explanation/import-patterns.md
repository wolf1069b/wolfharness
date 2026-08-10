# Import Patterns

```python
# Avoid circular imports - use TYPE_CHECKING
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wolfharness.delegation import AgentPool

# Config models are in wolfharness_config to avoid circular deps
from wolfharness_config.teams import TeamConfig
```
