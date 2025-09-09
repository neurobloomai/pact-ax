# PACT-AX: Agent Collaboration Layer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

PACT-AX provides primitives for safe collaboration, context sharing, and knowledge transfer between heterogeneous AI agents.

## Quick Start

```python
from pact_ax.primitives.context_share import ContextShareManager

# Create context sharing manager
manager = ContextShareManager("agent-001")

# Share context with another agent
context_packet = manager.create_context_packet(
    target_agent="agent-002",
    context_type="task_knowledge",
    payload={"current_task": "customer_support", "priority": "high"}
)
```

## Features

- **Context Sharing**: Safe and interpretable context exchange between agents
- **State Transfer**: Seamless agent handoff protocols
- **Policy Alignment**: Cross-agent coordination and conflict resolution
- **Trust Scoring**: Confidence levels for agent interactions

## Installation

```bash
pip install pact-ax
```

## Documentation

- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api_reference/)
- [Examples](examples/)

## Contributing

See [CONTRIBUTING.md](docs/contributing.md)

## License

MIT License - see [LICENSE](LICENSE) file.
