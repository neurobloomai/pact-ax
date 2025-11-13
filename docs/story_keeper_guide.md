# Story Keeper Guide

## Overview

Story Keeper is a PACT-AX primitive that enables AI agents to maintain **narrative continuity** across conversations. Instead of just storing chat history, Story Keeper maintains a living "story" of the interaction that evolves with each exchange.

## Core Concept

**Traditional Approach:**
- Store messages in a list
- Retrieve relevant chunks when needed
- Generate contextually appropriate responses

**Story Keeper Approach:**
- Maintain a narrative state (characters, arc, themes)
- Evolve the story with each interaction
- Ground responses in the living narrative

The difference: **simulation vs embodiment of coherence**

---

## Installation

```bash
pip install pact-ax
```

Or install from source:

```bash
git clone https://github.com/neurobloomai/pact-ax.git
cd pact-ax
pip install -e .
```

---

## Quick Start

### Basic Usage

```python
from pact_ax.primitives.story_keeper import StoryKeeper

# Initialize Story Keeper
keeper = StoryKeeper(agent_id="my-agent")

# Start a conversation
user_message = "Help me plan a startup"

# Update story and get response
response = keeper.process_turn(
    user_message=user_message,
    user_id="user-123"
)

print(response)
```

### What Happens Internally

1. **Story Initialization** (first turn):
   - Creates initial story state
   - Identifies characters (user, agent)
   - Establishes initial arc and themes

2. **Story Evolution** (subsequent turns):
   - Analyzes new user message
   - Updates the narrative state
   - Evolves understanding of the relationship

3. **Story-Grounded Response**:
   - Reads current story state
   - Generates response grounded in narrative
   - Maintains continuity across turns

---

## Story State Structure

The Story Keeper maintains a structured narrative state:

```python
{
    "characters": {
        "user": "Entrepreneur at ideation stage, values clarity and structure",
        "agent": "Strategic advisor focused on practical next steps"
    },
    "arc": "Beginning: Exploring startup idea → Current: Planning execution strategy",
    "themes": ["clarity", "practical action", "reducing overwhelm"],
    "last_beat": "User just shared concern about funding options",
    "context": "This is a multi-session journey of entrepreneurial planning"
}
```

---

## Advanced Usage

### Multi-Session Continuity

```python
from pact_ax.primitives.story_keeper import StoryKeeper

# Initialize with session ID for persistence
keeper = StoryKeeper(
    agent_id="my-agent",
    session_id="user-123-startup-planning"
)

# First session
response1 = keeper.process_turn("Help me plan a startup")

# ... Later session (story state is maintained)
response2 = keeper.process_turn("What about funding now?")
# Response will be grounded in the evolved narrative
```

### Custom Story Configuration

```python
keeper = StoryKeeper(
    agent_id="my-agent",
    config={
        "story_update_frequency": "every_turn",  # or "every_n_turns"
        "theme_extraction": True,
        "arc_tracking": True,
        "max_story_length": 500  # tokens
    }
)
```

### Accessing Story State

```python
# Get current story state
story = keeper.get_story_state()

print(f"Characters: {story['characters']}")
print(f"Current Arc: {story['arc']}")
print(f"Themes: {story['themes']}")
```

### Resetting Story

```python
# Reset story state (new conversation)
keeper.reset_story()
```

---

## Integration with Other PACT-AX Primitives

### Story Keeper + Context Sharing

```python
from pact_ax.primitives.story_keeper import StoryKeeper
from pact_ax.primitives.context_share import ContextShareManager

keeper = StoryKeeper(agent_id="agent-001")
context_manager = ContextShareManager("agent-001")

# Process turn and share story context
response = keeper.process_turn(user_message)
story_state = keeper.get_story_state()

# Share narrative context with another agent
context_packet = context_manager.create_context_packet(
    target_agent="agent-002",
    context_type="narrative_state",
    payload={"story": story_state}
)
```

### Story Keeper + Trust Scoring

```python
from pact_ax.primitives.story_keeper import StoryKeeper
from pact_ax.primitives.trust_score import TrustManager

keeper = StoryKeeper(agent_id="agent-001")
trust_manager = TrustManager("agent-001")

# Update trust based on story evolution
story = keeper.get_story_state()
if "positive_interaction" in story['themes']:
    trust_manager.update_trust("user-123", delta=0.1)
```

---

## Best Practices

### 1. **Initialize Early**
Create the Story Keeper at the start of a conversation session, not mid-way.

### 2. **Let Stories Evolve Naturally**
Don't force story updates. Let the narrative emerge from genuine interactions.

### 3. **Use Themes Wisely**
Themes should capture what matters to the user, not just conversation topics.

### 4. **Monitor Story Length**
Keep story state concise. Prune old narrative beats when they're no longer relevant.

### 5. **Combine with Other Primitives**
Story Keeper works best when integrated with Context Sharing and Trust Scoring.

---

## Example Use Cases

### 1. **Long-term Coaching/Advisory**
Maintain continuity across weeks or months of mentorship conversations.

```python
keeper = StoryKeeper(
    agent_id="coach-agent",
    session_id="user-coaching-journey"
)
```

### 2. **Customer Support**
Remember not just what the user said, but their journey with your product.

```python
keeper = StoryKeeper(
    agent_id="support-agent",
    config={"theme_extraction": True}  # Track recurring pain points
)
```

### 3. **Multi-Agent Collaboration**
Share narrative state between agents for seamless handoffs.

```python
# Agent 1 builds story
agent1_keeper = StoryKeeper(agent_id="agent-001")
story = agent1_keeper.get_story_state()

# Agent 2 continues the narrative
agent2_keeper = StoryKeeper(agent_id="agent-002")
agent2_keeper.load_story_state(story)
```

---
## Use Cases

### Personal AI Relationships
- Long-term coaching/mentorship
- Therapy continuity
- Learning companion

### Enterprise
- Customer support memory
- Sales relationship tracking
- Employee onboarding

### Creative Work
- **Character consistency for writers**
- World-building continuity
- Interactive fiction
- RPG game mastering

### AI Agent Coordination
- Multi-agent systems maintaining shared context
- AI company "culture" preservation
- Decentralized organizations with value alignment

---

## Troubleshooting

### Story State Not Persisting
**Problem:** Story resets between sessions.

**Solution:** Provide a `session_id` when initializing Story Keeper.

```python
keeper = StoryKeeper(agent_id="my-agent", session_id="user-123")
```

### Story Becoming Too Long
**Problem:** Story state grows too large over time.

**Solution:** Set a `max_story_length` in config or periodically reset old narrative beats.

```python
keeper = StoryKeeper(
    agent_id="my-agent",
    config={"max_story_length": 300}
)
```

### Responses Feel Generic
**Problem:** Responses don't feel grounded in the story.

**Solution:** Check that story state is being properly updated and accessed before each response.

```python
# Debug: Print story state before generating response
print(keeper.get_story_state())
response = keeper.process_turn(user_message)
```

---

## API Reference

### `StoryKeeper(agent_id, session_id=None, config=None)`

Initialize a Story Keeper instance.

**Parameters:**
- `agent_id` (str): Unique identifier for the agent
- `session_id` (str, optional): Session ID for persistence
- `config` (dict, optional): Configuration options

**Returns:** StoryKeeper instance

---

### `process_turn(user_message, user_id=None)`

Process a conversation turn and generate a story-grounded response.

**Parameters:**
- `user_message` (str): User's message
- `user_id` (str, optional): User identifier

**Returns:** Response string

---

### `get_story_state()`

Get the current story state.

**Returns:** Dictionary containing story state (characters, arc, themes, etc.)

---

### `load_story_state(story_state)`

Load a previously saved story state.

**Parameters:**
- `story_state` (dict): Story state dictionary

---

### `reset_story()`

Reset the story state to initial conditions.

---

## Philosophy

Story Keeper embodies the PACT-AX principle: **"Trust scales while control just moves bottlenecks."**

By maintaining narrative continuity, agents build trust through consistent, coherent interactions over time. The story state acts as a shared understanding that scales better than rigid context management.

### Core Principles

1. **Narrative over History**: Stories evolve, logs just append
2. **Embodiment over Simulation**: Carry understanding forward, don't just retrieve
3. **Relationship over Transaction**: Build ongoing connections, not just solve individual queries
4. **Organic Growth**: Let stories emerge naturally from genuine interaction

---

## Contributing

We welcome contributions! See the main [PACT-AX repository](https://github.com/neurobloomai/pact-ax) for contribution guidelines.

**Areas for Contribution:**
- Story compression algorithms
- Multi-agent narrative synchronization
- Theme extraction improvements
- Story visualization tools

---

## Related Resources

- [Blog Post: "Why AI Agents Need Conscious Continuity"](../blog/conscious-continuity.md)
- [PACT-AX Main Repository](https://github.com/neurobloomai/pact-ax)
- [PACT-AX Documentation](https://github.com/neurobloomai/pact-ax#readme)

---

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/neurobloomai/pact-ax/issues)
- Discussions: [Ask questions and share ideas](https://github.com/neurobloomai/pact-ax/discussions)
- Email: founders@neurobloom.ai

---

**Built with 🎵 by the neurobloom.ai community.**

*Where Artificial Intelligence meets Emotional Intelligence, and conversations become stories.*
