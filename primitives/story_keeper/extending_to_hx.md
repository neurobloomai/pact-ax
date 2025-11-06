# Extending Story Keeper to HX Layer

**Document Location:** `pact-ax/primitives/story_keeper/extending_to_hx.md`

**Audience:** AX layer developers building Story Keeper substrate

**Purpose:** Guide for making Story Keeper extensible to Human Experience (HX) layer

---

## Overview

Story Keeper is an AX primitive that maintains narrative continuity across agent interactions. This document explains how to build Story Keeper so that HX layer primitives can extend it for human-facing personalization.

---

## Core Principle

**Story Keeper = Substrate for Both Layers**

- **AX Layer:** Uses Story Keeper for agent coordination
- **HX Layer:** Uses same Story Keeper for user relationship continuity
- **Shared Foundation:** One narrative substrate, multiple interfaces

---

## Design Requirements for HX Extensibility

### 1. Layer-Aware Event Storage

Story Keeper must distinguish between AX events (agent coordination) and HX events (user interaction).

**Implementation Pattern:**

```python
class StoryKeeper:
    def add_event(self, event, layer="ax", event_type="default", metadata=None):
        """
        Universal event storage with layer awareness
        
        Args:
            event: The event description
            layer: "ax" (agent) or "hx" (human) layer
            event_type: Type of event (handoff, preference, state, etc.)
            metadata: Additional context
        """
        story_event = {
            'timestamp': now(),
            'event': event,
            'layer': layer,  # ← Critical for filtering
            'type': event_type,
            'metadata': metadata or {}
        }
        
        self.narrative.append(story_event)
        self._notify_listeners(layer, event_type)
```

**Why This Matters:**

- HX layer can filter for user-relevant events only
- AX layer can filter for coordination events only
- Both layers can access full narrative when needed

---

### 2. Flexible Querying Interface

Provide query methods that work for both layers' needs.

**Implementation Pattern:**

```python
class StoryKeeper:
    def get_narrative(self, filter_layer=None, filter_type=None, 
                     time_range=None, limit=None):
        """
        Flexible narrative retrieval
        
        Args:
            filter_layer: "ax", "hx", or None (all)
            filter_type: Specific event type or None (all)
            time_range: (start, end) or None (all time)
            limit: Max events to return or None (unlimited)
        
        Returns:
            List of events matching filters
        """
        events = self.narrative
        
        if filter_layer:
            events = [e for e in events if e['layer'] == filter_layer]
        
        if filter_type:
            events = [e for e in events if e['type'] == filter_type]
        
        if time_range:
            start, end = time_range
            events = [e for e in events 
                     if start <= e['timestamp'] <= end]
        
        if limit:
            events = events[-limit:]  # Most recent N events
        
        return events
    
    def get_recent_context(self, window_size=10):
        """Recent events for immediate context"""
        return self.narrative[-window_size:]
    
    def get_full_narrative(self):
        """Complete narrative (use carefully)"""
        return self.narrative.copy()
```

**Why This Matters:**

- AX layer: Query for agent coordination events
- HX layer: Query for user preference events
- Both: Query for relevant time windows
- Flexible access patterns

---

### 3. Event Type Taxonomy

Define clear event types that make sense for both layers.

**Recommended Event Types:**

```python
# AX Layer Event Types
AX_EVENT_TYPES = {
    'handoff': 'Agent-to-agent context transfer',
    'coordination': 'Multi-agent coordination point',
    'state_change': 'Agent state transition',
    'completion': 'Agent task completion'
}

# HX Layer Event Types
HX_EVENT_TYPES = {
    'preference': 'User preference learned',
    'emotional_state': 'User emotional state detected',
    'interaction': 'User interaction event',
    'adaptation': 'System adaptation to user'
}

# Shared Event Types
SHARED_EVENT_TYPES = {
    'context_update': 'Context information updated',
    'milestone': 'Significant progress point'
}
```

**Usage in Story Keeper:**

```python
class StoryKeeper:
    EVENT_TYPES = {**AX_EVENT_TYPES, **HX_EVENT_TYPES, **SHARED_EVENT_TYPES}
    
    def add_event(self, event, layer, event_type, metadata=None):
        # Validate event type
        if event_type not in self.EVENT_TYPES:
            raise ValueError(f"Unknown event type: {event_type}")
        
        # Store with validated type
        # ... (storage logic)
```

---

### 4. Metadata Extensibility

Allow arbitrary metadata to support diverse use cases.

**Implementation Pattern:**

```python
# AX Layer Usage
story.add_event(
    event="Agent handoff: GPT-4 → Claude",
    layer="ax",
    event_type="handoff",
    metadata={
        'from_agent': 'gpt4',
        'to_agent': 'claude',
        'context_size': 1024,
        'task_id': 'analysis_789'
    }
)

# HX Layer Usage  
story.add_event(
    event="User prefers morning reflections",
    layer="hx",
    event_type="preference",
    metadata={
        'preference_category': 'timing',
        'confidence': 0.9,
        'learned_from': 'repeated_pattern'
    }
)
```

**Why This Matters:**

- Each layer can store layer-specific context
- Extensible without modifying core Story Keeper
- Metadata supports diverse use cases

---

### 5. Read-Only Views for HX Layer

Provide safe interfaces for HX layer to read narrative without modifying AX coordination data.

**Implementation Pattern:**

```python
class StoryKeeper:
    def get_hx_view(self):
        """
        Get HX-specific read-only view
        Returns: HXNarrativeView
        """
        return HXNarrativeView(self)

class HXNarrativeView:
    """Read-only view of narrative for HX layer"""
    
    def __init__(self, story_keeper):
        self._story = story_keeper
    
    def get_user_preferences(self):
        """Get all user preference events"""
        return self._story.get_narrative(
            filter_layer="hx",
            filter_type="preference"
        )
    
    def get_recent_interactions(self, limit=10):
        """Get recent user interactions"""
        return self._story.get_narrative(
            filter_layer="hx",
            filter_type="interaction",
            limit=limit
        )
    
    def get_emotional_history(self):
        """Get user emotional state history"""
        return self._story.get_narrative(
            filter_layer="hx",
            filter_type="emotional_state"
        )
    
    # No write methods - read only!
```

**Why This Matters:**

- HX layer can't accidentally corrupt AX coordination data
- Clear separation of concerns
- Type-safe interface for HX developers

---

## Extension Points

### 1. Event Listeners

Allow HX primitives to subscribe to Story Keeper events.

**Implementation Pattern:**

```python
class StoryKeeper:
    def __init__(self):
        self.narrative = []
        self.listeners = defaultdict(list)  # layer → [callbacks]
    
    def register_listener(self, layer, callback):
        """
        Register callback for layer events
        
        Args:
            layer: "ax", "hx", or "all"
            callback: Function to call on events
        """
        self.listeners[layer].append(callback)
    
    def _notify_listeners(self, layer, event_type):
        """Notify registered listeners of new event"""
        # Notify layer-specific listeners
        for callback in self.listeners[layer]:
            callback(layer, event_type)
        
        # Notify "all" listeners
        for callback in self.listeners["all"]:
            callback(layer, event_type)
```

**HX Layer Usage:**

```python
# HX primitive subscribing to Story Keeper
def on_user_event(layer, event_type):
    if event_type == "preference":
        # Update HX primitive's internal state
        update_user_model()

story_keeper.register_listener("hx", on_user_event)
```

---

### 2. Narrative Summarization

Provide hooks for HX layer to generate user-facing summaries.

**Implementation Pattern:**

```python
class StoryKeeper:
    def summarize_for_user(self, time_window=None):
        """
        Generate user-facing narrative summary
        
        This is a hook that HX layer can override/extend
        """
        events = self.get_narrative(
            filter_layer="hx",
            time_range=time_window
        )
        
        # Basic summarization
        summary = {
            'interaction_count': len([e for e in events 
                                     if e['type'] == 'interaction']),
            'preferences_learned': len([e for e in events 
                                       if e['type'] == 'preference']),
            'adaptations_made': len([e for e in events 
                                    if e['type'] == 'adaptation'])
        }
        
        return summary
    
    def get_relationship_timeline(self):
        """
        Generate timeline of user-system relationship
        
        For HX layer to display to users
        """
        hx_events = self.get_narrative(filter_layer="hx")
        
        timeline = []
        for event in hx_events:
            timeline.append({
                'date': event['timestamp'],
                'what_happened': event['event'],
                'type': event['type']
            })
        
        return timeline
```

---

## Implementation Checklist

When building Story Keeper with HX extensibility:

- [ ] **Layer tagging:** Events tagged with "ax" or "hx" layer
- [ ] **Flexible querying:** Filter by layer, type, time range
- [ ] **Event types:** Clear taxonomy for both layers
- [ ] **Metadata support:** Arbitrary metadata allowed
- [ ] **HX views:** Read-only interfaces for HX layer
- [ ] **Event listeners:** Subscription mechanism for HX primitives
- [ ] **Summarization hooks:** User-facing summary generation
- [ ] **Documentation:** Clear examples for HX developers

---

## Example: Complete Integration

```python
# story_keeper.py (AX primitive with HX extensibility)

class StoryKeeper:
    def __init__(self):
        self.narrative = []
        self.listeners = defaultdict(list)
    
    # Core AX functionality
    def add_event(self, event, layer="ax", event_type="default", 
                  metadata=None):
        story_event = {
            'timestamp': now(),
            'event': event,
            'layer': layer,
            'type': event_type,
            'metadata': metadata or {}
        }
        self.narrative.append(story_event)
        self._notify_listeners(layer, event_type)
    
    # Flexible querying for both layers
    def get_narrative(self, filter_layer=None, filter_type=None):
        events = self.narrative
        if filter_layer:
            events = [e for e in events if e['layer'] == filter_layer]
        if filter_type:
            events = [e for e in events if e['type'] == filter_type]
        return events
    
    # HX layer extensions
    def get_hx_view(self):
        return HXNarrativeView(self)
    
    def register_listener(self, layer, callback):
        self.listeners[layer].append(callback)
    
    def _notify_listeners(self, layer, event_type):
        for callback in self.listeners[layer]:
            callback(layer, event_type)
        for callback in self.listeners["all"]:
            callback(layer, event_type)

# Usage by AX layer
story = StoryKeeper()
story.add_event(
    "Agent handoff: GPT-4 → Claude",
    layer="ax",
    event_type="handoff"
)

# Usage by HX layer (see pact-hx docs for details)
hx_view = story.get_hx_view()
preferences = hx_view.get_user_preferences()
```

---

## Testing HX Extensibility

```python
def test_hx_layer_can_read():
    """HX layer can read from Story Keeper"""
    story = StoryKeeper()
    
    # AX adds coordination event
    story.add_event("Agent coordination", layer="ax")
    
    # HX adds preference event
    story.add_event("User preference", layer="hx", 
                   event_type="preference")
    
    # HX can filter for its events only
    hx_view = story.get_hx_view()
    prefs = hx_view.get_user_preferences()
    
    assert len(prefs) == 1
    assert prefs[0]['event'] == "User preference"

def test_layer_isolation():
    """Layers don't interfere with each other"""
    story = StoryKeeper()
    
    story.add_event("AX event 1", layer="ax")
    story.add_event("HX event 1", layer="hx")
    story.add_event("AX event 2", layer="ax")
    
    ax_events = story.get_narrative(filter_layer="ax")
    hx_events = story.get_narrative(filter_layer="hx")
    
    assert len(ax_events) == 2
    assert len(hx_events) == 1
```

---

## Related Documentation

- **Architecture Overview:** `pact/docs/architecture/hx_ax_integration.md`
- **HX Layer Usage:** `pact-hx/primitives/memory/using_ax_substrate.md`
- **Story Keeper Implementation:** `pact-ax/primitives/story_keeper/README.md`

---

## Key Takeaways

1. **Design for extension from the start**
2. **Layer tagging is critical for filtering**
3. **Provide flexible query interfaces**
4. **Event types should work for both layers**
5. **Read-only views protect layer boundaries**
6. **Event listeners enable reactive HX primitives**

---

**Document Version:** 1.0  
**Last Updated:** November 2025  
**Maintainer:** PACT-AX Team
