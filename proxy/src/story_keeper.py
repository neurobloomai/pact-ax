"""
Story Keeper - Relational State Persistence for MCP Sessions

Unlike audit logs that record "what happened", Story Keeper maintains
the evolving relational context: how trust developed, what patterns
emerged, and where the interaction trajectory is heading.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import hashlib
import json


class TrustTrajectory(Enum):
    """Direction of trust evolution in session"""
    BUILDING = "building"       # Trust increasing through consistent behavior
    STABLE = "stable"           # Established pattern, no significant change
    ERODING = "eroding"         # Inconsistencies appearing
    VIOLATED = "violated"       # Clear breach of relational context


@dataclass
class RelationalEvent:
    """Single interaction within the relational context"""
    timestamp: datetime
    method: str                 # MCP method called
    resource_pattern: str       # Abstracted resource (e.g., "repo:read" not specific file)
    context_hash: str           # Hash of relevant context at this point
    trust_delta: float          # How this event affected trust (-1 to +1)
    coherence_score: float      # How well this fits established pattern (0 to 1)
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "method": self.method,
            "resource_pattern": self.resource_pattern,
            "context_hash": self.context_hash,
            "trust_delta": self.trust_delta,
            "coherence_score": self.coherence_score
        }


@dataclass 
class SessionStory:
    """The complete relational narrative of an MCP session"""
    session_id: str
    started_at: datetime
    client_identity: str        # Who initiated (e.g., "cursor:user123")
    server_target: str          # What they're connecting to (e.g., "github:orgname")
    
    # Relational state
    events: list[RelationalEvent] = field(default_factory=list)
    trust_level: float = 0.5    # Current trust (0 = none, 1 = full)
    trajectory: TrustTrajectory = TrustTrajectory.BUILDING
    
    # Pattern recognition
    established_patterns: set[str] = field(default_factory=set)
    anomaly_count: int = 0
    last_coherence_score: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "client_identity": self.client_identity,
            "server_target": self.server_target,
            "trust_level": self.trust_level,
            "trajectory": self.trajectory.value,
            "event_count": len(self.events),
            "established_patterns": list(self.established_patterns),
            "anomaly_count": self.anomaly_count,
            "last_coherence_score": self.last_coherence_score
        }


class StoryKeeper:
    """
    Maintains relational state across MCP sessions.
    
    This is the core differentiator from policy-based security:
    - Policy engines ask: "Is this action permitted?"
    - StoryKeeper asks: "Does this action cohere with relational context?"
    """
    
    def __init__(self):
        self.sessions: dict[str, SessionStory] = {}
        self.pattern_window = 10  # Events to consider for pattern establishment
        self.drift_threshold = 0.3  # Coherence below this triggers drift alert
    
    def create_session(
        self, 
        session_id: str, 
        client_identity: str, 
        server_target: str
    ) -> SessionStory:
        """Initialize a new relational context"""
        story = SessionStory(
            session_id=session_id,
            started_at=datetime.utcnow(),
            client_identity=client_identity,
            server_target=server_target
        )
        self.sessions[session_id] = story
        return story
    
    def record_event(
        self,
        session_id: str,
        method: str,
        params: dict,
    ) -> tuple[RelationalEvent, Optional[str]]:
        """
        Record an MCP interaction and evaluate relational coherence.
        
        Returns:
            - The recorded event
            - Optional alert message if drift detected
        """
        story = self.sessions.get(session_id)
        if not story:
            raise ValueError(f"Unknown session: {session_id}")
        
        # Abstract the resource pattern (don't track specific files, track behavior)
        resource_pattern = self._extract_pattern(method, params)
        
        # Calculate coherence with established patterns
        coherence_score = self._calculate_coherence(story, resource_pattern)
        
        # Determine trust delta based on coherence
        trust_delta = self._calculate_trust_delta(coherence_score, story)
        
        # Create and record the event
        event = RelationalEvent(
            timestamp=datetime.utcnow(),
            method=method,
            resource_pattern=resource_pattern,
            context_hash=self._hash_context(story, params),
            trust_delta=trust_delta,
            coherence_score=coherence_score
        )
        story.events.append(event)
        
        # Update session state
        story.trust_level = max(0, min(1, story.trust_level + trust_delta))
        story.last_coherence_score = coherence_score
        self._update_patterns(story, resource_pattern)
        self._update_trajectory(story)
        
        # Check for drift
        alert = None
        if coherence_score < self.drift_threshold:
            story.anomaly_count += 1
            alert = self._generate_drift_alert(story, event)
        
        return event, alert
    
    def _extract_pattern(self, method: str, params: dict) -> str:
        """
        Abstract specific requests into behavioral patterns.
        
        e.g., "tools/call with repo.getContents for /src/main.py" 
              becomes "repo:read:source"
        """
        # GitHub MCP specific patterns
        if method == "tools/call":
            tool_name = params.get("name", "unknown")
            
            if "contents" in tool_name.lower() or "file" in tool_name.lower():
                path = str(params.get("arguments", {}).get("path", ""))
                if "/src/" in path or "/lib/" in path:
                    return "repo:read:source"
                elif "config" in path or ".yaml" in path or ".json" in path:
                    return "repo:read:config"
                else:
                    return "repo:read:other"
            
            elif "commit" in tool_name.lower():
                return "repo:write:commit"
            
            elif "issue" in tool_name.lower():
                return "repo:interact:issues"
            
            elif "org" in tool_name.lower() or "member" in tool_name.lower():
                return "org:access:elevated"  # Flag for drift detection
            
            elif "search" in tool_name.lower():
                return "repo:search"
            
            else:
                return f"tool:{tool_name}"
        
        elif method == "resources/read":
            return "resource:read"
        
        elif method == "resources/list":
            return "resource:list"
        
        else:
            return f"method:{method}"
    
    def _calculate_coherence(self, story: SessionStory, pattern: str) -> float:
        """
        How well does this pattern fit the established relational context?
        
        Returns 0-1 where:
        - 1.0 = perfectly consistent with established patterns
        - 0.5 = neutral (new but not contradictory)
        - 0.0 = contradicts established patterns
        """
        if not story.established_patterns:
            # Early in session, everything is exploratory
            return 0.7
        
        # Check if pattern exists in established set
        if pattern in story.established_patterns:
            return 1.0
        
        # Check for pattern escalation (e.g., read -> write -> org access)
        pattern_type = pattern.split(":")[0] if ":" in pattern else pattern
        
        # Org-level access when only repo-level established = potential drift
        if "org:" in pattern:
            if not any("org:" in p for p in story.established_patterns):
                return 0.2  # Significant drift indicator
        
        # Write access when only read established
        if ":write:" in pattern:
            if not any(":write:" in p for p in story.established_patterns):
                return 0.4  # Moderate drift
        
        # New but related pattern
        if any(pattern_type in p for p in story.established_patterns):
            return 0.6
        
        # Completely new pattern type
        return 0.5
    
    def _calculate_trust_delta(self, coherence: float, story: SessionStory) -> float:
        """
        How does this interaction affect the trust relationship?
        """
        if coherence >= 0.8:
            # Consistent behavior builds trust (diminishing returns)
            return 0.02 * (1 - story.trust_level)
        elif coherence >= 0.5:
            # Neutral - slight trust building
            return 0.01
        elif coherence >= 0.3:
            # Concerning - trust erosion
            return -0.05
        else:
            # Significant deviation - trust impact
            return -0.15
    
    def _hash_context(self, story: SessionStory, params: dict) -> str:
        """Create a hash representing the current relational context"""
        context = {
            "trust_level": round(story.trust_level, 2),
            "pattern_count": len(story.established_patterns),
            "trajectory": story.trajectory.value,
            "recent_patterns": list(story.established_patterns)[:5]
        }
        return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:12]
    
    def _update_patterns(self, story: SessionStory, pattern: str) -> None:
        """Update established patterns based on recent behavior"""
        # Count pattern occurrences in recent events
        recent = story.events[-self.pattern_window:]
        pattern_counts = {}
        for event in recent:
            p = event.resource_pattern
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
        
        # Patterns seen 3+ times in window become established
        for p, count in pattern_counts.items():
            if count >= 3:
                story.established_patterns.add(p)
    
    def _update_trajectory(self, story: SessionStory) -> None:
        """Update trust trajectory based on recent events"""
        if len(story.events) < 3:
            return
        
        recent_deltas = [e.trust_delta for e in story.events[-5:]]
        avg_delta = sum(recent_deltas) / len(recent_deltas)
        
        if avg_delta > 0.01:
            story.trajectory = TrustTrajectory.BUILDING
        elif avg_delta > -0.02:
            story.trajectory = TrustTrajectory.STABLE
        elif avg_delta > -0.1:
            story.trajectory = TrustTrajectory.ERODING
        else:
            story.trajectory = TrustTrajectory.VIOLATED
    
    def _generate_drift_alert(self, story: SessionStory, event: RelationalEvent) -> str:
        """Generate human-readable drift alert"""
        return (
            f"[PACT-AX DRIFT ALERT]\n"
            f"Session: {story.session_id}\n"
            f"Pattern: {event.resource_pattern}\n"
            f"Coherence: {event.coherence_score:.2f} (threshold: {self.drift_threshold})\n"
            f"Trust Level: {story.trust_level:.2f} → {story.trajectory.value}\n"
            f"Established Patterns: {', '.join(story.established_patterns)}\n"
            f"Anomaly Count: {story.anomaly_count}\n"
            f"---\n"
            f"This action deviates from the established relational context.\n"
            f"Policy may permit it, but relational coherence suggests review."
        )
    
    def get_session_summary(self, session_id: str) -> dict:
        """Get current relational state for external systems (e.g., policy engines)"""
        story = self.sessions.get(session_id)
        if not story:
            return {"error": "session not found"}
        
        return {
            **story.to_dict(),
            "recent_events": [e.to_dict() for e in story.events[-5:]],
            "drift_risk": "high" if story.anomaly_count > 2 else "low" if story.anomaly_count == 0 else "medium"
        }
