"""
Story Keeper: Conscious Continuity for AI Agents

Transforms interaction history into narrative memory.
Because consciousness organizes itself through stories, not states.
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any


class StoryArc(Enum):
    """The fundamental phases of relationship development"""
    EXPLORATION = "exploration"      # Getting to know each other
    COLLABORATION = "collaboration"  # Working together actively
    INTEGRATION = "integration"      # Reflecting, deepening, internalizing


_STOPWORDS = {
    "i", "me", "my", "the", "a", "an", "is", "it", "to", "for", "in",
    "of", "and", "or", "but", "how", "what", "why", "when", "where",
    "who", "do", "did", "does", "will", "can", "could", "should", "would",
    "help", "please", "this", "that", "with", "about", "on", "at", "from",
    "by", "be", "am", "are", "was", "were", "i'm", "i've", "i'll", "just",
    "very", "much", "some", "any", "all", "not", "no", "yes", "get", "got",
    "have", "has", "had", "its", "we", "they", "he", "she", "you", "your",
    "our", "their", "his", "her", "let", "let's", "want", "need", "what's",
    "there", "then", "than", "so", "if", "up", "out", "too", "also", "more",
    "now", "new", "use", "used", "us", "her", "him", "its", "been", "into",
    "through", "after", "before", "she's", "he's", "they're", "we're",
}


class StoryKeeper:
    """
    Maintains conscious continuity through narrative coherence.

    Not just logging interactions — understanding them as story.
    """

    def __init__(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.session_id = session_id
        self.config = config or {}
        self.interactions: List[Dict[str, Any]] = []
        self.current_arc = StoryArc.EXPLORATION
        self.arc_history: List[Dict[str, Any]] = []
        self.story_state: Dict[str, Any] = self._initial_story_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_turn(
        self,
        user_message: str,
        user_id: Optional[str] = None
    ) -> str:
        """
        Process a conversation turn and evolve the story state.

        Returns a narrative beat string summarising what happened.
        """
        interaction = {
            "timestamp": datetime.now(),
            "user_input": user_message,
            "arc": self.current_arc,
            "metadata": {"user_id": user_id} if user_id else {}
        }
        self.interactions.append(interaction)

        new_arc = self._detect_story_arc(interaction)
        if new_arc != self.current_arc:
            self._record_arc_transition(self.current_arc, new_arc)
            self.current_arc = new_arc

        self._update_story_state(user_message)
        return self.story_state["last_beat"]

    def process_interaction(
        self,
        user_input: str,
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an interaction and integrate it into the ongoing story.

        Returns an enriched interaction dict with story context.
        """
        interaction = {
            "timestamp": datetime.now(),
            "user_input": user_input,
            "agent_response": agent_response,
            "arc": self.current_arc,
            "metadata": metadata or {}
        }
        self.interactions.append(interaction)

        new_arc = self._detect_story_arc(interaction)
        if new_arc != self.current_arc:
            self._record_arc_transition(self.current_arc, new_arc)
            self.current_arc = new_arc

        self._update_story_state(user_input)
        return interaction

    def get_story_state(self) -> Dict[str, Any]:
        """Return a snapshot of the current story state."""
        return self.story_state.copy()

    def load_story_state(self, state: Dict[str, Any]) -> None:
        """Replace the current story state with a previously saved one."""
        self.story_state = state.copy()

    def reset_story(self) -> None:
        """Reset all story state back to initial conditions."""
        self.interactions = []
        self.current_arc = StoryArc.EXPLORATION
        self.arc_history = []
        self.story_state = self._initial_story_state()

    def recall_from_arc(
        self,
        arc_type: StoryArc,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """Return the most recent k interactions from a given story arc."""
        arc_interactions = [
            ix for ix in self.interactions if ix["arc"] == arc_type
        ]
        return arc_interactions[-k:] if arc_interactions else []

    def recall_for_context(
        self,
        query: Optional[str] = None,
        prefer_current_arc: bool = True,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Story-aware context retrieval."""
        if prefer_current_arc:
            relevant = self.recall_from_arc(self.current_arc, k=k)
            if len(relevant) < k:
                remaining = k - len(relevant)
                for arc in StoryArc:
                    if arc == self.current_arc or remaining <= 0:
                        continue
                    chunk = self.recall_from_arc(arc, k=remaining)
                    relevant.extend(chunk)
                    remaining -= len(chunk)
        else:
            relevant = self.interactions[-k:]
        return relevant

    def get_story_summary(self) -> Dict[str, Any]:
        """Return a summary of the current story state."""
        return {
            "agent_id": self.agent_id,
            "current_arc": self.current_arc.value,
            "total_interactions": len(self.interactions),
            "arc_transitions": len(self.arc_history),
            "arc_history": [
                {
                    "from": t["from"].value,
                    "to": t["to"].value,
                    "at": t["at"].isoformat()
                }
                for t in self.arc_history
            ]
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_story_state(self) -> Dict[str, Any]:
        return {
            "characters": {"user": "", "agent": self.agent_id},
            "arc": StoryArc.EXPLORATION.value,
            "themes": [],
            "last_beat": "",
            "context": ""
        }

    def _update_story_state(self, user_message: str) -> None:
        max_t = self._max_themes()

        # Accumulate themes, deduplicate, cap
        new_themes = self._extract_themes(user_message)
        merged = list(dict.fromkeys(self.story_state["themes"] + new_themes))
        themes = merged[-max_t:] if len(merged) > max_t else merged

        # Accumulate context as a sorted word cloud
        context = self._accumulate_context(user_message)

        # User character reflects their most recent message
        user_char = self._describe_user_character(user_message)

        # Arc string embeds recent themes so it evolves naturally
        arc_str = self._describe_arc(themes)

        # Narrative beat for this turn
        last_beat = self._generate_beat(user_message)

        self.story_state.update({
            "themes": themes,
            "context": context,
            "characters": {
                "user": user_char,
                "agent": self.story_state["characters"].get("agent", self.agent_id)
            },
            "arc": arc_str,
            "last_beat": last_beat
        })

    def _max_themes(self) -> int:
        max_length = self.config.get("max_story_length")
        if max_length:
            return max(5, max_length // 5)
        return 100

    def _extract_themes(self, user_message: str) -> List[str]:
        if not user_message:
            return []
        themes = []
        for word in user_message.lower().split():
            cleaned = word.strip("?!.,;:'\"()[]")
            if cleaned and cleaned not in _STOPWORDS and len(cleaned) > 2:
                themes.append(cleaned)
        return themes

    def _accumulate_context(self, user_message: str) -> str:
        existing = set(self.story_state["context"].split()) if self.story_state["context"] else set()
        existing |= set(self._extract_themes(user_message))
        return " ".join(sorted(existing))

    def _describe_user_character(self, user_message: str) -> str:
        if not user_message:
            return self.story_state["characters"].get("user", "")
        excerpt = user_message[:80].strip()
        if len(user_message) > 80:
            excerpt += "..."
        return excerpt

    def _describe_arc(self, themes: List[str]) -> str:
        arc_name = self.current_arc.value.capitalize()
        recent = themes[-2:] if themes else []
        if recent:
            return f"{arc_name}: {', '.join(recent)}"
        return arc_name

    def _generate_beat(self, user_message: str) -> str:
        if not user_message:
            return "Continued the conversation"
        arc_name = self.current_arc.value.capitalize()
        excerpt = user_message[:60].strip()
        if len(user_message) > 60:
            excerpt += "..."
        return f"{arc_name}: {excerpt}"

    def _detect_story_arc(self, interaction: Dict[str, Any]) -> StoryArc:
        """Heuristic arc detection from interaction content."""
        user_text = interaction["user_input"].lower()

        exploration_keywords = [
            "what is", "what's", "how does", "how do",
            "tell me", "explain", "why", "who", "where"
        ]
        exploration_score = sum(
            1 for kw in exploration_keywords if kw in user_text
        )

        collaboration_keywords = [
            "let's", "we could", "we should", "build",
            "create", "make", "together", "start", "design"
        ]
        collaboration_score = sum(
            1 for kw in collaboration_keywords if kw in user_text
        )

        integration_keywords = [
            "internalize", "reflect", "thinking about",
            "resonates", "feeling", "sense", "back with",
            "sitting with", "processing"
        ]
        integration_score = sum(
            1 for kw in integration_keywords if kw in user_text
        )

        scores = {
            StoryArc.EXPLORATION: exploration_score,
            StoryArc.COLLABORATION: collaboration_score,
            StoryArc.INTEGRATION: integration_score
        }

        max_score = max(scores.values())
        if max_score == 0:
            return self.current_arc

        return max(scores, key=scores.get)

    def _record_arc_transition(self, from_arc: StoryArc, to_arc: StoryArc) -> None:
        """Record a story arc shift."""
        self.arc_history.append({
            "from": from_arc,
            "to": to_arc,
            "at": datetime.now(),
            "interaction_count": len(self.interactions)
        })
