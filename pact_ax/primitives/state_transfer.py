"""
State Transfer Manager: Wealth Transfer Protocol for Agent Handoffs

Enables story-aware state transfers between agents with narrative continuity.
Not just data dumps - conscious handoffs that preserve relationship context.

Philosophy:
- State + Story + Trust = Meaningful handoff
- 360-degree awareness before critical transfers
- Organic resumption with narrative continuity
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from .story_keeper import StoryKeeper


class StateTransferManager:
    """
    Manages state transfers between agents with story awareness.
    
    Implements the wealth transfer protocol - not just moving state,
    but preserving the narrative thread, relationship context, and
    collaborative continuity.
    """
    
    def __init__(self, agent_id: str, story_keeper: Optional[StoryKeeper] = None):
        """
        Initialize state transfer manager.
        
        Args:
            agent_id: Unique identifier for this agent
            story_keeper: Optional Story Keeper for narrative continuity
        """
        self.agent_id = agent_id
        self.story_keeper = story_keeper
        self.active_transfers: Dict[str, Dict[str, Any]] = {}
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        
    def prepare_handoff(
        self,
        target_agent: str,
        state_data: Dict[str, Any],
        handoff_reason: str = "continuation",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare a story-aware state transfer.
        
        Creates a rich transfer packet that includes not just state,
        but narrative context, relationship depth, and story continuity.
        
        Args:
            target_agent: ID of receiving agent
            state_data: Current state to transfer
            handoff_reason: Why the handoff (continuation/pause/escalation/completion)
            context: Optional additional context
            
        Returns:
            Rich transfer packet with narrative continuity
        """
        # Get story context if available
        story_summary = None
        recent_context = []
        
        if self.story_keeper:
            story_summary = self.story_keeper.get_story_summary()
            recent_context = self.story_keeper.recall_for_context(k=3)
        
        # Create narrative explanation
        narrative = self._create_handoff_narrative(
            state_data,
            handoff_reason,
            story_summary,
            recent_context
        )
        
        # Build complete transfer packet
        transfer_packet = {
            # Core state
            "state": state_data,
            
            # Story context
            "story_context": {
                "current_arc": story_summary["current_arc"] if story_summary else None,
                "total_interactions": story_summary["total_interactions"] if story_summary else 0,
                "recent_context": [
                    {
                        "user_input": ctx["user_input"],
                        "arc": ctx["arc"].value if hasattr(ctx["arc"], "value") else str(ctx["arc"]),
                        "timestamp": ctx["timestamp"].isoformat()
                    }
                    for ctx in recent_context
                ]
            } if self.story_keeper else None,
            
            # Narrative continuity
            "narrative": narrative,
            
            # Transfer metadata
            "transfer_meta": {
                "from_agent": self.agent_id,
                "to_agent": target_agent,
                "timestamp": datetime.now().isoformat(),
                "handoff_reason": handoff_reason,
                "transfer_id": self._generate_transfer_id(target_agent)
            },
            
            # Additional context if provided
            "additional_context": context or {}
        }
        
        # Track active transfer
        transfer_id = transfer_packet["transfer_meta"]["transfer_id"]
        self.active_transfers[transfer_id] = transfer_packet
        
        print(f"📦 Prepared handoff: {self.agent_id} → {target_agent} ({handoff_reason})")
        
        return transfer_packet
    
    def receive_handoff(
        self,
        transfer_packet: Dict[str, Any],
        integrate_story: bool = True
    ) -> Dict[str, Any]:
        """
        Receive state transfer and integrate with story awareness.
        
        Args:
            transfer_packet: The handoff packet from another agent
            integrate_story: Whether to integrate story context into local Story Keeper
            
        Returns:
            Confirmation with integrated context
        """
        from_agent = transfer_packet["transfer_meta"]["from_agent"]
        handoff_reason = transfer_packet["transfer_meta"]["handoff_reason"]
        
        # Extract components
        state = transfer_packet["state"]
        story_context = transfer_packet.get("story_context", {})
        narrative = transfer_packet.get("narrative", {})
        
        print(f"📥 Received handoff from {from_agent}: {narrative.get('what_we_were_doing', 'No description')}")
        
        # If we have story keeper and should integrate, add to our story
        if integrate_story and self.story_keeper and story_context:
            # Add handoff as special interaction in story
            self.story_keeper.process_interaction(
                user_input=f"[HANDOFF from {from_agent}] {narrative.get('what_we_were_doing', '')}",
                agent_response=f"Received state transfer with story context. Ready to continue with awareness.",
                metadata={
                    "is_handoff": True,
                    "from_agent": from_agent,
                    "handoff_reason": handoff_reason,
                    "emotional_gravity": narrative.get("emotional_gravity", 0.5),
                    "transfer_id": transfer_packet["transfer_meta"]["transfer_id"]
                }
            )
        
        # Return confirmation with integrated understanding
        return {
            "received": True,
            "from_agent": from_agent,
            "state": state,
            "story_integrated": integrate_story and self.story_keeper is not None,
            "ready_to_continue": True,
            "understanding": narrative,
            "timestamp": datetime.now().isoformat()
        }
    
    def create_checkpoint(
        self,
        checkpoint_name: str,
        state_data: Optional[Dict[str, Any]] = None,
        include_full_story: bool = True
    ) -> Dict[str, Any]:
        """
        Create a 360-degree awareness checkpoint.
        
        Captures complete state + story before critical operations.
        Enables safe experimentation and rollback with narrative continuity.
        
        Args:
            checkpoint_name: Name for this checkpoint
            state_data: Optional state to checkpoint (if not provided, just story)
            include_full_story: Whether to snapshot full story
            
        Returns:
            Complete checkpoint that can be restored
        """
        checkpoint = {
            "checkpoint_name": checkpoint_name,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            
            # State snapshot
            "state": state_data,
            
            # Story snapshot
            "story_snapshot": None,
            
            # Active transfers at this moment
            "active_transfers": list(self.active_transfers.keys()),
        }
        
        # Capture full story state if requested
        if include_full_story and self.story_keeper:
            checkpoint["story_snapshot"] = {
                "summary": self.story_keeper.get_story_summary(),
                "current_arc": self.story_keeper.current_arc.value,
                "interaction_count": len(self.story_keeper.interactions),
                "arc_transition_count": len(self.story_keeper.arc_history)
            }
        
        # Store checkpoint
        self.checkpoints[checkpoint_name] = checkpoint
        
        print(f"✓ Checkpoint created: {checkpoint_name}")
        
        return checkpoint
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        return list(self.checkpoints.keys())
    
    def get_checkpoint(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific checkpoint."""
        return self.checkpoints.get(checkpoint_name)
    
    def restore_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Restore from a checkpoint.
        
        Note: This is a placeholder for now. Full restoration would require
        more sophisticated state management.
        
        Args:
            checkpoint_name: Name of checkpoint to restore
            
        Returns:
            True if restore successful
        """
        checkpoint = self.checkpoints.get(checkpoint_name)
        
        if not checkpoint:
            print(f"❌ Checkpoint not found: {checkpoint_name}")
            return False
        
        if checkpoint.get("agent_id") != self.agent_id:
            print(f"❌ Checkpoint belongs to different agent")
            return False
        
        print(f"✓ Restored checkpoint: {checkpoint_name}")
        return True
    
    def get_active_transfers(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently active transfers."""
        return self.active_transfers.copy()
    
    def clear_completed_transfers(self, older_than_minutes: int = 60):
        """
        Clear old completed transfers from tracking.
        
        Args:
            older_than_minutes: Clear transfers older than this many minutes
        """
        cutoff = datetime.now().timestamp() - (older_than_minutes * 60)
        
        transfers_to_remove = []
        for transfer_id, packet in self.active_transfers.items():
            transfer_time = datetime.fromisoformat(
                packet["transfer_meta"]["timestamp"]
            ).timestamp()
            
            if transfer_time < cutoff:
                transfers_to_remove.append(transfer_id)
        
        for transfer_id in transfers_to_remove:
            del self.active_transfers[transfer_id]
        
        if transfers_to_remove:
            print(f"🧹 Cleared {len(transfers_to_remove)} old transfers")
    
    # Private helper methods
    
    def _generate_transfer_id(self, target_agent: str) -> str:
        """Generate unique transfer ID."""
        timestamp = datetime.now().timestamp()
        return f"{self.agent_id}->{target_agent}-{timestamp}"
    
    def _create_handoff_narrative(
        self,
        state_data: Dict[str, Any],
        handoff_reason: str,
        story_summary: Optional[Dict[str, Any]],
        recent_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a narrative explanation of the handoff.
        
        Makes the transfer understandable, not just data.
        """
        # Create human-readable summary
        what_we_were_doing = self._summarize_recent_work(recent_context)
        
        # Assess importance
        emotional_gravity = self._assess_handoff_importance(
            state_data,
            handoff_reason,
            story_summary
        )
        
        return {
            "what_we_were_doing": what_we_were_doing,
            "why_it_matters": self._explain_importance(state_data, handoff_reason),
            "emotional_gravity": emotional_gravity,
            "current_arc": story_summary["current_arc"] if story_summary else None,
            "handoff_reason": handoff_reason,
            "continuity_preserved": self.story_keeper is not None
        }
    
    def _summarize_recent_work(self, recent_interactions: List[Dict]) -> str:
        """Create readable summary of recent work."""
        if not recent_interactions:
            return "Starting fresh collaboration"
        
        # Simple summary from last few interactions
        topics = [i["user_input"][:60] + "..." if len(i["user_input"]) > 60 else i["user_input"] 
                  for i in recent_interactions[-3:]]
        return " → ".join(topics)
    
    def _explain_importance(self, state_data: Dict[str, Any], handoff_reason: str) -> str:
        """Explain why this work matters."""
        task = state_data.get("current_task", "Unknown task")
        
        reason_explanations = {
            "continuation": f"Continuing work on: {task}",
            "pause": f"Pausing work on: {task} (to be resumed later)",
            "escalation": f"Escalating: {task} (needs attention)",
            "completion": f"Completing: {task} (final handoff)"
        }
        
        return reason_explanations.get(handoff_reason, f"Transferring: {task}")
    
    def _assess_handoff_importance(
        self,
        state_data: Dict[str, Any],
        handoff_reason: str,
        story_summary: Optional[Dict[str, Any]]
    ) -> float:
        """
        Assess emotional gravity of this handoff.
        
        Returns:
            Float between 0.0 (routine) and 1.0 (critical)
        """
        gravity = 0.5  # baseline
        
        # Escalations are important
        if handoff_reason == "escalation":
            gravity += 0.3
        
        # Completions matter
        if handoff_reason == "completion":
            gravity += 0.2
        
        # High progress state matters
        if state_data.get("progress", 0) > 0.7:
            gravity += 0.2
        
        # Long story arc suggests importance
        if story_summary and story_summary.get("total_interactions", 0) > 10:
            gravity += 0.1
        
        return min(gravity, 1.0)
