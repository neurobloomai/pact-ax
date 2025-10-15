"""
Demo: State Transfer with Story Awareness

Shows how State Transfer Manager preserves narrative continuity
during agent handoffs.
"""

from pact_ax.primitives.story_keeper import StoryKeeper
from pact_ax.primitives.state_transfer import StateTransferManager


def main():
    print("=" * 70)
    print("State Transfer Demo: Conscious Handoffs Between Agents")
    print("=" * 70)
    print()
    
    # Create two agents with their own Story Keepers
    print("📝 Setting up Agent 001 (Initial Agent)...")
    story_keeper_001 = StoryKeeper("agent-001")
    transfer_mgr_001 = StateTransferManager("agent-001", story_keeper_001)
    
    print("📝 Setting up Agent 002 (Receiving Agent)...")
    story_keeper_002 = StoryKeeper("agent-002")
    transfer_mgr_002 = StateTransferManager("agent-002", story_keeper_002)
    
    print()
    print("-" * 70)
    print("Phase 1: Agent 001 builds story through interactions")
    print("-" * 70)
    print()
    
    # Agent 001 has some interactions first
    interactions = [
        ("Let's build a state transfer system", 
         "Great idea! We'll make it story-aware so handoffs preserve context."),
        
        ("How should we structure the transfer packet?",
         "It should include state, story context, and narrative explanation."),
        
        ("Let's add checkpoint functionality too",
         "Perfect - 360-degree awareness before critical operations."),
    ]
    
    for user_msg, agent_msg in interactions:
        print(f"User: {user_msg}")
        print(f"Agent 001: {agent_msg}")
        story_keeper_001.process_interaction(user_msg, agent_msg)
        print()
    
    # Show Agent 001's story state
    print("📖 Agent 001's Story Summary:")
    summary_001 = story_keeper_001.get_story_summary()
    print(f"  Current Arc: {summary_001['current_arc']}")
    print(f"  Total Interactions: {summary_001['total_interactions']}")
    print()
    
    print("-" * 70)
    print("Phase 2: Creating a checkpoint before handoff")
    print("-" * 70)
    print()
    
    # Create checkpoint
    current_state = {
        "current_task": "implementing_state_transfer",
        "progress": 0.75,
        "next_steps": ["test with real agents", "add persistence"]
    }
    
    checkpoint = transfer_mgr_001.create_checkpoint(
        "before_handoff",
        state_data=current_state,
        include_full_story=True
    )
    print()
    
    print("-" * 70)
    print("Phase 3: Agent 001 prepares handoff to Agent 002")
    print("-" * 70)
    print()
    
    # Prepare handoff
    transfer_packet = transfer_mgr_001.prepare_handoff(
        target_agent="agent-002",
        state_data=current_state,
        handoff_reason="continuation",
        context={"reason": "Agent 002 specializes in testing"}
    )
    
    print()
    print("📦 Transfer Packet Contents:")
    print(f"  From: {transfer_packet['transfer_meta']['from_agent']}")
    print(f"  To: {transfer_packet['transfer_meta']['to_agent']}")
    print(f"  Reason: {transfer_packet['transfer_meta']['handoff_reason']}")
    print(f"  Narrative: {transfer_packet['narrative']['what_we_were_doing']}")
    print(f"  Importance: {transfer_packet['narrative']['emotional_gravity']:.2f}")
    print()
    
    print("-" * 70)
    print("Phase 4: Agent 002 receives handoff with story integration")
    print("-" * 70)
    print()
    
    # Agent 002 receives the handoff
    confirmation = transfer_mgr_002.receive_handoff(
        transfer_packet,
        integrate_story=True
    )
    
    print()
    print("✓ Handoff Confirmation:")
    print(f"  Received: {confirmation['received']}")
    print(f"  Story Integrated: {confirmation['story_integrated']}")
    print(f"  Ready to Continue: {confirmation['ready_to_continue']}")
    print()
    
    # Show Agent 002's story now includes the handoff
    print("📖 Agent 002's Story Summary (after receiving handoff):")
    summary_002 = story_keeper_002.get_story_summary()
    print(f"  Current Arc: {summary_002['current_arc']}")
    print(f"  Total Interactions: {summary_002['total_interactions']}")
    print(f"  (Includes handoff as interaction)")
    print()
    
    print("-" * 70)
    print("Phase 5: Agent 002 continues with story awareness")
    print("-" * 70)
    print()
    
    # Agent 002 continues the work with full context
    story_keeper_002.process_interaction(
        "Continue with testing the state transfer",
        "Understood! I have full context from Agent 001's work. Starting tests..."
    )
    
    print("Agent 002 now has:")
    print("  ✓ The state data")
    print("  ✓ The story context from Agent 001")
    print("  ✓ Narrative continuity")
    print("  ✓ Relationship patterns")
    print()
    
    print("=" * 70)
    print("Demo Complete: Conscious Handoff Successful! ✨")
    print("=" * 70)
    print()
    print("Key Features Demonstrated:")
    print("  • Story-aware state transfer")
    print("  • 360-degree checkpoints")
    print("  • Narrative continuity across agents")
    print("  • Emotional gravity assessment")
    print("  • Context preservation")


if __name__ == "__main__":
    main()
