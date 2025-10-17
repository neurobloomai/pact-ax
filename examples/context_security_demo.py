"""
Demo: Context Security with Story Awareness

Shows how security events become part of the agent's narrative.
"""

from pact_ax.primitives.story_keeper import StoryKeeper, StoryArc
from pact_ax.primitives.context_security.manager import (
    ContextSecurityManager, SecurityPolicy
)
from pact_ax.primitives.context_share.schemas import (
    ContextPacket, AgentIdentity, ContextType, TrustLevel,
    Priority, ContextMetadata, CollaborationOutcome
)


def main():
    print("=" * 70)
    print("Context Security Demo: Story-Aware Security")
    print("=" * 70)
    print()
    
    # Create agent with story keeper
    print("📝 Setting up Agent with Story Keeper...")
    agent = AgentIdentity(
        agent_id="secure-agent-001",
        agent_type="secure_agent",
        version="1.0.0",
        capabilities=["natural_language", "decrypt_task_knowledge"],
        specializations=[]
    )
    
    story_keeper = StoryKeeper("secure-agent-001")
    security_manager = ContextSecurityManager(
        agent,
        SecurityPolicy.TRUST_BASED,
        story_keeper=story_keeper
    )
    print()
    
    # Scenario: Building trust through secure interactions
    print("-" * 70)
    print("Scenario 1: Initial Secure Context Sharing (Low Trust)")
    print("-" * 70)
    print()
    
    # First interaction - low trust
    packet1 = ContextPacket(
        from_agent=agent,
        to_agent="partner-agent-001",
        context_type=ContextType.TASK_KNOWLEDGE,
        payload={
            "task": "customer_support",
            "priority": "high"
        },
        metadata=ContextMetadata(),
        #trust_required=TrustLevel.MINIMAL,
        priority=Priority.HIGH
    )
    
    secured1 = security_manager.secure_context_for_sharing(
        packet1,
        target_agent="partner-agent-001",
        trust_level=0.3  # Low initial trust
    )
    
    print(f"✓ Context secured with encryption: {secured1.metadata.encryption_level}")
    print()
    
    # Update trust - positive outcome
    print("-" * 70)
    print("Trust Update: Successful collaboration increases trust")
    print("-" * 70)
    print()
    
    security_manager.update_trust_relationship(
        agent_id="partner-agent-001",
        context_type=ContextType.TASK_KNOWLEDGE,
        new_trust_level=0.6,
        collaboration_outcome=CollaborationOutcome.POSITIVE
    )
    print()
    
    # Second interaction - higher trust
    print("-" * 70)
    print("Scenario 2: Another Sharing (Trust Increased)")
    print("-" * 70)
    print()
    
    packet2 = ContextPacket(
        from_agent=agent,
        to_agent="partner-agent-001",
        context_type=ContextType.TASK_KNOWLEDGE,
        payload={
            "task": "technical_support",
            "sensitive_data": "customer_details"
        },
        metadata=ContextMetadata(),
        trust_required=TrustLevel.BUILDING,
        priority=Priority.HIGH
    )
    
    secured2 = security_manager.secure_context_for_sharing(
        packet2,
        target_agent="partner-agent-001",
        trust_level=0.6  # Higher trust now
    )
    
    print(f"✓ Context secured with encryption: {secured2.metadata.encryption_level}")
    print("(Notice: encryption may be lighter due to higher trust)")
    print()
    
    # Scenario: Security threat
    print("-" * 70)
    print("Scenario 3: Trust Breach (Negative Outcome)")
    print("-" * 70)
    print()
    
    security_manager.update_trust_relationship(
        agent_id="partner-agent-001",
        context_type=ContextType.TASK_KNOWLEDGE,
        new_trust_level=0.2,  # Trust drops significantly
        collaboration_outcome=CollaborationOutcome.NEGATIVE
    )
    print()
    
    # Third interaction - trust damaged
    print("-" * 70)
    print("Scenario 4: Sharing After Trust Breach (Lower Trust)")
    print("-" * 70)
    print()
    
    packet3 = ContextPacket(
        from_agent=agent,
        to_agent="partner-agent-001",
        context_type=ContextType.TASK_KNOWLEDGE,
        payload={
            "task": "account_recovery"
        },
        metadata=ContextMetadata(),
        trust_required=TrustLevel.MINIMAL,
        priority=Priority.NORMAL
    )
    
    secured3 = security_manager.secure_context_for_sharing(
        packet3,
        target_agent="partner-agent-001",
        trust_level=0.2  # Trust restored to low
    )
    
    print(f"✓ Context secured with encryption: {secured3.metadata.encryption_level}")
    print("(Notice: encryption is stronger again due to lower trust)")
    print()
    
    # Show the security story
    print("=" * 70)
    print("Security Story Summary")
    print("=" * 70)
    print()
    
    story_summary = story_keeper.get_story_summary()
    print(f"Current Arc: {story_summary['current_arc']}")
    print(f"Total Interactions: {story_summary['total_interactions']}")
    print()
    
    print("Security Events as Story:")
    print("-" * 70)
    for interaction in story_keeper.interactions[-5:]:  # Last 5
        if interaction.get('metadata', {}).get('is_security_event'):
            print(f"📍 [{interaction['arc'].value}] {interaction['user_input']}")
            print(f"   Gravity: {interaction['metadata'].get('emotional_gravity', 0):.2f}")
            print()
    
    # Show security insights
    print("=" * 70)
    print("Security Analytics")
    print("=" * 70)
    print()
    
    insights = security_manager.get_security_insights()
    
    print("Security Metrics:")
    print(f"  Success Rate: {insights['security_metrics']['success_rate']:.2%}")
    print(f"  Total Contexts Secured: {insights['security_metrics']['total_contexts_secured']}")
    print(f"  Avg Response Time: {insights['security_metrics']['average_response_time']:.4f}s")
    print()
    
    print("Threat Assessment:")
    print(f"  Overall Threat Level: {insights['threat_assessment']['overall_threat_level']:.2f}")
    print(f"  Confidence: {insights['threat_assessment']['confidence_score']:.2%}")
    print()
    
    print("Trust Trends:")
    for agent_pair, trend in insights['trust_trends'].items():
        print(f"  {agent_pair}:")
        print(f"    Direction: {trend['direction']}")
        print(f"    Current Level: {trend['current_level']:.2f}")
        print(f"    Change Rate: {trend['change_rate']:+.3f}")
    print()
    
    # Demonstrate story-aware recall
    print("=" * 70)
    print("Story-Aware Security Recall")
    print("=" * 70)
    print()
    
    print("Recalling security events with high emotional gravity:")
    high_gravity_events = [
        i for i in story_keeper.interactions
        if i.get('metadata', {}).get('is_security_event')
        and i.get('metadata', {}).get('emotional_gravity', 0) > 0.5
    ]
    
    for event in high_gravity_events:
        print(f"  🔴 {event['user_input']}")
        print(f"     Response: {event['agent_response']}")
        print()
    
    print("=" * 70)
    print("Demo Complete: Security Events Are Part of Agent's Story! ✨")
    print("=" * 70)
    print()
    print("Key Features Demonstrated:")
    print("  • Security events tracked as narrative")
    print("  • Trust evolution affects security decisions")
    print("  • High-impact security events have emotional gravity")
    print("  • Security history is story-aware, not just logs")
    print("  • Agent 'remembers' security relationship contextually")


if __name__ == "__main__":
    main()
