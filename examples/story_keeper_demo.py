"""
Demo: Story Keeper in Action

Shows how Story Keeper maintains narrative coherence
across a conversation.
"""

from pact_ax.primitives.story_keeper import StoryKeeper, StoryArc


def main():
    print("=" * 60)
    print("Story Keeper Demo: Conscious Continuity in Action")
    print("=" * 60)
    print()
    
    # Initialize Story Keeper
    sk = StoryKeeper("demo-agent-001")
    
    # Simulate a natural conversation with arc transitions
    conversation = [
        # EXPLORATION phase
        ("What is Story Keeper?", 
         "Story Keeper maintains conscious continuity by transforming interactions into narrative memory."),
        
        ("How does it work?", 
         "It detects story arcs (exploration, collaboration, integration) and recalls context based on narrative coherence."),
        
        ("Why is this different from regular context?",
         "Regular systems retrieve similar text. Story Keeper retrieves narratively relevant context - it remembers the *story*."),
        
        # Transition to COLLABORATION
        ("Let's build something together",
         "Yes! I'd love to co-create with you. What should we build?"),
        
        ("We could start with improving arc detection",
         "Great idea. We could add more sophisticated signals or even use ML for arc classification."),
        
        ("Let's sketch the architecture first",
         "Perfect. Starting with clear design before coding is wise."),
        
        # Transition to INTEGRATION  
        ("Let me think about this for a bit",
         "Of course. Take the time you need to internalize."),
        
        ("I'm back with some thoughts",
         "Welcome back! What emerged during your reflection?"),
    ]
    
    # Process each interaction
    for user_msg, agent_msg in conversation:
        print(f"User: {user_msg}")
        print(f"Agent: {agent_msg}")
        sk.process_interaction(user_msg, agent_msg)
        print()
    
    # Show story summary
    print("=" * 60)
    print("Story Summary")
    print("=" * 60)
    summary = sk.get_story_summary()
    print(f"Current Arc: {summary['current_arc']}")
    print(f"Total Interactions: {summary['total_interactions']}")
    print(f"Arc Transitions: {summary['arc_transitions']}")
    print()
    
    # Show arc history
    if summary['arc_history']:
        print("Arc Transition History:")
        for transition in summary['arc_history']:
            print(f"  {transition['from']} → {transition['to']}")
        print()
    
    # Demonstrate arc-aware recall
    print("=" * 60)
    print("Arc-Aware Context Recall")
    print("=" * 60)
    print()
    
    print("📖 Recalling from EXPLORATION arc:")
    exploration_moments = sk.recall_from_arc(StoryArc.EXPLORATION, k=2)
    for moment in exploration_moments:
        print(f"  User: {moment['user_input']}")
    print()
    
    print("📖 Recalling from COLLABORATION arc:")
    collab_moments = sk.recall_from_arc(StoryArc.COLLABORATION, k=2)
    for moment in collab_moments:
        print(f"  User: {moment['user_input']}")
    print()
    
    # Demonstrate story-aware context retrieval
    print("=" * 60)
    print("Story-Aware vs Regular Retrieval")
    print("=" * 60)
    print()
    print("Query: 'What did we talk about building?'")
    print()
    print("Story-aware (prefers current arc = INTEGRATION):")
    context = sk.recall_for_context(prefer_current_arc=True, k=3)
    for ctx in context:
        print(f"  [{ctx['arc'].value}] User: {ctx['user_input']}")
    print()


if __name__ == "__main__":
    main()
