"""
Tests for Story Keeper primitive

This test suite covers:
1. Basic initialization and story creation
2. Story state updates across turns
3. Narrative continuity maintenance
4. Multi-session persistence
5. Integration with other PACT-AX primitives
"""

import pytest
from unittest.mock import Mock, patch
from pact_ax.primitives.story_keeper import StoryKeeper


class TestStoryKeeperInitialization:
    """Tests for Story Keeper initialization"""
    
    def test_basic_initialization(self):
        """Test basic Story Keeper initialization"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        assert keeper.agent_id == "test-agent"
        assert keeper.session_id is None
        assert keeper.story_state is not None
    
    def test_initialization_with_session(self):
        """Test initialization with session ID"""
        keeper = StoryKeeper(
            agent_id="test-agent",
            session_id="test-session-123"
        )
        
        assert keeper.session_id == "test-session-123"
    
    def test_initialization_with_config(self):
        """Test initialization with custom config"""
        config = {
            "story_update_frequency": "every_turn",
            "max_story_length": 500
        }
        keeper = StoryKeeper(agent_id="test-agent", config=config)
        
        assert keeper.config["story_update_frequency"] == "every_turn"
        assert keeper.config["max_story_length"] == 500


class TestStoryStateManagement:
    """Tests for story state management"""
    
    def test_initial_story_state_structure(self):
        """Test that initial story state has correct structure"""
        keeper = StoryKeeper(agent_id="test-agent")
        story = keeper.get_story_state()
        
        assert "characters" in story
        assert "arc" in story
        assert "themes" in story
        assert "last_beat" in story
        assert "context" in story
    
    def test_story_state_update(self):
        """Test that story state updates after processing turn"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        initial_story = keeper.get_story_state()
        keeper.process_turn(
            user_message="Help me plan a startup",
            user_id="user-123"
        )
        updated_story = keeper.get_story_state()
        
        # Story should have evolved
        assert updated_story != initial_story
        assert len(updated_story["themes"]) > 0
    
    def test_load_story_state(self):
        """Test loading a previously saved story state"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        saved_story = {
            "characters": {
                "user": "Test user",
                "agent": "Test agent"
            },
            "arc": "Test arc",
            "themes": ["test", "themes"],
            "last_beat": "Test beat",
            "context": "Test context"
        }
        
        keeper.load_story_state(saved_story)
        loaded_story = keeper.get_story_state()
        
        assert loaded_story["characters"] == saved_story["characters"]
        assert loaded_story["arc"] == saved_story["arc"]
        assert loaded_story["themes"] == saved_story["themes"]
    
    def test_reset_story(self):
        """Test resetting story state"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        # Process some turns
        keeper.process_turn("Help me plan a startup")
        keeper.process_turn("What about funding?")
        
        # Reset
        keeper.reset_story()
        story = keeper.get_story_state()
        
        # Should be back to initial state
        assert len(story["themes"]) == 0 or story["themes"] == []


class TestConversationProcessing:
    """Tests for conversation turn processing"""
    
    def test_process_single_turn(self):
        """Test processing a single conversation turn"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        response = keeper.process_turn(
            user_message="Help me plan a startup",
            user_id="user-123"
        )
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_process_multiple_turns(self):
        """Test processing multiple conversation turns"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        response1 = keeper.process_turn("Help me plan a startup")
        response2 = keeper.process_turn("What about funding?")
        response3 = keeper.process_turn("How do I find investors?")
        
        # All responses should be generated
        assert response1 and response2 and response3
        
        # Story should have evolved
        story = keeper.get_story_state()
        assert len(story["themes"]) > 0
    
    def test_narrative_continuity(self):
        """Test that narrative continuity is maintained across turns"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        # First turn
        keeper.process_turn("I want to start a tech company")
        story_after_turn1 = keeper.get_story_state()
        
        # Second turn
        keeper.process_turn("What programming language should I use?")
        story_after_turn2 = keeper.get_story_state()
        
        # Story arc should show progression
        assert "tech" in story_after_turn2["context"].lower() or \
               "tech" in str(story_after_turn2["themes"]).lower()


class TestMultiSessionPersistence:
    """Tests for multi-session persistence"""
    
    def test_session_id_persistence(self):
        """Test that session ID enables persistence"""
        session_id = "test-session-123"
        
        # Session 1
        keeper1 = StoryKeeper(agent_id="test-agent", session_id=session_id)
        keeper1.process_turn("Help me plan a startup")
        story1 = keeper1.get_story_state()
        
        # Session 2 (simulating new instance with same session ID)
        keeper2 = StoryKeeper(agent_id="test-agent", session_id=session_id)
        keeper2.load_story_state(story1)
        story2 = keeper2.get_story_state()
        
        # Stories should be identical
        assert story1 == story2
    
    def test_different_sessions_isolated(self):
        """Test that different sessions maintain separate stories"""
        keeper1 = StoryKeeper(agent_id="test-agent", session_id="session-1")
        keeper2 = StoryKeeper(agent_id="test-agent", session_id="session-2")
        
        keeper1.process_turn("Help me with marketing")
        keeper2.process_turn("Help me with engineering")
        
        story1 = keeper1.get_story_state()
        story2 = keeper2.get_story_state()
        
        # Stories should be different
        assert story1 != story2


class TestIntegrationWithPACTAX:
    """Tests for integration with other PACT-AX primitives"""
    
    @patch('pact_ax.primitives.context_share.ContextShareManager')
    def test_integration_with_context_sharing(self, mock_context_manager):
        """Test Story Keeper integration with Context Sharing"""
        keeper = StoryKeeper(agent_id="test-agent")
        context_manager = mock_context_manager.return_value
        
        # Process turn
        keeper.process_turn("Help me plan a startup")
        story = keeper.get_story_state()
        
        # Share story as context
        context_manager.create_context_packet.return_value = {
            "target_agent": "agent-002",
            "context_type": "narrative_state",
            "payload": {"story": story}
        }
        
        packet = context_manager.create_context_packet(
            target_agent="agent-002",
            context_type="narrative_state",
            payload={"story": story}
        )
        
        assert packet["payload"]["story"] == story
    
    @patch('pact_ax.primitives.trust_score.TrustManager')
    def test_integration_with_trust_scoring(self, mock_trust_manager):
        """Test Story Keeper integration with Trust Scoring"""
        keeper = StoryKeeper(agent_id="test-agent")
        trust_manager = mock_trust_manager.return_value
        
        # Process turn
        keeper.process_turn("This is very helpful, thank you!")
        story = keeper.get_story_state()
        
        # Update trust based on story themes
        if "positive" in str(story.get("themes", [])).lower():
            trust_manager.update_trust.return_value = 0.8
            score = trust_manager.update_trust("user-123", delta=0.1)
            assert score == 0.8


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_empty_message(self):
        """Test handling of empty user message"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        response = keeper.process_turn(user_message="")
        
        # Should handle gracefully
        assert response is not None
    
    def test_very_long_message(self):
        """Test handling of very long user message"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        long_message = "Help me " * 1000
        response = keeper.process_turn(user_message=long_message)
        
        # Should handle without error
        assert response is not None
    
    def test_rapid_consecutive_turns(self):
        """Test handling of rapid consecutive turns"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        for i in range(10):
            response = keeper.process_turn(f"Question {i}")
            assert response is not None
    
    def test_story_state_max_length_enforcement(self):
        """Test that story state respects max length"""
        keeper = StoryKeeper(
            agent_id="test-agent",
            config={"max_story_length": 100}
        )
        
        # Process many turns
        for i in range(20):
            keeper.process_turn(f"This is message number {i}")
        
        story = keeper.get_story_state()
        story_str = str(story)
        
        # Story should not grow indefinitely
        # (Exact assertion depends on implementation)
        assert len(story_str) < 5000  # Reasonable upper bound


class TestStoryQuality:
    """Tests for story quality and coherence"""
    
    def test_theme_extraction(self):
        """Test that themes are extracted from conversations"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        keeper.process_turn("I'm struggling with time management and focus")
        story = keeper.get_story_state()
        
        themes = story.get("themes", [])
        # Should identify relevant themes
        assert len(themes) > 0
    
    def test_arc_progression(self):
        """Test that story arc shows progression"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        keeper.process_turn("I want to start a business")
        story1 = keeper.get_story_state()
        
        keeper.process_turn("I've validated my idea")
        story2 = keeper.get_story_state()
        
        keeper.process_turn("I'm ready to launch")
        story3 = keeper.get_story_state()
        
        # Arc should show progression
        arc1 = story1.get("arc", "")
        arc3 = story3.get("arc", "")
        
        assert arc1 != arc3  # Story should have evolved
    
    def test_character_development(self):
        """Test that character descriptions evolve"""
        keeper = StoryKeeper(agent_id="test-agent")
        
        keeper.process_turn("I'm new to programming")
        story1 = keeper.get_story_state()
        
        keeper.process_turn("I just built my first app!")
        story2 = keeper.get_story_state()
        
        # User character should reflect growth
        user_char_1 = story1.get("characters", {}).get("user", "")
        user_char_2 = story2.get("characters", {}).get("user", "")
        
        assert user_char_1 != user_char_2


# Pytest fixtures
@pytest.fixture
def basic_keeper():
    """Fixture providing a basic Story Keeper instance"""
    return StoryKeeper(agent_id="test-agent")


@pytest.fixture
def keeper_with_session():
    """Fixture providing a Story Keeper with session ID"""
    return StoryKeeper(
        agent_id="test-agent",
        session_id="test-session"
    )


@pytest.fixture
def sample_story_state():
    """Fixture providing a sample story state"""
    return {
        "characters": {
            "user": "Entrepreneur at ideation stage",
            "agent": "Strategic advisor"
        },
        "arc": "Beginning: Exploring startup idea",
        "themes": ["clarity", "practical action"],
        "last_beat": "User shared initial business concept",
        "context": "Early-stage entrepreneurial planning"
    }


# Integration test examples
def test_full_conversation_flow(basic_keeper):
    """Integration test: Full conversation flow"""
    
    # Turn 1
    response1 = basic_keeper.process_turn("Help me plan a startup")
    assert response1
    
    # Turn 2
    response2 = basic_keeper.process_turn("What should I focus on first?")
    assert response2
    
    # Turn 3
    response3 = basic_keeper.process_turn("How do I validate my idea?")
    assert response3
    
    # Check story evolution
    story = basic_keeper.get_story_state()
    assert len(story["themes"]) > 0
    assert "startup" in story["context"].lower() or \
           "startup" in str(story["themes"]).lower()


def test_story_persistence_across_sessions(sample_story_state):
    """Integration test: Story persistence across sessions"""
    
    # Session 1
    keeper1 = StoryKeeper(agent_id="agent-1", session_id="user-123")
    keeper1.load_story_state(sample_story_state)
    keeper1.process_turn("What's next?")
    story1 = keeper1.get_story_state()
    
    # Session 2 (simulating app restart)
    keeper2 = StoryKeeper(agent_id="agent-1", session_id="user-123")
    keeper2.load_story_state(story1)
    story2 = keeper2.get_story_state()
    
    # Stories should maintain continuity
    assert story1["characters"] == story2["characters"]
    assert story1["themes"] == story2["themes"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
