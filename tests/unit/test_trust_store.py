"""
tests/unit/test_trust_store.py
───────────────────────────────
Unit tests for TrustStore (SQLite persistence) and TrustManager
save/load round-trips.

Run with:  pytest tests/unit/test_trust_store.py -v
"""

import pytest

from pact_ax.primitives.trust_store import TrustStore
from pact_ax.primitives.trust_score import TrustManager
from pact_ax.primitives.context_share.schemas import (
    AgentTrustProfile,
    ContextType,
    CollaborationOutcome,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store():
    """In-memory SQLite store — isolated per test."""
    return TrustStore(":memory:")


@pytest.fixture
def profile():
    p = AgentTrustProfile(agent_id="agent-002")
    p.update_trust(
        context_type=ContextType.TASK_KNOWLEDGE,
        outcome=CollaborationOutcome.POSITIVE,
        impact=1.0,
        context={"note": "first interaction"},
    )
    return p


# ── TrustStore: save / load ───────────────────────────────────────────────────

class TestSaveAndLoad:

    def test_save_and_load_roundtrip(self, store, profile):
        store.save_profile("agent-001", profile)
        loaded = store.load_profiles("agent-001")
        assert "agent-002" in loaded

    def test_overall_trust_preserved(self, store, profile):
        store.save_profile("agent-001", profile)
        loaded = store.load_profiles("agent-001")
        assert abs(loaded["agent-002"].overall_trust - profile.overall_trust) < 1e-6

    def test_context_trust_preserved(self, store, profile):
        store.save_profile("agent-001", profile)
        loaded = store.load_profiles("agent-001")
        orig = profile.get_trust_for_context(ContextType.TASK_KNOWLEDGE)
        restored = loaded["agent-002"].get_trust_for_context(ContextType.TASK_KNOWLEDGE)
        assert abs(orig - restored) < 1e-6

    def test_interactions_list_preserved(self, store, profile):
        store.save_profile("agent-001", profile)
        loaded = store.load_profiles("agent-001")
        evo = loaded["agent-002"].trust_evolution.get(ContextType.TASK_KNOWLEDGE)
        assert evo is not None
        assert len(evo.interactions) > 0

    def test_unknown_owner_returns_empty(self, store):
        result = store.load_profiles("no-such-owner")
        assert result == {}

    def test_upsert_updates_existing(self, store, profile):
        store.save_profile("agent-001", profile)
        profile.update_trust(
            context_type=ContextType.TASK_KNOWLEDGE,
            outcome=CollaborationOutcome.POSITIVE,
            impact=1.0,
            context={},
        )
        store.save_profile("agent-001", profile)
        loaded = store.load_profiles("agent-001")
        assert len(loaded["agent-002"].trust_evolution[ContextType.TASK_KNOWLEDGE].interactions) >= 2

    def test_multiple_profiles_saved(self, store):
        for i in range(3):
            p = AgentTrustProfile(agent_id=f"target-{i}")
            store.save_profile("owner", p)
        loaded = store.load_profiles("owner")
        assert len(loaded) == 3

    def test_save_all_roundtrip(self, store):
        profiles = {
            f"agent-{i}": AgentTrustProfile(agent_id=f"agent-{i}")
            for i in range(4)
        }
        store.save_all("owner", profiles)
        loaded = store.load_profiles("owner")
        assert set(loaded.keys()) == set(profiles.keys())


# ── TrustStore: delete ────────────────────────────────────────────────────────

class TestDelete:

    def test_delete_returns_true(self, store, profile):
        store.save_profile("owner", profile)
        assert store.delete_profile("owner", "agent-002") is True

    def test_delete_removes_profile(self, store, profile):
        store.save_profile("owner", profile)
        store.delete_profile("owner", "agent-002")
        loaded = store.load_profiles("owner")
        assert "agent-002" not in loaded

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete_profile("owner", "ghost") is False


# ── TrustStore: list_owners ───────────────────────────────────────────────────

class TestListOwners:

    def test_empty_store(self, store):
        assert store.list_owners() == []

    def test_lists_after_save(self, store, profile):
        store.save_profile("owner-A", profile)
        store.save_profile("owner-B", profile)
        owners = store.list_owners()
        assert "owner-A" in owners
        assert "owner-B" in owners

    def test_distinct_owners_only(self, store):
        for i in range(3):
            p = AgentTrustProfile(agent_id=f"t-{i}")
            store.save_profile("same-owner", p)
        assert store.list_owners().count("same-owner") == 1


# ── TrustManager: save / load ─────────────────────────────────────────────────

class TestTrustManagerPersistence:

    def test_save_and_reload(self):
        tm = TrustManager("agent-001")
        tm.update_trust("agent-002", CollaborationOutcome.POSITIVE, ContextType.TASK_KNOWLEDGE)
        tm.save(":memory:")   # in-memory can't be reloaded, used via classmethod below
        # Use a real temp file for the roundtrip
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            tm.save(db_path)
            tm2 = TrustManager.load(db_path, "agent-001")
            assert "agent-002" in tm2._profiles
        finally:
            os.unlink(db_path)

    def test_trust_score_survives_reload(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            tm = TrustManager("owner")
            for _ in range(5):
                tm.update_trust("peer", CollaborationOutcome.POSITIVE, ContextType.TASK_KNOWLEDGE)
            original_score = tm.get_trust("peer", ContextType.TASK_KNOWLEDGE)
            tm.save(db_path)
            tm2 = TrustManager.load(db_path, "owner")
            reloaded_score = tm2.get_trust("peer", ContextType.TASK_KNOWLEDGE)
            assert abs(original_score - reloaded_score) < 1e-6
        finally:
            os.unlink(db_path)

    def test_load_missing_file_returns_default(self, tmp_path):
        tm = TrustManager.load(tmp_path / "no-such.db", "agent-X")
        assert tm.get_trust("anyone") == 0.5

    def test_load_returns_trusts_manager_type(self, tmp_path):
        tm = TrustManager.load(tmp_path / "fresh.db", "agent-Y")
        assert isinstance(tm, TrustManager)
        assert tm.agent_id == "agent-Y"

    def test_multiple_agents_isolated(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            tmA = TrustManager("A")
            tmA.update_trust("shared-peer", CollaborationOutcome.POSITIVE, ContextType.TASK_KNOWLEDGE)
            tmA.save(db_path)

            tmB = TrustManager("B")
            tmB.update_trust("shared-peer", CollaborationOutcome.NEGATIVE, ContextType.TASK_KNOWLEDGE)
            tmB.save(db_path)

            reloaded_A = TrustManager.load(db_path, "A")
            reloaded_B = TrustManager.load(db_path, "B")

            score_A = reloaded_A.get_trust("shared-peer", ContextType.TASK_KNOWLEDGE)
            score_B = reloaded_B.get_trust("shared-peer", ContextType.TASK_KNOWLEDGE)
            assert score_A > 0.5
            assert score_B < 0.5
        finally:
            os.unlink(db_path)
