"""
tests/unit/test_capability_registry.py
────────────────────────────────────────
Unit tests for CapabilityRegistry.

Run with:  pytest tests/unit/test_capability_registry.py -v
"""

import pytest
from pact_ax.primitives.capability_registry import CapabilityRegistry, Capability


@pytest.fixture
def reg():
    return CapabilityRegistry(":memory:")


@pytest.fixture
def populated(reg):
    reg.register("agent-a", "contract_review",  "Reviews NDAs",              tags=["legal"])
    reg.register("agent-b", "contract_review",  "Reviews service agreements", tags=["legal"])
    reg.register("agent-a", "tax_analysis",     "Federal tax implications",   tags=["legal", "finance"])
    reg.register("agent-c", "ip_licensing",     "Patent licensing",           tags=["legal", "ip"])
    return reg


# ── register ──────────────────────────────────────────────────────────────────

class TestRegister:

    def test_returns_capability(self, reg):
        cap = reg.register("agent-a", "contract_review")
        assert isinstance(cap, Capability)
        assert cap.agent_id == "agent-a"
        assert cap.skill == "contract_review"

    def test_stores_description_and_tags(self, reg):
        reg.register("agent-a", "contract_review", description="NDA review", tags=["legal"])
        caps = reg.get_agent_capabilities("agent-a")
        assert caps[0].description == "NDA review"
        assert "legal" in caps[0].tags

    def test_upsert_updates_description(self, reg):
        reg.register("agent-a", "contract_review", description="old")
        reg.register("agent-a", "contract_review", description="new")
        caps = reg.get_agent_capabilities("agent-a")
        assert len(caps) == 1
        assert caps[0].description == "new"

    def test_version_default(self, reg):
        cap = reg.register("agent-a", "contract_review")
        assert cap.version == "1.0"

    def test_custom_version(self, reg):
        cap = reg.register("agent-a", "contract_review", version="2.0")
        assert cap.version == "2.0"


# ── find_capable ──────────────────────────────────────────────────────────────

class TestFindCapable:

    def test_returns_all_capable_agents(self, populated):
        caps = populated.find_capable("contract_review")
        agent_ids = {c.agent_id for c in caps}
        assert agent_ids == {"agent-a", "agent-b"}

    def test_returns_empty_for_unknown_skill(self, populated):
        assert populated.find_capable("nonexistent") == []

    def test_single_agent_skill(self, populated):
        caps = populated.find_capable("ip_licensing")
        assert len(caps) == 1
        assert caps[0].agent_id == "agent-c"


# ── search ────────────────────────────────────────────────────────────────────

class TestSearch:

    def test_matches_skill_name(self, populated):
        results = populated.search("contract")
        skills = {r.skill for r in results}
        assert "contract_review" in skills

    def test_matches_description(self, populated):
        results = populated.search("Federal tax")
        assert any(r.skill == "tax_analysis" for r in results)

    def test_matches_tags(self, populated):
        results = populated.search("finance")
        assert any(r.skill == "tax_analysis" for r in results)

    def test_case_insensitive(self, populated):
        lower = populated.search("legal")
        upper = populated.search("LEGAL")
        assert len(lower) == len(upper)

    def test_no_match_returns_empty(self, populated):
        assert populated.search("zzznomatch") == []


# ── deregister ────────────────────────────────────────────────────────────────

class TestDeregister:

    def test_removes_specific_skill(self, populated):
        assert populated.deregister("agent-a", "tax_analysis") is True
        skills = {c.skill for c in populated.get_agent_capabilities("agent-a")}
        assert "tax_analysis" not in skills

    def test_returns_false_for_missing(self, populated):
        assert populated.deregister("agent-a", "nonexistent") is False

    def test_deregister_agent_removes_all(self, populated):
        count = populated.deregister_agent("agent-a")
        assert count == 2
        assert populated.get_agent_capabilities("agent-a") == []


# ── all_skills ────────────────────────────────────────────────────────────────

class TestAllSkills:

    def test_returns_unique_skills(self, populated):
        skills = populated.all_skills()
        assert sorted(skills) == sorted({"contract_review", "tax_analysis", "ip_licensing"})

    def test_empty_registry(self, reg):
        assert reg.all_skills() == []


# ── all_capabilities ──────────────────────────────────────────────────────────

class TestAllCapabilities:

    def test_returns_all_entries(self, populated):
        caps = populated.all_capabilities()
        assert len(caps) == 4

    def test_empty(self, reg):
        assert reg.all_capabilities() == []


# ── persistence ───────────────────────────────────────────────────────────────

class TestPersistence:

    def test_survives_new_instance(self, tmp_path):
        db = str(tmp_path / "caps.db")
        r1 = CapabilityRegistry(db)
        r1.register("agent-a", "contract_review", description="NDA")
        r2 = CapabilityRegistry(db)
        caps = r2.get_agent_capabilities("agent-a")
        assert len(caps) == 1
        assert caps[0].description == "NDA"
