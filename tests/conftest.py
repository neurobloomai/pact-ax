import pytest
from pathlib import Path
import tempfile

@pytest.fixture
def temp_dir():
    """Temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_agent_ids():
    """Sample agent IDs for testing"""
    return ["test-agent-001", "test-agent-002", "test-agent-003"]
