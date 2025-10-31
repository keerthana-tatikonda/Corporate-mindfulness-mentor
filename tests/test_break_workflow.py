import json, re
from utils.messages import get_random_break_message
from services.break_agent import MindfulBreakAgent
from graph.break_graph import run_llm_break_workflow

# ------------------- UNIT TESTS -------------------

def test_random_break_message_not_empty():
    msg = get_random_break_message()
    assert isinstance(msg, str)
    assert len(msg) > 5

def test_record_break_creates_log(tmp_path):
    agent = MindfulBreakAgent()
    agent.log_path = tmp_path / "test_break_log.json"
    result = agent.record_break("Time to stretch your body.")

    with open(agent.log_path) as f:
        logs = json.load(f)

    assert len(logs) == 1
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", logs[0]["scheduled_time"])

def test_llm_workflow_output_structure(monkeypatch):
    """Mock the LLM so we don't call the API each time."""
    class MockResponse:
        def __init__(self, content): self.content = "Mocked reflection"
    monkeypatch.setattr("graph.break_graph.ChatOpenAI.invoke", lambda *a, **k: MockResponse("Mocked"))
    
    output = run_llm_break_workflow()
    assert isinstance(output, dict)
    assert any(k in output for k in ["message", "reflection", "recommendation"])
