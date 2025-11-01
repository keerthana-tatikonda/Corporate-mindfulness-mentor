# tests/test_mentor_graph.py
from services.mentor_graph import run_mentor_cycle


import pytest
from services import mentor_graph

@pytest.fixture(autouse=True)
def mock_run_mentor_cycle(monkeypatch):
    def fake_run_mentor_cycle(*args, **kwargs):
        return {
            "techniques": [
                {
                    "title": "Box Breathing",
                    "duration_min": 5,
                    "description": "A calming breath cycle",
                    "steps": ["Inhale", "Hold", "Exhale", "Hold"],
                    "motivation": "Stay calm and grounded"
                }
            ],
            "summary": "Keep breathing mindfully."
        }
    monkeypatch.setattr(mentor_graph, "run_mentor_cycle", fake_run_mentor_cycle)


def test_graph_returns_recommendations():
    out = run_mentor_cycle(
        user_goal="Feeling anxious before a client demo",
        stress_level=8,
        profile={"time_available": "5 min", "experience": "Beginner"},
        history=[]
    )
    assert isinstance(out["recommended_ids"], list)
    assert len(out["recommended_ids"]) > 0
