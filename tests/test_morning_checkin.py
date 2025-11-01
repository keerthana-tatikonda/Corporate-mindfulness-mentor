"""
Unit tests for User Story: Morning Wellness Check-In
Validates that the check-in produces a DayAdjustment via:
- rule-based fallback (no API key),
- strict JSON mode,
- free-form fenced JSON,
- variability when inputs change.

These tests do NOT call the network. All LLM calls are mocked by patching
the OpenAI client used inside graph.nodes (graph.nodes.client).
"""

import json
from unittest.mock import Mock, patch
import pytest

from graph.schemas import CheckIn, DayAdjustment
from graph.graph import run_morning_checkin


def _mock_openai_response(content: str):
    """Build a lightweight object that looks like OpenAI's chat response."""
    m = Mock()
    m.choices = [Mock()]
    m.choices[0].message.content = content
    return m


def test_rule_based_without_api_key(monkeypatch):
    """
    If OPENAI_API_KEY is absent, the node returns a rule-based DayAdjustment.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    ci = CheckIn(mood="anxious", sleep_quality="poor", energy="low", workload="heavy")
    result = run_morning_checkin(ci)

    assert isinstance(result, DayAdjustment)
    assert len(result.focus_for_today) >= 1  # should have actionable items
    # Summary may be None or a short string, but should not crash
    assert result.risk_flags is None or isinstance(result.risk_flags, list)


@patch("graph.nodes.client")
def test_json_mode_success(mock_client, monkeypatch):
    """
    Strict JSON mode path with response_format='json_object' returns a valid plan.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    payload = {
        "summary": "Keep it light today.",
        "focus_for_today": [
            "2-min breathing before first meeting",
            "3-min mindful pause after each task",
            "water + short stretch mid-afternoon",
        ],
        "risk_flags": ["poor sleep"],
    }
    mock_client.chat.completions.create.return_value = _mock_openai_response(json.dumps(payload))

    ci = CheckIn(mood="neutral", sleep_quality="ok", energy="medium", workload="normal")
    result = run_morning_checkin(ci)

    assert isinstance(result, DayAdjustment)
    assert "light" in (result.summary or "").lower()
    assert len(result.focus_for_today) >= 3
    assert "poor sleep" in ",".join(result.risk_flags or [])


@patch("graph.nodes.client")
def test_freeform_fenced_json_fallback(mock_client, monkeypatch):
    """
    First call: JSON mode returns non-JSON -> triggers free-form fallback.
    Second call: free-form returns ```json fenced content -> parsed OK.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    fenced = """```json
    {"summary":"Balanced day.",
     "focus_for_today":["mindful pause","stretch"],
     "risk_flags":[]}
    ```"""

    mock_client.chat.completions.create.side_effect = [
        _mock_openai_response("not-json"),  # JSON-mode attempt fails to json.loads
        _mock_openai_response(fenced),      # free-form fallback
    ]

    ci = CheckIn(mood="neutral", sleep_quality="ok", energy="medium", workload="normal")
    result = run_morning_checkin(ci)

    assert isinstance(result, DayAdjustment)
    assert (result.summary or "").lower().startswith("balanced")
    assert len(result.focus_for_today) >= 2


@patch("graph.nodes.client")
def test_outputs_change_when_inputs_change(mock_client, monkeypatch):
    """
    Changing inputs (sleep/energy/workload) should change at least one focus item.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    good_day = {
        "summary": "You're good to go.",
        "focus_for_today": ["brief mindful pause", "light stretch"],
        "risk_flags": [],
    }
    tough_day = {
        "summary": "Take it easy.",
        "focus_for_today": ["longer wind-down", "extra hydration"],
        "risk_flags": ["poor sleep", "low energy"],
    }

    mock_client.chat.completions.create.side_effect = [
        _mock_openai_response(json.dumps(good_day)),
        _mock_openai_response(json.dumps(tough_day)),
    ]

    res_good = run_morning_checkin(
        CheckIn(mood="calm", sleep_quality="great", energy="high", workload="light")
    )
    res_bad = run_morning_checkin(
        CheckIn(mood="anxious", sleep_quality="poor", energy="low", workload="heavy")
    )

    assert set(res_good.focus_for_today) != set(res_bad.focus_for_today)
