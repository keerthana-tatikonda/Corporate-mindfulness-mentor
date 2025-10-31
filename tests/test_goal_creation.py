# tests/test_goal_creation.py
"""
Unit tests for User Story 1: Goal Creation
Tests validate goal creation functionality and parsing logic with LangGraph.
"""

import pytest
from graph.schemas import PlanResponse
from graph.graph import run_goal_creation
from graph.nodes import (
    generate_plan_node,
    _parse_bullets,
    _get_fallback_activities,
    _get_fallback_summary,
    _parse_structured_response
)


# ---- Test Scenarios ----

def test_goal_creation_with_valid_inputs():
    """
    Scenario 1: User creates a goal with valid inputs
    Given: A user provides goal name, duration, and description
    When: The system processes the goal
    Then: A plan with activities and summary is generated
    """
    result = run_goal_creation(
        goal_name="reduce daily stress",
        duration_type="daily",
        description="Work 9-5, high-pressure meetings"
    )
    
    assert isinstance(result, PlanResponse)
    assert result.goal == "reduce daily stress"
    assert len(result.suggested_activities) >= 3
    assert len(result.suggested_activities) <= 5
    assert len(result.ai_summary) > 0
    assert result.ai_summary != ""


def test_goal_creation_without_description():
    """
    Scenario 2: User creates a goal without description
    Given: A user provides only goal name and duration
    When: The system processes the goal
    Then: A plan is still generated successfully
    """
    result = run_goal_creation(
        goal_name="improve focus",
        duration_type="weekly",
        description=""
    )
    
    assert isinstance(result, PlanResponse)
    assert result.goal == "improve focus"
    assert len(result.suggested_activities) > 0


def test_parse_bullets_with_dashes():
    """
    Scenario 3: Parse bullet points with dash format
    Given: Text contains activities with dash bullets
    When: Parsing is performed
    Then: Activities are correctly extracted
    """
    text = """
    - Morning meditation (10 minutes)
    - Afternoon walk (15 minutes)
    - Evening journaling (5 minutes)
    
    This routine helps reduce stress throughout the day.
    """
    
    activities, summary = _parse_bullets(text)
    
    assert len(activities) == 3
    assert "Morning meditation (10 minutes)" in activities
    assert "stress" in summary.lower()


def test_parse_bullets_with_numbers():
    """
    Scenario 4: Parse bullet points with numbered format
    Given: Text contains activities with numbered bullets
    When: Parsing is performed
    Then: Activities are correctly extracted
    """
    text = """
    1. Box breathing exercise (5 minutes)
    2. Desk stretches (10 minutes)
    3. Mindful eating at lunch
    
    These activities can be done at your desk.
    """
    
    activities, summary = _parse_bullets(text)
    
    assert len(activities) == 3
    assert "Box breathing exercise (5 minutes)" in activities
    assert "desk" in summary.lower()


def test_fallback_activities_daily():
    """
    Scenario 5: Fallback activities for daily duration
    Given: The system needs fallback activities for daily plan
    When: Fallback is requested
    Then: Appropriate daily activities are provided
    """
    activities = _get_fallback_activities("daily")
    
    assert len(activities) >= 3
    assert any("breathing" in act.lower() for act in activities)


def test_fallback_activities_weekly():
    """
    Scenario 6: Fallback activities for weekly duration
    Given: The system needs fallback activities for weekly plan
    When: Fallback is requested
    Then: Appropriate weekly activities are provided
    """
    activities = _get_fallback_activities("weekly")
    
    assert len(activities) >= 3
    assert any("week" in act.lower() for act in activities)


def test_fallback_activities_monthly():
    """
    Scenario 7: Fallback activities for monthly duration
    Given: The system needs fallback activities for monthly plan
    When: Fallback is requested
    Then: Appropriate monthly activities are provided
    """
    activities = _get_fallback_activities("monthly")
    
    assert len(activities) >= 3
    assert any("month" in act.lower() or "weekly" in act.lower() for act in activities)


def test_fallback_summary_generation():
    """
    Scenario 8: Generate fallback summary
    Given: The system needs a fallback summary
    When: Summary is requested for each duration type
    Then: Appropriate summaries are provided
    """
    daily_summary = _get_fallback_summary("daily")
    weekly_summary = _get_fallback_summary("weekly")
    monthly_summary = _get_fallback_summary("monthly")
    
    assert len(daily_summary) > 20
    assert len(weekly_summary) > 20
    assert len(monthly_summary) > 20
    assert "stress" in daily_summary.lower() or "mindfulness" in daily_summary.lower()


def test_goal_with_long_description():
    """
    Scenario 9: Goal with detailed description
    Given: A user provides a lengthy context description
    When: The system processes the goal
    Then: The plan is generated without errors
    """
    long_description = (
        "I work in a high-pressure tech company with back-to-back meetings. "
        "I often skip lunch and feel anxious by end of day. "
        "I have 30 minutes in the morning and evening for self-care. "
        "I prefer activities I can do at my desk or during short breaks."
    )
    
    result = run_goal_creation(
        goal_name="manage workplace anxiety",
        duration_type="daily",
        description=long_description
    )
    
    assert isinstance(result, PlanResponse)
    assert len(result.suggested_activities) >= 3


def test_activities_are_limited_to_five():
    """
    Scenario 10: Ensure maximum 5 activities
    Given: The LLM might return more than 5 activities
    When: The plan is generated
    Then: Only 5 activities are included in the response
    """
    result = run_goal_creation(
        goal_name="comprehensive wellness plan",
        duration_type="daily",
        description=""
    )
    
    assert len(result.suggested_activities) <= 5


def test_langgraph_node_function():
    """
    Scenario 11: Test the LangGraph node function directly
    Given: A state dictionary with goal information
    When: The generate_plan_node is called
    Then: State is updated with activities and summary
    """
    input_state = {
        "goal_name": "reduce stress",
        "duration_type": "daily",
        "description": "Test context",
        "activities": [],
        "summary": ""
    }
    
    result_state = generate_plan_node(input_state)
    
    assert "activities" in result_state
    assert "summary" in result_state
    assert len(result_state["activities"]) >= 3
    assert len(result_state["summary"]) > 0


def test_langgraph_workflow_execution():
    """
    Scenario 12: Test complete LangGraph workflow
    Given: Valid goal inputs
    When: run_goal_creation is called
    Then: The LangGraph workflow executes successfully
    """
    result = run_goal_creation(
        goal_name="improve mental clarity",
        duration_type="weekly",
        description="Need to focus better at work"
    )
    
    assert isinstance(result, PlanResponse)
    assert result.goal == "improve mental clarity"
    assert len(result.suggested_activities) >= 3
    assert len(result.ai_summary) > 0


# ---- Running Tests ----
if __name__ == "__main__":
    pytest.main([__file__, "-v"])