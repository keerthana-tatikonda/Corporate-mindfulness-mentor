# graph/graph.py

from typing import TypedDict, List, Optional, Dict, Any

from langgraph.graph import StateGraph, END

from services.llm import client, MODEL
from .schemas import (
    Goal,
    PlanResponse,
    DecomposedPlan,
    UserProfile,
    PersonalizedPlanRequest,
    PersonalizedPlanResponse,
    WorkloadReport,
    AdaptedPlanResponse,
    CheckIn,
    DayAdjustment,  # <-- this exists in schemas.py
    StressAnalyticsResult, ProductivityInsightsResult
)
from .nodes import (
    generate_plan_node,
    generate_decomposed_plan,
    personalize_plan_node,
    adapt_plan_node,
    morning_checkin_node,
    motivational_message_node, hr_insights_node,
    stress_analytics_node, productivity_insights_node
)

# ---------------------------------------------------------------------
# 🧭 Goal Creation
# ---------------------------------------------------------------------


class GoalState(TypedDict):
    """State for the goal creation workflow."""
    goal_name: str
    duration_type: str
    description: str
    activities: List[str]
    summary: str
    confidence: Optional[float]
    confidence_note: Optional[str]


def create_goal_graph():
    """Create the LangGraph workflow for goal creation."""
    workflow = StateGraph(GoalState)
    workflow.add_node("generate_plan", generate_plan_node)
    workflow.set_entry_point("generate_plan")
    workflow.add_edge("generate_plan", END)
    return workflow.compile()


def run_goal_creation(
    goal_name: str, duration_type: str, description: str = ""
) -> PlanResponse:
    """
    Run the goal creation workflow using LangGraph.
    """
    graph = create_goal_graph()
    result = graph.invoke(
        {
            "goal_name": goal_name,
            "duration_type": duration_type,
            "description": description,
            "activities": [],
            "summary": "",
            # seed confidence fields so they stay in the graph state
            "confidence": None,
            "confidence_note": None,
        }
    )

    return PlanResponse(
        goal=result["goal_name"],
        suggested_activities=result["activities"],
        ai_summary=result["summary"],
        confidence=result.get("confidence"),
        confidence_note=result.get("confidence_note"),
    )


def run_goal_decomposition(
    goal_name: str, duration_type: str, description: str = ""
) -> DecomposedPlan:
    """
    High-level helper: create base plan and then a decomposed plan.
    """
    # 1) make base plan with existing pipeline
    base = run_goal_creation(goal_name, duration_type, description)

    # 2) ask the node to break it into subgoals
    g = Goal(
        goal_name=goal_name,
        duration_type=duration_type,
        description=description or None,
    )
    return generate_decomposed_plan(g, base)


# ---------------------------------------------------------------------
# 🎯 Profile Personalization
# ---------------------------------------------------------------------


class PersonalizeState(TypedDict, total=False):
    goal_name: str
    duration_type: str
    description: str
    profile: UserProfile  # Pydantic object
    p_activities: List[str]
    p_summary: str
    task_feedback: Dict[str, str]  # activity -> "helpful" | "not helpful"
    completion: Dict[str, bool]  # activity -> completed?
    p_confidence: Optional[float]
    p_confidence_note: Optional[str]


def create_personalize_graph():
    g = StateGraph(PersonalizeState)
    g.add_node("personalize", personalize_plan_node)
    g.set_entry_point("personalize")
    g.add_edge("personalize", END)
    return g.compile()


def run_personalized_goal(
    goal: Goal,
    profile: UserProfile,
    task_feedback: Optional[Dict[str, str]] = None,
    completion: Optional[Dict[str, bool]] = None,
) -> PersonalizedPlanResponse:
    graph = create_personalize_graph()
    out = graph.invoke(
        {
            "goal_name": goal.goal_name,
            "duration_type": goal.duration_type,
            "description": goal.description or "",
            "profile": profile,
            "task_feedback": task_feedback or {},
            "completion": completion or {},
            "p_activities": [],
            "p_summary": "",
            "p_confidence": None,
            "p_confidence_note": None,
        }
    )
    return PersonalizedPlanResponse(
        goal=goal.goal_name,
        activities=out["p_activities"],
        summary=out["p_summary"],
        confidence=out.get("p_confidence"),
        confidence_note=out.get("p_confidence_note"),
    )


# ---------------------------------------------------------------------
# ⚙️ Workload-Based Adaptation
# ---------------------------------------------------------------------


class AdaptState(TypedDict, total=False):
    goal_name: str
    duration_type: str
    base_activities: List[str]
    workload: WorkloadReport
    adapted_plan: List[str]
    adapted_rationale: str
    task_feedback: Dict[str, str]
    completion: Dict[str, bool]
    adapted_confidence: Optional[float]
    adapted_confidence_note: Optional[str]


def create_adaptation_graph():
    g = StateGraph(AdaptState)
    g.add_node("adapt", adapt_plan_node)
    g.set_entry_point("adapt")
    g.add_edge("adapt", END)
    return g.compile()


def run_workload_adaptation(
    goal: Goal,
    base_activities: List[str],
    workload: WorkloadReport,
    task_feedback: Optional[Dict[str, str]] = None,
    completion: Optional[Dict[str, bool]] = None,
) -> AdaptedPlanResponse:
    graph = create_adaptation_graph()
    out = graph.invoke(
        {
            "goal_name": goal.goal_name,
            "duration_type": goal.duration_type,
            "base_activities": base_activities,
            "workload": workload,
            "task_feedback": task_feedback or {},
            "completion": completion or {},
            "adapted_plan": [],
            "adapted_rationale": "",
            "adapted_confidence": None,
            "adapted_confidence_note": None,
        }
    )
    return AdaptedPlanResponse(
        goal=goal.goal_name,
        day_plan=out["adapted_plan"],
        rationale=out["adapted_rationale"],
        confidence=out.get("adapted_confidence"),
        confidence_note=out.get("adapted_confidence_note"),
    )


# ---------------------------------------------------------------------
# 🌅 Morning Check-In (LangGraph version, using DayAdjustment)
# ---------------------------------------------------------------------


class CheckInState(TypedDict, total=False):
    checkin: dict
    day_adjustment: dict


def create_checkin_graph():
    g = StateGraph(CheckInState)
    g.add_node("morning_checkin", morning_checkin_node)
    g.set_entry_point("morning_checkin")
    g.add_edge("morning_checkin", END)
    return g.compile()


def run_morning_checkin(checkin: CheckIn) -> DayAdjustment:
    """
    Run the morning check-in workflow via LangGraph and return a DayAdjustment
    Pydantic model (summary, focus_for_today, risk_flags, etc.).
    """
    compiled = create_checkin_graph()
    result = compiled.invoke({"checkin": checkin.model_dump()})
    da = result.get("day_adjustment") or {}
    return DayAdjustment(**da)


# Add these sections to the END of your graph/graph.py file

# ---------------------------------------------------------------------
# 📊 Stress Analytics Dashboard
# ---------------------------------------------------------------------

class StressAnalyticsState(TypedDict, total=False):
    checkins: List[Dict[str, Any]]
    stress_analytics: Dict[str, Any]  # StressAnalyticsResult as dict


def create_stress_analytics_graph():
    g = StateGraph(StressAnalyticsState)
    g.add_node("analyze_stress", stress_analytics_node)
    g.set_entry_point("analyze_stress")
    g.add_edge("analyze_stress", END)
    return g.compile()


def run_stress_analytics(checkins: List[Dict[str, Any]]) -> str:
    """
    LangGraph entrypoint for Stress Analytics user story.
    
    Parameters
    ----------
    checkins : List[Dict[str, Any]]
        Raw check-in records with mood, sleep, energy, workload, etc.
    
    Returns
    -------
    str
        AI-generated summary of stress trends and patterns.
    """
    compiled = create_stress_analytics_graph()
    out = compiled.invoke({
        "checkins": checkins,
        "stress_analytics": {},
    })
    
    # Extract the result
    analytics = out.get("stress_analytics") or {}
    
    # Build a readable summary from the StressAnalyticsResult
    summary = analytics.get("summary", "No stress trend analysis available.")
    key_drivers = analytics.get("key_drivers", [])
    suggestions = analytics.get("suggestions", [])
    
    # Format as a nice text block
    text = f"{summary}\n\n"
    
    if key_drivers:
        text += "**Main stress drivers:**\n"
        for driver in key_drivers:
            text += f"- {driver}\n"
        text += "\n"
    
    if suggestions:
        text += "**Suggestions for next week:**\n"
        for suggestion in suggestions:
            text += f"- {suggestion}\n"
    
    return text.strip()


# ---------------------------------------------------------------------
# 📈 Productivity vs Stress Insights
# ---------------------------------------------------------------------

class ProductivityInsightsState(TypedDict, total=False):
    records: List[Dict[str, Any]]  # {date, stress_score, productivity, ...}
    productivity_insights: Dict[str, Any]  # ProductivityInsightsResult as dict


def create_productivity_insights_graph():
    g = StateGraph(ProductivityInsightsState)
    g.add_node("analyze_productivity", productivity_insights_node)
    g.set_entry_point("analyze_productivity")
    g.add_edge("analyze_productivity", END)
    return g.compile()


def run_productivity_insights(records: List[Dict[str, Any]]) -> str:
    """
    LangGraph entrypoint for Productivity vs Stress Insights user story.
    
    Parameters
    ----------
    records : List[Dict[str, Any]]
        Each record has: date, stress_score (0-100), productivity (0-10),
        plus optional fields like mood, workload, prod_notes.
    
    Returns
    -------
    str
        AI-generated insight about the relationship between stress and productivity.
    """
    compiled = create_productivity_insights_graph()
    out = compiled.invoke({
        "records": records,
        "productivity_insights": {},
    })
    
    # Extract the result
    insights = out.get("productivity_insights") or {}
    
    correlation = insights.get("correlation_summary", "No productivity data available.")
    risk_windows = insights.get("risk_windows", [])
    suggestions = insights.get("suggestions", [])
    
    # Format as readable text
    text = f"{correlation}\n\n"
    
    if risk_windows:
        text += "**High-risk patterns:**\n"
        for window in risk_windows:
            text += f"- {window}\n"
        text += "\n"
    
    if suggestions:
        text += "**Recommendations:**\n"
        for suggestion in suggestions:
            text += f"- {suggestion}\n"
    
    return text.strip()


class MotivationState(TypedDict, total=False):
    completed: int
    total: int
    activities: List[str]
    message: str
    confidence: Optional[float]
    confidence_note: Optional[str]


def create_motivation_graph():
    g = StateGraph(MotivationState)
    g.add_node("motivate", motivational_message_node)
    g.set_entry_point("motivate")
    g.add_edge("motivate", END)
    return g.compile()


def run_motivation_message(
    completed: int,
    total: int,
    activities: List[str],
) -> dict:
    """
    LangGraph entrypoint for Motivational Messaging user story.
    Returns dict with message, confidence, and confidence_note.
    """
    compiled = create_motivation_graph()
    out = compiled.invoke({
        "completed": completed,
        "total": total,
        "activities": activities,
        "message": "",
        "confidence": None,
        "confidence_note": None,
    })
    return {
        "message": out.get("message", ""),
        "confidence": out.get("confidence"),
        "confidence_note": out.get("confidence_note", ""),
    }

class HRInsightsState(TypedDict, total=False):
    stress_series: List[Dict[str, Any]]
    summary: str
    confidence: Optional[float]
    confidence_note: Optional[str]


def create_hr_insights_graph():
    g = StateGraph(HRInsightsState)
    g.add_node("hr_insights", hr_insights_node)
    g.set_entry_point("hr_insights")
    g.add_edge("hr_insights", END)
    return g.compile()


def run_hr_insights(stress_series: List[Dict[str, Any]]) -> dict:
    """
    LangGraph entrypoint for HR Wellness Insights user story.
    Returns dict with summary, confidence, and confidence_note.
    """
    compiled = create_hr_insights_graph()
    out = compiled.invoke({
        "stress_series": stress_series,
        "summary": "",
        "confidence": None,
        "confidence_note": None,
    })
    return {
        "summary": out.get("summary", ""),
        "confidence": out.get("confidence"),
        "confidence_note": out.get("confidence_note", ""),
    }


