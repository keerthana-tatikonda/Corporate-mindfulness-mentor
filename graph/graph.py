# graph/graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict
from .schemas import (
    Goal, PlanResponse, DecomposedPlan,
    UserProfile, PersonalizedPlanRequest, PersonalizedPlanResponse,
    WorkloadReport, AdaptedPlanResponse
)
from .nodes import (
    generate_plan_node, generate_decomposed_plan,
    personalize_plan_node, adapt_plan_node
)


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


def run_goal_creation(goal_name: str, duration_type: str, description: str = "") -> PlanResponse:
    """
    Run the goal creation workflow using LangGraph.
    """
    graph = create_goal_graph()
    result = graph.invoke({
        "goal_name": goal_name,
        "duration_type": duration_type,
        "description": description,
        "activities": [],
        "summary": "",
        # 👇 seed confidence fields so they stay in the graph state
        "confidence": None,
        "confidence_note": None,
    })

    return PlanResponse(
        goal=result["goal_name"],
        suggested_activities=result["activities"],
        ai_summary=result["summary"],
        confidence=result.get("confidence"),
        confidence_note=result.get("confidence_note"),
    )


def run_goal_decomposition(goal_name: str, duration_type: str, description: str = "") -> DecomposedPlan:
    """
    High-level helper: create base plan and then a decomposed plan.
    """
    # 1) make base plan with existing pipeline
    base = run_goal_creation(goal_name, duration_type, description)

    # 2) ask the node to break it into subgoals
    g = Goal(goal_name=goal_name, duration_type=duration_type, description=description or None)
    return generate_decomposed_plan(g, base)

# --- Profile Personalization ---

class PersonalizeState(TypedDict, total=False):
    goal_name: str
    duration_type: str
    description: str
    profile: UserProfile  # Pydantic object
    p_activities: List[str]
    p_summary: str
    task_feedback: Dict[str, str]   # optional; activity -> "helpful" | "not helpful"
    completion: Dict[str, bool]     # optional; activity -> completed?
    #confidence
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
    out = graph.invoke({
        "goal_name": goal.goal_name,
        "duration_type": goal.duration_type,
        "description": goal.description or "",
        "profile": profile,
        "task_feedback": task_feedback or {},
        "completion": completion or {},
        "p_activities": [],
        "p_summary": "",
        # NEW seeds so these exist in state
        "p_confidence": None,
        "p_confidence_note": None,
    })
    return PersonalizedPlanResponse(
        goal=goal.goal_name,
        activities=out["p_activities"],
        summary=out["p_summary"],
        # NEW: surface confidence in the response model
        confidence=out.get("p_confidence"),
        confidence_note=out.get("p_confidence_note"),
    )


# --- Workload-Based Adaptation ---

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
    out = graph.invoke({
        "goal_name": goal.goal_name,
        "duration_type": goal.duration_type,
        "base_activities": base_activities,
        "workload": workload,
        "task_feedback": task_feedback or {},
        "completion": completion or {},
        "adapted_plan": [],
        "adapted_rationale": "",
                # confidence
        "adapted_confidence": None,
        "adapted_confidence_note": None,
    })
    return AdaptedPlanResponse(
        goal=goal.goal_name,
        day_plan=out["adapted_plan"],
        rationale=out["adapted_rationale"],
                # confidence
        confidence=out.get("adapted_confidence"),
        confidence_note=out.get("adapted_confidence_note"),
    )


# --- Morning Check-In Workflow (additive) ---
from typing import TypedDict
from langgraph.graph import StateGraph, END
from .nodes import morning_checkin_node
from .schemas import CheckIn, DayAdjustment

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
    compiled = create_checkin_graph()
    result = compiled.invoke({"checkin": checkin.model_dump()})
    da = result.get("day_adjustment") or {}
    return DayAdjustment(**da)

