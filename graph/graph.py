# graph/graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from .schemas import Goal, PlanResponse, DecomposedPlan
from .nodes import generate_plan_node, generate_decomposed_plan  # NOTE: no import of generate_goal_plan


class GoalState(TypedDict):
    """State for the goal creation workflow."""
    goal_name: str
    duration_type: str
    description: str
    activities: List[str]
    summary: str


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
        "summary": ""
    })
    return PlanResponse(
        goal=result["goal_name"],
        suggested_activities=result["activities"],
        ai_summary=result["summary"]
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
