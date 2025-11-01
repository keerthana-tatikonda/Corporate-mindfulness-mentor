# services/mentor_graph.py
from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ---------- State ----------
class MentorState(TypedDict):
    user_goal: str
    user_profile: Dict[str, Any]
    current_stress_level: int
    session_history: List[Dict[str, Any]]
    recommended_ids: List[str]
    adaptation_needed: bool

# ---------- Helper: simple rules ----------
def _pick_category(goal: str, stress: int) -> str:
    g = goal.lower()
    if "panic" in g or "anx" in g or stress >= 7:
        return "breathing"
    if "tight" in g or "tension" in g or "back" in g:
        return "bodyscan"
    return "meditations" if stress <= 5 else "breathing"

def _rank_ids(cat_items, stress: int) -> List[str]:
    # simple heuristic: shorter items first when stress is high
    ranked = sorted(cat_items, key=lambda t: (t.duration_min if stress >= 7 else -t.duration_min))
    return [t.id for t in ranked[:3]]

# ---------- Nodes ----------
def analyze_needs(state: MentorState) -> MentorState:
    # derive a category from goal + stress
    cat_id = _pick_category(state["user_goal"], state["current_stress_level"])
    state["user_profile"]["_cat_id"] = cat_id
    return state

def plan_recommendations(state: MentorState) -> MentorState:
    """
    Generate AI-powered mindfulness or breathing techniques based on
    the user's current stress level and goal, replacing static library logic.
    """
    try:
        user_goal = state["user_profile"].get("goal", "reduce stress")
        stress_level = state.get("current_stress_level", 5)
        profile = state["user_profile"]

        # 🔮 Call the AI agent to generate personalized recommendations
        result = run_mentor_cycle(
            user_goal=user_goal,
            stress_level=stress_level,
            profile=profile,
            history=state.get("session_history", []),
        )

        # 🧩 Store generated techniques directly into state
        state["techniques"] = result.get("techniques", [])
        state["summary"] = result.get("summary", "Stay mindful and consistent!")
        state["recommended_ids"] = [i for i, _ in enumerate(state["techniques"])]

    except Exception as e:
        print("⚠️ Error in plan_recommendations:", e)
        state["techniques"] = []
        state["recommended_ids"] = []
        state["summary"] = "AI recommendation generation failed."

    return state

def monitor_progress(state: MentorState) -> MentorState:
    hist = state.get("session_history", [])
    if not hist:
        state["adaptation_needed"] = False
        return state
    # look at last 3 sessions: completion + after-stress
    last3 = hist[-3:]
    completed = sum(1 for s in last3 if s.get("completed"))
    avg_after = sum(s.get("stress_after", 6) for s in last3) / len(last3)
    state["adaptation_needed"] = (completed < 2) or (avg_after >= 6.5)
    return state

def adapt_plan(state: MentorState) -> MentorState:
    # if struggling: prefer the shortest techniques (usually breathing)
    state["user_profile"]["_cat_id"] = "breathing"
    return state

def motivate(state: MentorState) -> MentorState:
    # no-LLM, short message pattern (you can surface this in UI)
    state["user_profile"]["_motivation"] = (
        "Nice work staying consistent. Even 2–3 minutes helps reset your nervous system."
    )
    return state

# ---------- Build graph ----------
def build_graph():
    g = StateGraph(MentorState)
    g.add_node("analyze", analyze_needs)
    g.add_node("plan", plan_recommendations)
    g.add_node("monitor", monitor_progress)
    g.add_node("adapt", adapt_plan)
    g.add_node("motivate", motivate)

    g.set_entry_point("analyze")
    g.add_edge("analyze", "plan")
    g.add_edge("plan", "monitor")

    def route_after_monitor(state: MentorState) -> str:
        return "adapt" if state.get("adaptation_needed") else "motivate"

    g.add_conditional_edges("monitor", route_after_monitor, {"adapt": "adapt", "motivate": "motivate"})
    g.add_edge("adapt", "plan")
    g.add_edge("motivate", END)

    return g.compile(checkpointer=MemorySaver())

_graph = None
def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

def run_mentor_cycle(user_goal: str, stress_level: int, profile: Dict[str, Any], history: List[Dict[str, Any]]):
    state: MentorState = {
        "user_goal": user_goal.strip() or "reduce daily stress",
        "user_profile": profile or {},
        "current_stress_level": int(stress_level),
        "session_history": history or [],
        "recommended_ids": [],
        "adaptation_needed": False,
    }
    final = get_graph().invoke(state, config={"configurable": {"thread_id": "local-user"}})
    return {
        "recommended_ids": final.get("recommended_ids", []),
        "motivation": final.get("user_profile", {}).get("_motivation", "")
    }
