from pydantic import BaseModel
from typing import List, Optional

class Goal(BaseModel):
    goal_name: str
    duration_type: str  # "daily", "weekly", or "monthly"
    description: Optional[str] = None

class PlanRequest(BaseModel):
    goal: Goal

class PlanResponse(BaseModel):
    goal: str
    suggested_activities: List[str]
    ai_summary: str

class SubGoal(BaseModel):
    """One milestone in the decomposed plan."""
    title: str                 # e.g., "Identify stress triggers"
    timeframe: str             # e.g., "Week 1", "Month 2", "Day 1–3"
    activities: List[str] = [] # 2–4 concrete example actions

class DecomposedPlan(BaseModel):
    """Full decomposition of the user's goal into milestones."""
    goal: str
    duration_type: str               # "daily" | "weekly" | "monthly"
    subgoals: List[SubGoal]
    ai_summary: str
# --- Morning Check-In (additive) ---
from typing import Optional, List
from pydantic import BaseModel

class CheckIn(BaseModel):
    mood: Optional[str] = None            # "calm" | "neutral" | "anxious" | ...
    sleep_quality: Optional[str] = None   # "poor" | "ok" | "great"
    energy: Optional[str] = None          # "low" | "medium" | "high"
    workload: Optional[str] = None        # "light" | "normal" | "heavy"
    notes: Optional[str] = None

class DayAdjustment(BaseModel):
    summary: Optional[str] = None
    focus_for_today: Optional[List[str]] = None   # 3–5 short actions
    risk_flags: Optional[List[str]] = None        # e.g., ["poor sleep", "heavy workload"]
