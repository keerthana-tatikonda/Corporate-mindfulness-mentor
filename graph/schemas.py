from pydantic import BaseModel,Field
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
    confidence: Optional[float] = None
    confidence_note: Optional[str] = ""

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
    confidence: Optional[float] = None          # <— add
    confidence_note: Optional[str] = None 
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
    focus_for_today: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)

    # NEW for Interaction Design / confidence
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_note: Optional[str] = None

# --- Profile Personalization & Workload Adaptation ---

class UserProfile(BaseModel):
    work_schedule: str                   # e.g., "Mon–Fri, 9–6 with 1h commute"
    typical_stress_level: int            # 0–10
    preferences: Optional[List[str]] = []  # e.g., ["short sessions", "breathing", "no yoga"]
    constraints: Optional[List[str]] = []  # e.g., ["no audio", "shared desk"]

class PersonalizedPlanRequest(BaseModel):
    goal: Goal
    profile: UserProfile

class PersonalizedPlanResponse(BaseModel):
    goal: str
    summary: str
    activities: List[str]
    # NEW: confidence for profile personalization
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_note: Optional[str] = None

class WorkloadReport(BaseModel):
    date: str                            # ISO date "YYYY-MM-DD"
    meetings: int                        # number of meetings
    busy_hours: float                    # total hours in meetings/deep work blocks
    fatigue: Optional[str] = None        # "low" | "medium" | "high"
    blockers: Optional[List[str]] = []   # e.g., ["release", "oncall"]

class AdaptedPlanResponse(BaseModel):
    goal: str
    day_plan: List[str]                  # 3–5 micro-steps adapted to workload
    rationale: str
    # NEW: confidence for workload-based adaptation
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_note: Optional[str] = None




class CheckIn(BaseModel):
    mood: Optional[str] = Field(
        default=None,
        description="User mood label: calm, neutral, anxious, frustrated",
    )
    sleep_quality: Optional[str] = Field(
        default=None,
        description="Sleep quality: poor, ok, great",
    )
    energy: Optional[str] = Field(
        default=None,
        description="Energy level: low, medium, high",
    )
    workload: Optional[str] = Field(
        default=None,
        description="Workload level: light, normal, heavy",
    )
    notes: Optional[str] = Field(default=None, description="Free-text notes")


class CheckInAdjustment(BaseModel):
    summary: str = Field(..., description="Short AI summary for today")
    focus_for_today: List[str] = Field(
        default_factory=list,
        description="Concrete activities to focus on today",
    )
    risk_flags: List[str] = Field(
        default_factory=list,
        description="Any stress-risk warnings to highlight",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="0–1 confidence score from the model (optional)",
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description="Explanation of the confidence level",
    )


class StressAnalyticsResult(BaseModel):
    summary: str
    key_drivers: List[str] = []
    suggestions: List[str] = []
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_note: Optional[str] = None


class ProductivityInsightsResult(BaseModel):
    correlation_summary: str
    risk_windows: List[str] = []      # e.g. "Late evenings after 20:00", "Fridays"
    suggestions: List[str] = []
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_note: Optional[str] = None
