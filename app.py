import os
import re
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import altair as alt
import math

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title="Corporate Mindfulness Mentor",
    page_icon="🧘",
    layout="centered",
)
from services.mentor_graph import run_mentor_conversation
# ──────────────────────────────────────────────────────────────
# Imports from your project
# ──────────────────────────────────────────────────────────────
from graph.graph import (
    run_goal_creation,
    run_goal_decomposition,
    run_personalized_goal,
    run_workload_adaptation,
    run_morning_checkin,
    run_motivation_message,
    run_hr_insights,
    run_stress_analytics, run_productivity_insights
)

from services.llm import MODEL, client
from services.storage import save_plan
from services.break_agent import auto_mindfulness_reminder
from services.checkin_storage import save_checkin, load_checkins
from services.productivity_storage import save_productivity, load_productivity



from graph.break_graph import run_break_workflow, run_llm_break_workflow
from graph.schemas import CheckIn, StressAnalyticsResult, ProductivityInsightsResult

from services.session import (
    init_session,
    get_state,
    next_step,
    stop_session,
)
from services.langgraph_agent import run_mentor_cycle

try:
    from services.storage import save_profile, load_profile
except Exception:
    save_profile = None
    load_profile = lambda *args, **kwargs: None



# ──────────────────────────────────────────────────────────────
# 🌐 Global AI Coaching Style Labels (Used Across All Stories)
# ──────────────────────────────────────────────────────────────
MODE_LABELS = {
    "passive": "Passive Observer – mostly reflective, very low autonomy.",
    "gentle": "Gentle Suggester – soft, optional suggestions (default).",
    "active": "Active Coach – concrete action steps, medium autonomy.",
    "directive": "Directive Guide – very structured plan, higher autonomy.",
}


# ──────────────────────────────────────────────────────────────
# Helper: AI confidence chip
# ──────────────────────────────────────────────────────────────
def render_confidence(provenance: str | None, confidence: float | None, key: str):
    """
    Display model confidence for a result (0–1) as a small chip + progress bar.
    Provenance is accepted for compatibility but not used here.
    """
    if confidence is None:
        return  # nothing to show if the model didn't provide it

    pct = int(max(0, min(100, round(confidence * 100))))

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.5rem;margin:.5rem 0 .25rem 0;">
            <span style="font-size:.85rem;padding:.15rem .5rem;border-radius:999px;background:#eef; color:#223;">
                AI Confidence
            </span>
            <span style="font-size:.8rem;color:#666;">{pct}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(pct, text="Model confidence")

def get_autonomy_system_prompt(base_prompt: str) -> str:
    """
    Modify system prompt based on user's selected autonomy level.
    
    Args:
        base_prompt: The original system prompt
        
    Returns:
        Modified prompt that reflects the user's autonomy preference
    """
    autonomy_level = st.session_state.get("autonomy_level", "Gentle Suggester")
    
    if autonomy_level == "Passive Observer":
        modifier = (
            "\n\nAUTONOMY SETTING: The user prefers minimal AI direction. "
            "Provide observations and options WITHOUT strong recommendations. "
            "Use phrases like 'you might consider', 'one option could be', 'some people find'. "
            "Never use imperative language. Let the user decide everything."
        )
    elif autonomy_level == "Gentle Suggester":
        modifier = (
            "\n\nAUTONOMY SETTING: The user prefers collaborative suggestions. "
            "Offer gentle recommendations while emphasizing user choice. "
            "Use phrases like 'I suggest', 'you could try', 'this might help'. "
            "Frame everything as suggestions they can accept or decline."
        )
    elif autonomy_level == "Active Coach":
        modifier = (
            "\n\nAUTONOMY SETTING: The user wants active coaching. "
            "Provide clear, confident recommendations with rationale. "
            "Use phrases like 'I recommend', 'you should', 'the best approach is'. "
            "Be directive but explain why each recommendation matters."
        )
    else:  # Directive Guide
        modifier = (
            "\n\nAUTONOMY SETTING: The user wants prescriptive guidance. "
            "Give specific, actionable instructions based on best practices. "
            "Use phrases like 'do this', 'start with', 'the most effective approach is'. "
            "Be confident and direct—tell them exactly what to do and when."
        )
    
    return base_prompt + modifier


def render_uncertainty(confidence: float | None, key: str, show_explanation: bool = True):
    """
    Display AI uncertainty meter (inverse of confidence) with color-coded warnings.

    Args:
        confidence: Model confidence (0-1), uncertainty = 1 - confidence
        key: Unique key for Streamlit widget
        show_explanation: Whether to show interpretive text below meter
    """
    if confidence is None:
        return

    # Calculate uncertainty (inverse of confidence)
    uncertainty = 1.0 - confidence
    uncertainty_pct = int(max(0, min(100, round(uncertainty * 100))))

    # Determine color and interpretation based on uncertainty level
    if uncertainty_pct >= 50:  # High uncertainty (low confidence)
        color = "#ff6b6b"  # Red
        bar_color = "#ff6b6b"
        level = "⚠️ High Uncertainty"
        interpretation = "Treat this as a suggestion, not a recommendation. More data needed for reliable insights."
    elif uncertainty_pct >= 20:  # Moderate uncertainty
        color = "#ffa500"  # Orange
        bar_color = "#ffa500"
        level = "⚡ Moderate Uncertainty"
        interpretation = "This guidance is helpful but not definitive. Use your judgment alongside AI suggestions."
    else:  # Low uncertainty (high confidence)
        color = "#51cf66"  # Green
        bar_color = "#51cf66"
        level = "✓ Low Uncertainty"
        interpretation = "AI is confident in this analysis. You can rely on these insights with trust."

    # Display uncertainty meter
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.5rem;margin:.5rem 0 .25rem 0;">
            <span style="font-size:.85rem;padding:.15rem .5rem;border-radius:999px;background:{color}20;color:{color};font-weight:500;">
                {level}
            </span>
            <span style="font-size:.8rem;color:#666;">{uncertainty_pct}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Custom colored progress bar using HTML/CSS
    st.markdown(
        f"""
        <div style="width:100%;background:#f0f0f0;border-radius:4px;height:8px;overflow:hidden;">
            <div style="width:{uncertainty_pct}%;background:{bar_color};height:100%;transition:width 0.3s ease;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show interpretation if enabled
    if show_explanation:
        st.caption(f"💡 {interpretation}")


# ──────────────────────────────────────────────────────────────
# Explainability and Reasoning Transparency
# ──────────────────────────────────────────────────────────────

def render_explainability_view(reasoning_steps: list, key: str):
    """
    Display AI reasoning process in an expandable view.
    Shows step-by-step how the AI reached its conclusion.
    """
    with st.expander("🔍 AI Reasoning Process (How we got here)", expanded=False):
        st.caption("This shows the AI's thought process step-by-step:")

        for i, step in enumerate(reasoning_steps, 1):
            st.markdown(f"**Step {i}: {step.get('title', 'Processing')}**")
            st.write(f"→ {step.get('description', 'No description')}")
            if step.get('confidence'):
                st.progress(step['confidence'], text=f"Confidence: {int(step['confidence']*100)}%")
            st.markdown("---")


def detect_vague_goal(goal_text: str) -> dict:
    """
    Detect if a goal is too vague and needs clarification.
    Returns suggestions for making it more specific.
    """
    vague_indicators = ['better', 'more', 'less', 'improve', 'reduce', 'increase', 'feel']
    missing_specifics = {
        'timeframe': not any(word in goal_text.lower() for word in ['daily', 'weekly', 'monthly', 'day', 'week', 'month']),
        'measurable': not any(char.isdigit() for char in goal_text),
        'action': len(goal_text.split()) < 5,
        'vague_words': any(word in goal_text.lower() for word in vague_indicators)
    }

    is_vague = sum(missing_specifics.values()) >= 2

    suggestions = []
    if missing_specifics['timeframe']:
        suggestions.append("Add a timeframe (daily, weekly, monthly)")
    if missing_specifics['measurable']:
        suggestions.append("Include a specific number or measurement")
    if missing_specifics['action']:
        suggestions.append("Describe the specific action you'll take")
    if missing_specifics['vague_words']:
        suggestions.append("Replace vague words with concrete outcomes")

    return {
        'is_vague': is_vague,
        'issues': missing_specifics,
        'suggestions': suggestions,
        'clarity_score': 1.0 - (sum(missing_specifics.values()) / len(missing_specifics))
    }


def render_goal_clarification(goal_text: str, key: str):
    """
    Prompt user to clarify vague goals before processing.
    """
    analysis = detect_vague_goal(goal_text)

    if analysis['is_vague']:
        st.warning("⚠️ Your goal could be more specific. This will help the AI give better recommendations.")

        clarity_pct = int(analysis['clarity_score'] * 100)
        st.metric("Goal Clarity", f"{clarity_pct}%", delta=f"{100-clarity_pct}% to perfect clarity")

        with st.expander("💡 How to make this goal clearer", expanded=True):
            st.markdown("**Your goal:** " + goal_text)
            st.markdown("**Suggestions to improve:**")
            for suggestion in analysis['suggestions']:
                st.write(f"• {suggestion}")

            st.markdown("**Example transformation:**")
            st.info(
                f"**Vague:** 'Feel less stressed'\n\n"
                f"**Specific:** 'Practice 10-minute meditation daily for 30 days to reduce work stress'"
            )

            refine = st.text_area(
                "Refine your goal (optional):",
                value=goal_text,
                key=f"{key}_refined",
                help="Make it more specific using the suggestions above"
            )

            if st.button("Use refined goal", key=f"{key}_use_refined"):
                st.success("✅ Goal updated!")
                return refine

    return goal_text


# ──────────────────────────────────────────────────────────────
# Feedback System
# ──────────────────────────────────────────────────────────────

def save_feedback(feedback_type: str, rating: int | None, comment: str | None, metadata: dict = None):
    """Save user feedback to JSON file for analysis."""
    from datetime import datetime
    import os

    FEEDBACK_DIR = "data"
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "user_feedback.json")

    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": feedback_type,
        "rating": rating,
        "comment": comment,
        "metadata": metadata or {}
    }

    # Load existing feedback
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                feedback_data = json.load(f)
        else:
            feedback_data = []
    except Exception:
        feedback_data = []

    # Append new feedback
    feedback_data.append(feedback_entry)

    # Save back
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_data, f, indent=2)


def render_feedback_widget(
    key: str,
    title: str = "Was this helpful?",
    feedback_type: str = "general",
    show_comment: bool = True,
    metadata: dict = None,
    compact: bool = False
):
    """
    Render an interactive feedback widget for user input.

    Args:
        key: Unique key for widget
        title: Title/question to display
        feedback_type: Type of feedback (for categorization)
        show_comment: Whether to show optional comment field
        metadata: Additional context to save with feedback
        compact: Use compact layout (single row)
    """

    if compact:
        # Compact layout: emoji buttons in a row
        st.markdown(f"<p style='font-size:0.85rem;color:#666;margin:.25rem 0;'>{title}</p>", unsafe_allow_html=True)

        cols = st.columns([1, 1, 1, 1, 1, 3])

        feedback_given = False
        with cols[0]:
            if st.button("😍", key=f"{key}_love", help="Loved it!"):
                save_feedback(feedback_type, 5, "Loved it", metadata)
                st.success("Thanks!")
                feedback_given = True
        with cols[1]:
            if st.button("😊", key=f"{key}_good", help="Helpful"):
                save_feedback(feedback_type, 4, "Helpful", metadata)
                st.success("Thanks!")
                feedback_given = True
        with cols[2]:
            if st.button("😐", key=f"{key}_ok", help="Okay"):
                save_feedback(feedback_type, 3, "Okay", metadata)
                st.success("Thanks!")
                feedback_given = True
        with cols[3]:
            if st.button("😕", key=f"{key}_meh", help="Not very helpful"):
                save_feedback(feedback_type, 2, "Not helpful", metadata)
                st.success("Thanks!")
                feedback_given = True
        with cols[4]:
            if st.button("😞", key=f"{key}_bad", help="Not helpful at all"):
                save_feedback(feedback_type, 1, "Not helpful at all", metadata)
                st.success("Thanks!")
                feedback_given = True

    else:
        # Full layout with expander
        with st.expander(f"💬 {title}", expanded=False):
            # Star rating
            rating = st.select_slider(
                "Rate this feature:",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x,
                key=f"{key}_rating"
            )

            # Optional comment
            if show_comment:
                comment = st.text_area(
                    "Any additional thoughts? (optional)",
                    placeholder="What worked well? What could be improved?",
                    key=f"{key}_comment",
                    max_chars=500
                )
            else:
                comment = None

            # Submit button
            if st.button("Submit Feedback", key=f"{key}_submit"):
                save_feedback(feedback_type, rating, comment, metadata)
                st.success("✅ Thank you for your feedback! Your input helps us improve.")


def render_confidence_accuracy_check(key: str, confidence: float, feature_name: str):
    """
    Ask users if the AI's confidence level matched reality.
    This helps calibrate confidence scoring over time.
    """

    with st.expander("🎯 Confidence Check: Was the AI's confidence accurate?", expanded=False):
        st.caption(f"The AI was {int(confidence*100)}% confident in this {feature_name}.")

        accuracy = st.radio(
            "After seeing the results, how would you rate the AI's confidence?",
            options=[
                "Too confident (overestimated)",
                "Just right (well-calibrated)",
                "Too cautious (underestimated)"
            ],
            key=f"{key}_accuracy"
        )

        action_taken = st.checkbox(
            f"I followed this {feature_name}'s advice",
            key=f"{key}_action"
        )

        helpful_rating = st.slider(
            "How helpful was this overall?",
            1, 5, 3,
            format="%d ⭐",
            key=f"{key}_helpful"
        )

        if st.button("Submit", key=f"{key}_submit_accuracy"):
            metadata = {
                "feature": feature_name,
                "ai_confidence": confidence,
                "user_accuracy_rating": accuracy,
                "action_taken": action_taken,
                "helpful_rating": helpful_rating
            }
            save_feedback("confidence_calibration", helpful_rating, accuracy, metadata)
            st.success("✅ Thanks! Your input helps calibrate our AI.")


def render_action_tracker(key: str, recommendation: str, feature_name: str):
    """
    Track whether users actually followed AI recommendations.
    This measures real-world impact.
    """

    st.markdown("---")
    st.markdown("#### 🎯 Track Your Action")

    cols = st.columns([2, 1])
    with cols[0]:
        st.caption(f"Planning to follow this {feature_name}?")

    with cols[1]:
        if st.button("✅ Yes, I'll try this", key=f"{key}_yes"):
            save_feedback(
                "action_commitment",
                5,
                f"Committed to: {recommendation[:100]}",
                {"feature": feature_name, "committed": True}
            )
            st.success("Great! Check back to report results.")

        if st.button("❌ No, not for me", key=f"{key}_no"):
            save_feedback(
                "action_commitment",
                1,
                f"Declined: {recommendation[:100]}",
                {"feature": feature_name, "committed": False}
            )
            st.info("That's okay! Not every suggestion fits everyone.")





# ──────────────────────────────────────────────────────────────
# Helpers for Stress-Level & Productivity dashboards
# ──────────────────────────────────────────────────────────────
def _score_stress_from_checkin(entry: dict) -> float:
    """
    Turn one saved check-in entry into a numeric stress score (0–100)
    using mood, sleep quality, energy, and workload.
    """
    c = (entry or {}).get("checkin") or {}

    mood_map = {
        "calm": 0.1,
        "neutral": 0.4,
        "anxious": 0.7,
        "frustrated": 1.0,
    }
    sleep_map = {
        "great": 0.1,
        "ok": 0.4,
        "poor": 0.8,
    }
    energy_map = {
        "high": 0.1,
        "medium": 0.4,
        "low": 0.8,
    }
    workload_map = {
        "light": 0.2,
        "normal": 0.5,
        "heavy": 0.9,
    }

    mood = c.get("mood") or ""
    sleep = c.get("sleep_quality") or ""
    energy = c.get("energy") or ""
    workload = c.get("workload") or ""

    m = mood_map.get(mood, 0.5)
    s = sleep_map.get(sleep, 0.4)
    e = energy_map.get(energy, 0.4)
    w = workload_map.get(workload, 0.5)

    raw = (m + s + e + w) / 4.0
    return round(raw * 100, 1)


def build_stress_dataframe(raw_checkins: list) -> pd.DataFrame:
    """
    Build a DataFrame with:
      - date: datetime64
      - stress_score: float (0–100)
      - mood, sleep_quality, energy, workload
    """
    rows = []
    for item in raw_checkins:
        c = (item or {}).get("checkin") or {}
        ts = item.get("saved_at")

        try:
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts)
            else:
                dt = datetime.fromisoformat(str(ts))
        except Exception:
            dt = datetime.utcnow()

        rows.append(
            {
                "date": dt.date(),
                "stress_score": _score_stress_from_checkin(item),
                "mood": c.get("mood") or "unknown",
                "sleep_quality": c.get("sleep_quality") or "unknown",
                "energy": c.get("energy") or "unknown",
                "workload": c.get("workload") or "unknown",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "stress_score",
                "mood",
                "sleep_quality",
                "energy",
                "workload",
            ]
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["week_start"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time.date())
    return df


def build_productivity_dataframe(raw_entries: list) -> pd.DataFrame:
    """
    Build a DataFrame with:
      - date: datetime64
      - productivity: float (0–10)
      - prod_notes: notes for tooltip context
    """
    rows = []
    for item in raw_entries:
        date_str = str(item.get("date") or "")
        notes = item.get("notes") or ""
        ts = item.get("saved_at")

        try:
            if date_str:
                dt = datetime.fromisoformat(date_str)
            elif isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts)
            else:
                dt = datetime.utcnow()
        except Exception:
            dt = datetime.utcnow()

        rows.append(
            {
                "date": dt.date(),
                "productivity": float(item.get("productivity") or 0.0),
                "prod_notes": notes,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["date", "productivity", "prod_notes"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def build_stress_productivity_join(
    df_stress: pd.DataFrame, df_prod: pd.DataFrame
) -> pd.DataFrame:
    """
    Inner-join stress and productivity on date, so we only keep days
    where both were recorded.
    """
    if df_stress.empty or df_prod.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "stress_score",
                "productivity",
                "mood",
                "sleep_quality",
                "energy",
                "workload",
                "prod_notes",
            ]
        )

    df_s = df_stress.copy()
    df_p = df_prod.copy()
    df_s["date"] = df_s["date"].dt.date
    df_p["date"] = df_p["date"].dt.date

    joined = pd.merge(df_s, df_p, on="date", how="inner")
    joined["date"] = pd.to_datetime(joined["date"])
    return joined


def _compute_confidence_from_days(num_days: int):
    """
    Simple heuristic confidence based on how many days of data the user has.
    Returns (confidence_0_to_1, note).
    """
    if num_days >= 30:
        c = 0.95
        note = "High confidence – about a month of data."
    elif num_days >= 14:
        c = 0.85
        note = "Good confidence – around two weeks of patterns."
    elif num_days >= 7:
        c = 0.7
        note = "Moderate confidence – about a week of data."
    elif num_days >= 3:
        c = 0.5
        note = "Low confidence – only a few days logged."
    elif num_days >= 1:
        c = 0.3
        note = "Very low confidence – insights are based on a single day."
    else:
        c = 0.0
        note = "No data yet."
    return c, note


# Readable card style for plan summary (works in light/dark)
st.markdown(
    """
<style>
.plan-card {
  padding: 18px 20px;
  border-radius: 12px;
  border: 1px solid rgba(148,163,184,.35);
  background: rgba(2, 6, 23, 0.35);
  color: #e5e7eb;
  line-height: 1.7;
}
@media (prefers-color-scheme: light) {
  .plan-card {
    background: #f8fafc;
    color: #0f172a;
    border-color: rgba(15,23,42,.15);
  }
}
.plan-card p { margin: 0 0 .7rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Sidebar settings for mindful break notifications
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧘 Mindful Break Settings")
    enable_auto_breaks_sidebar = st.checkbox(
        "Enable automatic reminders", value=True, key="auto_breaks_toggle"
    )
    play_sound_sidebar = st.checkbox(
        "🔔 Sound alert", value=True, key="sound_alert_toggle"
    )

    st.markdown("---")
    st.markdown("### 🎚️ AI Coaching Style")
    # Initialize default value before widget creation
    if "autonomy_level" not in st.session_state:
        st.session_state["autonomy_level"] = "Gentle Suggester"

    autonomy_level = st.radio(
        "How directive should the AI be?",
        options=["Passive Observer", "Gentle Suggester", "Active Coach", "Directive Guide"],
        index=1,
        help="Controls how assertive the AI's recommendations are. Passive = minimal suggestions, Directive = proactive guidance",
        key="autonomy_level"
    )

# ──────────────────────────────────────────────────────────────
# Session State Init
# ──────────────────────────────────────────────────────────────
def init_session_state():
    if "result" not in st.session_state:
        st.session_state["result"] = None
    if "goal_history" not in st.session_state:
        st.session_state["goal_history"] = []
    if "saved_plans" not in st.session_state:
        st.session_state["saved_plans"] = {}
    if "current_goal_id" not in st.session_state:
        st.session_state["current_goal_id"] = None
    if "last_input" not in st.session_state:
        st.session_state["last_input"] = None
    if "decomposition" not in st.session_state:
        st.session_state["decomposition"] = None
    if "task_feedback" not in st.session_state:
        st.session_state["task_feedback"] = {}
    if "task_completion" not in st.session_state:
        st.session_state["task_completion"] = {}


init_session_state()

# ──────────────────────────────────────────────────────────────
# Persistence Helpers for Plans
# ──────────────────────────────────────────────────────────────
def save_plan_to_history(goal_name: str, duration_type: str, result):
    """Save a plan to the session history and return its ID."""
    goal_id = f"goal_{len(st.session_state['goal_history']) + 1}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    plan_data = {
        "id": goal_id,
        "name": goal_name,
        "duration": duration_type,
        "timestamp": timestamp,
        "activities": result.suggested_activities,
        "summary": result.ai_summary,
    }

    st.session_state["goal_history"].append(plan_data)
    st.session_state["saved_plans"][goal_id] = plan_data
    st.session_state["current_goal_id"] = goal_id
    return goal_id


def load_plan_from_history(goal_id: str):
    """Load a plan from session history."""
    if goal_id in st.session_state["saved_plans"]:
        plan = st.session_state["saved_plans"][goal_id]
        from graph.schemas import PlanResponse

        return PlanResponse(
            goal=plan["name"],
            suggested_activities=plan["activities"],
            ai_summary=plan["summary"],
        )
    return None


def export_all_plans_json():
    """Export all saved plans as JSON."""
    return json.dumps(st.session_state["saved_plans"], indent=2)


def export_plan_markdown(plan_data):
    """Export a single plan to Markdown."""
    md = f"""# Mindfulness Plan: {plan_data['name']}

**Duration:** {plan_data['duration']}  
**Created:** {plan_data['timestamp']}

## 🎯 Activities

"""
    for i, activity in enumerate(plan_data["activities"], 1):
        md += f"{i}. {activity}\n"

    md += f"\n## 📝 Summary\n\n{plan_data['summary']}\n"
    return md


# ──────────────────────────────────────────────────────────────
# UI: Title & Sidebar (Goal Creation FIRST)
# ──────────────────────────────────────────────────────────────
st.title("🧘‍♀️ Corporate Mindfulness Mentor")
st.subheader("Goal Creation & Personalized Plan")

with st.sidebar:
    st.markdown("### 📊 Your Progress")
    total_plans = len(st.session_state["goal_history"])
    st.metric("Total Plans Created", total_plans)

    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ API Key Missing")
        st.caption("Please add OPENAI_API_KEY to your .env file")

    st.markdown("---")

    if st.session_state["goal_history"]:
        st.markdown("### 📚 Previous Goals")

        for plan in reversed(st.session_state["goal_history"][-10:]):
            with st.expander(f"📋 {plan['name'][:35]}...", expanded=False):
                st.caption(f"Created: {plan['timestamp']}")
                st.caption(f"Duration: {plan['duration'].title()}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "📖 Load",
                        key=f"load_{plan['id']}",
                        use_container_width=True,
                    ):
                        loaded_plan = load_plan_from_history(plan["id"])
                        if loaded_plan:
                            st.session_state["result"] = loaded_plan
                            st.session_state["current_goal_id"] = plan["id"]
                            st.session_state["decomposition"] = None
                            st.rerun()

                with col2:
                    md_content = export_plan_markdown(plan)
                    st.download_button(
                        "💾 Save",
                        data=md_content,
                        file_name=f"{plan['name'][:20].replace(' ', '_')}.md",
                        mime="text/markdown",
                        key=f"dl_{plan['id']}",
                        use_container_width=True,
                    )

        st.markdown("---")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.caption("Export all your plans")
        with col2:
            json_data = export_all_plans_json()
            st.download_button(
                "📦 Export",
                data=json_data,
                file_name=f"all_plans_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True,
            )

    st.markdown("---")

    if st.button(
        "🔄 Start Fresh",
        help="Clear all saved plans",
        use_container_width=True,
    ):
        if st.session_state["goal_history"]:
            st.session_state.clear()
            st.rerun()

# ──────────────────────────────────────────────────────────────
# Main Form – Goal Creation
# ──────────────────────────────────────────────────────────────
with st.form("goal_form", clear_on_submit=False):
    goal_name = st.text_input(
        "What's your mindfulness goal?",
        value="",
        placeholder="e.g., Reduce daily stress, Improve focus, Manage anxiety",
        help="Be specific about what you want to achieve",
    )

    DURATION_CHOICES = ["— select frequency —", "daily", "weekly", "monthly"]
    duration_choice = st.selectbox(
        "How often will you practice?",
        options=DURATION_CHOICES,
        index=0,
        help="Choose how frequently you want to practice these activities",
    )
    duration_type = None if duration_choice == DURATION_CHOICES[0] else duration_choice

    description = st.text_area(
        "Additional context (optional)",
        value="",
        placeholder=(
            "Share details about your schedule, work environment, or "
            "specific challenges..."
        ),
        help="More context helps us create a better personalized plan",
        height=100,
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        submitted = st.form_submit_button(
            "✨ Generate My Plan",
            use_container_width=True,
            type="primary",
        )
    with col_btn2:
        decompose_request = st.form_submit_button(
            "📅 Break Into Steps",
            use_container_width=True,
        )

# ──────────────────────────────────────────────────────────────
# Handle Form Actions
# ──────────────────────────────────────────────────────────────
if submitted:
    goal_ok = bool(goal_name.strip())
    duration_ok = duration_type is not None

    if not goal_ok or not duration_ok:
        if not goal_ok:
            st.warning("💭 Please tell us your mindfulness goal")
        if not duration_ok:
            st.warning("📅 Please select how often you'll practice")
    else:
        # Check goal clarity and prompt for refinement if needed
        render_goal_clarification(goal_name.strip(), key="goal_clarity_check")
        try:
            with st.spinner("✨ Creating your personalized mindfulness plan..."):
                result = run_goal_creation(
                    goal_name.strip(),
                    duration_type,
                    description.strip(),
                )

            st.session_state["result"] = result

            st.session_state["last_input"] = {
                "goal_name": goal_name.strip(),
                "duration_type": duration_type,
                "description": description.strip(),
            }
            st.session_state["decomposition"] = None

            save_plan_to_history(goal_name.strip(), duration_type, result)

            try:
                save_plan(result.model_dump())
            except Exception:
                pass

            st.success("✅ Your personalized plan is ready!")
            st.balloons()

        except Exception as e:
            st.error("😔 Oops! Something went wrong creating your plan.")
            st.caption(f"Error details: {str(e)}")

if decompose_request:
    if st.session_state.get("result") is None:
        st.info("💡 First, generate a plan, then you can break it into weekly steps!")
    else:
        li = st.session_state.get("last_input") or {}
        try:
            cadence = li.get("duration_type", "weekly")
            with st.spinner(f"🔄 Breaking your goal into {cadence} milestones..."):
                dec = run_goal_decomposition(
                    goal_name=li.get("goal_name", st.session_state["result"].goal),
                    duration_type=cadence,
                    description=li.get("description", ""),
                )
            st.session_state["decomposition"] = dec
            st.success(f"✅ Your {cadence.title()} roadmap is ready!")
        except Exception as e:
            st.error("😔 Couldn't create weekly breakdown.")
            st.caption(f"Error: {str(e)}")

# ──────────────────────────────────────────────────────────────
# Display Decomposition (Milestones)
# ──────────────────────────────────────────────────────────────
dec = st.session_state.get("decomposition")
if dec:
    st.markdown("---")
    title_cadence = (dec.duration_type or "weekly").title()
    st.markdown(f"## 📅 Your {title_cadence} Roadmap")
    st.caption("Here's how to achieve your goal step by step")

    for idx, sg in enumerate(dec.subgoals, start=1):
        title = getattr(sg, "title", "This Week's Focus")
        timeframe = getattr(sg, "timeframe", f"Week {idx}")
        activities = getattr(sg, "activities", []) or []

        with st.expander(f"**{timeframe}: {title}**", expanded=(idx == 1)):
            st.markdown("**Activities for this period:**")
            for i, activity in enumerate(activities, 1):
                st.markdown(f"{i}. {activity}")

    if hasattr(dec, "ai_summary") and dec.ai_summary:
        st.markdown("---")
        st.markdown("### 🌱 Why This Sequence?")
        st.info(dec.ai_summary)

# ✅ ADDITION: Show both confidence and uncertainty
    dec_conf = getattr(dec, "confidence", None)
    cols = st.columns(2)
    with cols[0]:
        render_confidence(None, dec_conf, key="decomposition_conf")
    with cols[1]:
        render_uncertainty(dec_conf, key="decomposition_uncertainty", show_explanation=False)
    
    note = getattr(dec, "confidence_note", None)
    if note:
        st.caption(f"📊 {note}")
    
    # ✅ ADDITION: Explainability for decomposition
    reasoning_steps = [
        {
            "title": "Milestone Planning",
            "description": f"Broke your goal into {len(dec.subgoals)} progressive milestones",
            "confidence": dec_conf if dec_conf else 0.8
        },
        {
            "title": "Activity Distribution",
            "description": "Assigned specific activities to each timeframe based on skill progression",
            "confidence": dec_conf if dec_conf else 0.75
        }
    ]
    render_explainability_view(reasoning_steps, key="decomposition_reasoning")

# ──────────────────────────────────────────────────────────────
# Display Main Plan (Activities & Summary)
# ──────────────────────────────────────────────────────────────
result = st.session_state.get("result")

if result:
    st.markdown("---")

    st.markdown(f"## 🎯 Your Goal: {result.goal}")
    
    # ✅ ADDITION: Show both confidence and uncertainty side by side
    confidence_val = getattr(result, "confidence", None)
    cols = st.columns(2)
    with cols[0]:
        render_confidence(None, confidence_val, key="plan_conf")
    with cols[1]:
        render_uncertainty(confidence_val, key="plan_uncertainty", show_explanation=False)
    
    plan_conf_note = getattr(result, "confidence_note", None)
    if plan_conf_note:
        st.caption(f"📊 {plan_conf_note}")
    
    # ✅ ADDITION: Explainability view showing reasoning
    num_activities = len(result.suggested_activities)
    reasoning_steps = [
        {
            "title": "Goal Analysis",
            "description": f"Analyzed your goal '{result.goal}' and context to identify focus areas",
            "confidence": 0.9
        },
        {
            "title": "Activity Selection",
            "description": f"Selected {num_activities} evidence-based activities matching your needs",
            "confidence": confidence_val if confidence_val else 0.8
        },
        {
            "title": "Personalization",
            "description": "Tailored recommendations to fit corporate work environment",
            "confidence": confidence_val if confidence_val else 0.75
        }
    ]
    render_explainability_view(reasoning_steps, key="goal_creation_reasoning")

    current_id = st.session_state.get("current_goal_id")
    if current_id and current_id in st.session_state["saved_plans"]:
        st.caption(f"Saved as: {current_id}")

    st.markdown("### 📋 Your Daily Practices")
    st.caption("Check off activities as you complete them")

    completed_flags = {}
    total = len(result.suggested_activities)
    done = 0

    for i, act in enumerate(result.suggested_activities, start=1):
        cb_key = (
            f"activity_{current_id}_{i}" if current_id else f"activity_temp_{i}"
        )
        finished = bool(st.session_state.get(cb_key, False))
        st.checkbox(f"**{i}.** {act}", key=cb_key)
        completed_flags[act] = finished
        if finished:
            done += 1

    st.session_state["task_completion"] = completed_flags

    if total > 0:
        st.info(f"✅ You’ve completed {done} of {total} tasks for this plan.")

    # Edit tasks & feedback
    st.markdown("### ✏️ Edit Your Tasks & Give Feedback")
    st.caption(
        "You can rewrite any task and mark whether it’s working for you. "
        "The updated tasks will be used in later personalization and adaptation."
    )

    edited_activities = []
    feedback_flags = {}

    for i, act in enumerate(result.suggested_activities, start=1):
        col_e1, col_e2 = st.columns([3, 1])
        with col_e1:
            new_text = st.text_input(
                f"Task {i}",
                value=act,
                key=f"edit_task_{i}",
            )
        with col_e2:
            feedback = st.selectbox(
                "Feedback",
                ["no feedback", "helpful", "not helpful"],
                index=0,
                key=f"fb_task_{i}",
            )
        edited_activities.append(new_text)
        feedback_flags[new_text] = feedback

    if st.button("💾 Save edits & update plan"):
        result.suggested_activities = edited_activities
        st.session_state["result"] = result

        if "personalized" in st.session_state and st.session_state["personalized"]:
            try:
                st.session_state["personalized"].activities = edited_activities
            except Exception:
                pass

        current_id = st.session_state.get("current_goal_id")
        if current_id and current_id in st.session_state["saved_plans"]:
            st.session_state["saved_plans"][current_id]["activities"] = edited_activities

        st.session_state["task_feedback"] = feedback_flags

        st.success("✅ Tasks updated. Future adaptations will use your edited tasks.")

    st.markdown("---")
    st.markdown("### 💡 About Your Plan")
    clean_summary = re.sub(r"#+\s*", "", result.ai_summary or "").strip()
    html_summary = clean_summary.replace("\n", "<br>")
    st.markdown(
        f"<div class='plan-card'>{html_summary}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🆕 New Plan", use_container_width=True):
            st.session_state["result"] = None
            st.session_state["current_goal_id"] = None
            st.session_state["decomposition"] = None
            st.rerun()

    with col2:
        plan_text = f"""Goal: {result.goal}

Your Daily Practices:
{chr(10).join(f'{i}. {act}' for i, act in enumerate(result.suggested_activities, 1))}

About This Plan:
{clean_summary}

---
Created with Corporate Mindfulness Mentor
"""
        st.download_button(
            label="📥 Download",
            data=plan_text,
            file_name=f"{result.goal[:30].replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col3:
        if current_id and current_id in st.session_state["saved_plans"]:
            plan_data = st.session_state["saved_plans"][current_id]
            md_content = export_plan_markdown(plan_data)
            st.download_button(
                label="📄 Export MD",
                data=md_content,
                file_name=f"{result.goal[:30].replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

# ──────────────────────────────────────────────────────────────
# 👤 Profile Personalization
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 👤 Profile Personalization")

with st.form("profile_form", clear_on_submit=False):
    colA, colB = st.columns(2)
    with colA:
        work_schedule = st.text_input(
            "Work schedule", placeholder="Mon–Fri, 9–6; commute 30m"
        )
        typical_stress = st.slider("Typical stress (0–10)", 0, 10, 5)
    with colB:
        prefs = st.text_input(
            "Preferences (comma-separated)", placeholder="short sessions, breathing"
        )
        cons = st.text_input(
            "Constraints (comma-separated)", placeholder="no audio, shared desk"
        )

    personalize_submit = st.form_submit_button("🎯 Personalize my plan")

if personalize_submit:
    if st.session_state.get("result") is None:
        st.info("Generate a goal plan first, then personalize it.")
    else:
        from graph.schemas import Goal, UserProfile

        g = Goal(
            goal_name=st.session_state["result"].goal,
            duration_type=st.session_state.get("last_input", {}).get(
                "duration_type", "weekly"
            ),
            description=st.session_state.get("last_input", {}).get(
                "description", ""
            ),
        )
        profile = UserProfile(
            work_schedule=work_schedule.strip() or "Mon–Fri",
            typical_stress_level=int(typical_stress),
            preferences=[
                p.strip() for p in (prefs or "").split(",") if p.strip()
            ],
            constraints=[
                c.strip() for c in (cons or "").split(",") if c.strip()
            ],
        )
        try:
            task_feedback = st.session_state.get("task_feedback") or {}
            task_completion = st.session_state.get("task_completion") or {}

            with st.spinner("Tailoring your plan to your schedule..."):
                p = run_personalized_goal(
                    g,
                    profile,
                    task_feedback=task_feedback,
                    completion=task_completion,
                )
            st.success("✅ Personalized plan created")
            st.markdown("### 🎯 Personalized Activities")
            for i, a in enumerate(p.activities, 1):
                st.markdown(f"{i}. {a}")
            st.markdown("### 📝 Why this fits you")
            st.info(p.summary)
            st.session_state["personalized"] = p
        # ✅ ADDITION: Show both confidence and uncertainty
            p_conf = getattr(p, "confidence", None)
            cols = st.columns(2)
            with cols[0]:
                render_confidence(None, p_conf, key="profile_conf")
            with cols[1]:
                render_uncertainty(p_conf, key="profile_uncertainty", show_explanation=False)
            
            note = getattr(p, "confidence_note", None)
            if note:
                st.caption(f"📊 {note}")
            
            # ✅ ADDITION: Explainability for personalization
            reasoning_steps = [
                {
                    "title": "Schedule Analysis",
                    "description": f"Analyzed your work schedule: {profile.work_schedule}",
                    "confidence": 0.95
                },
                {
                    "title": "Preference Matching",
                    "description": f"Incorporated {len(profile.preferences or [])} preferences and {len(profile.constraints or [])} constraints",
                    "confidence": 0.9
                },
                {
                    "title": "Timing Optimization",
                    "description": f"Scheduled activities around stress level {profile.typical_stress_level}/10",
                    "confidence": p_conf if p_conf else 0.8
                }
            ]
            render_explainability_view(reasoning_steps, key="personalization_reasoning")
            
            # ✅ ADDITION: Trust caption
            st.markdown("#### 🔒 Trust and Predictability by Design")
            st.caption(
                "Personalization uses your work schedule, stress level, preferences, and constraints. "
                "Activities are timed to fit your actual availability. Higher confidence means better "
                "alignment with your profile."
            )
        except Exception as e:
            st.error(f"Could not personalize plan: {e}")

# ──────────────────────────────────────────────────────────────
# ⚙️ Workload-Based Adaptation
# ──────────────────────────────────────────────────────────────
st.markdown("## ⚙️ Workload-Based Adaptation (Today)")

with st.form("workload_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        w_date = st.date_input("Date")
        meetings = st.number_input(
            "Meetings (count)", min_value=0, step=1, value=4
        )
    with col2:
        busy_hours = st.number_input(
            "Busy hours (today)", min_value=0.0, step=0.5, value=4.0
        )
        fatigue = st.selectbox(
            "Fatigue", ["", "low", "medium", "high"], index=2
        )
    with col3:
        blockers = st.text_input(
            "Blockers (comma-separated)", placeholder="oncall, release"
        )
    adapt_submit = st.form_submit_button("🔧 Adapt today’s plan")

if adapt_submit:
    base = st.session_state.get("personalized") or st.session_state.get("result")
    base_acts = []
    if base:
        base_acts = (
            getattr(base, "activities", None)
            or getattr(base, "suggested_activities", None)
            or []
        )

    if not base_acts:
        st.info("Create or personalize a plan first, then adapt it.")
    else:
        from graph.schemas import Goal, WorkloadReport

        g = Goal(
            goal_name=(
                getattr(base, "goal", None)
                or st.session_state["result"].goal
            ),
            duration_type=st.session_state.get("last_input", {}).get(
                "duration_type", "weekly"
            ),
            description=st.session_state.get("last_input", {}).get(
                "description", ""
            ),
        )
        wl = WorkloadReport(
            date=w_date.isoformat(),
            meetings=int(meetings),
            busy_hours=float(busy_hours),
            fatigue=(fatigue or None),
            blockers=[
                b.strip() for b in (blockers or "").split(",") if b.strip()
            ],
        )
        try:
            task_feedback = st.session_state.get("task_feedback") or {}
            task_completion = st.session_state.get("task_completion") or {}

            with st.spinner("Right-sizing today’s steps..."):
                adapted = run_workload_adaptation(
                    g,
                    base_acts,
                    wl,
                    task_feedback=task_feedback,
                    completion=task_completion,
                )

            st.success("✅ Adapted plan for today")
            st.markdown("### 📋 Today’s Micro-Plan")
            for i, a in enumerate(adapted.day_plan, 1):
                st.markdown(f"{i}. {a}")
            st.markdown("### 💡 Rationale")
            st.caption(adapted.rationale)
           # ✅ ADDITION: Show both confidence and uncertainty
            adapt_conf = getattr(adapted, "confidence", None)
            cols = st.columns(2)
            with cols[0]:
                render_confidence(None, adapt_conf, key="adapt_conf")
            with cols[1]:
                render_uncertainty(adapt_conf, key="adapt_uncertainty", show_explanation=False)
            
            note = getattr(adapted, "confidence_note", None)
            if note:
                st.caption(f"📊 {note}")
            
            # ✅ ADDITION: Explainability for adaptation
            reasoning_steps = [
                {
                    "title": "Workload Assessment",
                    "description": f"Today: {wl.meetings} meetings, {wl.busy_hours} busy hours, fatigue: {wl.fatigue or 'not reported'}",
                    "confidence": 0.95
                },
                {
                    "title": "Activity Scaling",
                    "description": f"Reduced plan to {len(adapted.day_plan)} manageable micro-tasks",
                    "confidence": 0.9
                },
                {
                    "title": "Timing Adjustment",
                    "description": "Scheduled activities during lower-demand periods today",
                    "confidence": adapt_conf if adapt_conf else 0.8
                }
            ]
            render_explainability_view(reasoning_steps, key="adaptation_reasoning")
            
            # ✅ ADDITION: Trust caption
            st.markdown("#### 🔒 Trust and Predictability by Design")
            st.caption(
                "Workload adaptation uses today's meeting count, busy hours, and fatigue level "
                "to right-size your plan. When workload is heavy, fewer activities are suggested. "
                "Lower confidence suggests unusual workload patterns."
            )
        except Exception as e:
            st.error(f"Could not adapt today’s plan: {e}")

# ──────────────────────────────────────────────────────────────
# 🧘 Mindfulness Break Notifications
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🕒 Mindfulness Break Notifications")

if st.button("🌼 Take a Mindful Break", key="manual_break"):
    run_break_workflow()
    st.balloons()
    st.session_state["last_reminder_time"] = datetime.now()
    st.info("✅ Mindful break recorded successfully.")

if st.button("🤖 AI-Powered Mindful Break"):
    llm_output = run_llm_break_workflow() or {}

    st.markdown("### 🌼 AI-Powered Mindful Suggestion")

    # Main suggestion
    if llm_output.get("message"):
        st.info(llm_output["message"])

    if llm_output.get("recommendation"):
        st.success(llm_output["recommendation"])

    # 🔍 Confidence bar (interaction design: visual cue)
# ✅ ADDITION: Show both confidence and uncertainty
    break_conf = llm_output.get("confidence")
    cols = st.columns(2)
    with cols[0]:
        render_confidence(None, break_conf, key="break_confidence")
    with cols[1]:
        render_uncertainty(break_conf, key="break_uncertainty", show_explanation=False)
    
    if llm_output.get("confidence_note"):
        st.caption(f"📊 {llm_output['confidence_note']}")
    
    # ✅ ADDITION: Explainability view
    reasoning_steps = [
        {
            "title": "Break Reminder Selection",
            "description": "Chose a brief mindfulness prompt from evidence-based techniques",
            "confidence": 0.9
        },
        {
            "title": "Benefit Reflection",
            "description": "Explained why this type of break supports mental health and focus",
            "confidence": break_conf if break_conf else 0.8
        },
        {
            "title": "Activity Recommendation",
            "description": "Suggested a concrete 2-5 minute action you can take right now",
            "confidence": break_conf if break_conf else 0.75
        }
    ]
    render_explainability_view(reasoning_steps, key="break_reasoning")
    
    # ✅ ADDITION: Trust caption
    st.markdown("### 🔒 Trust and Predictability by Design")
    st.write(
        "Mindful break suggestions are generated using a three-step reasoning "
        "flow (reminder → reflection → recommendation). The confidence bar above "
        "shows how reliable the mentor believes today's suggestion is. "
        "When confidence is lower, treat the suggestion as a gentle nudge rather "
        "than a strong instruction—you stay fully in control of when and how "
        "you take breaks."
    )

    # 🧠 Explainability View — why this break?
    with st.expander("Why this break? (Explainability view)"):
        if llm_output.get("reflection"):
            st.write(llm_output["reflection"])

        st.caption(
            "This explanation shows why the mentor suggested this kind of break. "
            "You stay in control — feel free to ignore or adapt any suggestion."
        )
    # ──────────────────────────────────────────────────────────────
    # 🔒 Trust and Predictability by Design
    # ──────────────────────────────────────────────────────────────
st.markdown("### 🔒 Trust and Predictability by Design")

st.write(
        "Mindful break suggestions are generated using a three-step reasoning "
        "flow (reminder → reflection → recommendation). The confidence bar above "
        "shows how reliable the mentor believes today's suggestion is. "
        "When confidence is lower, treat the suggestion as a gentle nudge rather "
        "than a strong instruction — you stay fully in control of when and how "
        "you take breaks."
)



enable_auto_breaks_main = st.checkbox(
    "Enable automatic reminders", value=True, key="auto_reminder_box"
)
interval = st.slider(
    "Remind me every (minutes)",
    15,
    120,
    60,
    15,
    key="interval_main_slider",
)
play_sound_main = st.checkbox(
    "🔔 Sound alert", value=True, key="sound_alert_box"
)

if enable_auto_breaks_main:
    auto_mindfulness_reminder(
        interval_minutes=interval,
        enable_sound=play_sound_main,
    )

if os.path.exists("data/break_log.json"):
    try:
        with open("data/break_log.json", "r") as f:
            logs = json.load(f)
        if logs:
            latest = logs[-1]
            st.markdown(
                f"<p style='font-size:16px;'><b>🕒 Last Break:</b> "
                f"{latest['scheduled_time']} — {latest['message']}</p>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No recent break logs found.")
    except Exception as e:
        st.error(f"Error reading break log: {e}")


def show_break_chart():
    if not os.path.exists("data/break_log.json"):
        st.info("No breaks logged yet.")
        return
    logs = json.load(open("data/break_log.json"))
    df = pd.DataFrame(logs)
    df["time"] = pd.to_datetime(df["scheduled_time"])
    today = df[df["time"].dt.date == datetime.today().date()]
    if today.empty:
        st.info("No breaks logged today yet.")
        return
    st.markdown("### 🧘 Today's Breaks")
    chart = (
        alt.Chart(today)
        .mark_bar()
        .encode(x="time:T", y="count()", tooltip=["message"])
        .properties(height=200)
    )
    st.altair_chart(chart, use_container_width=True)


show_break_chart()

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 24px 0;'>
    <p style='font-size: 16px; margin-bottom: 8px;'>
        💡 <strong>Remember:</strong> Small, consistent steps lead to lasting change.
    </p>
    <p style='font-size: 13px; color: #999;'>
        Your plans are saved in this session. Download them to keep permanently.
    </p>
</div>
""",
    unsafe_allow_html=True,
)



# ──────────────────────────────────────────────────────────────
# 🌅 Morning Wellness Check-In
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🌅 Morning Wellness Check-In")

with st.form("checkin_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        mood = st.selectbox("Mood", ["", "calm", "neutral", "anxious", "frustrated"])
        energy = st.selectbox("Energy", ["", "low", "medium", "high"])
    with col2:
        sleep = st.selectbox("Sleep quality", ["", "poor", "ok", "great"])
        workload = st.selectbox("Workload", ["", "light", "normal", "heavy"])
    notes = st.text_area("Notes (optional)")
    do_checkin = st.form_submit_button("Adjust my plan for today")

if do_checkin:
    ck = CheckIn(
        mood=mood or None,
        sleep_quality=sleep or None,
        energy=energy or None,
        workload=workload or None,
        notes=notes or None,
    )

    # ✅ ADDITION: Get autonomy level and map to coach_mode
    with st.spinner("Adjusting your plan for today…"):
        try:
            autonomy_level = st.session_state.get("autonomy_level", "Gentle Suggester")
            
            # Map UI label to mode string
            mode_map = {
                "Passive Observer": "passive",
                "Gentle Suggester": "gentle",
                "Active Coach": "active",
                "Directive Guide": "directive"
            }
            coach_mode = mode_map.get(autonomy_level, "gentle")
            
            adj = run_morning_checkin(ck, coach_mode=coach_mode)
        except Exception as e:
            st.error(f"Could not adjust based on check-in: {e}")
            adj = None

    if adj is not None:
        # ✅ Save check-in + AI adjustment for dashboards
        try:
            save_checkin(ck.model_dump(), adj.model_dump())
        except Exception as e:
            st.warning(
                f"Check-in saved only for this session (storage error: {e})"
            )

        # 🤖 AI feedback section (keeps your heading)
        st.markdown("### 🤖 AI Feedback on Today’s Check-In")

        if getattr(adj, "summary", None):
            st.success(adj.summary)

        # Focus + risk flags (unchanged order)
        if getattr(adj, "focus_for_today", None):
            st.markdown("**Focus for today**")
            for a in adj.focus_for_today:
                st.write(f"- {a}")

        if getattr(adj, "risk_flags", None):
            st.caption("Flags: " + ", ".join(adj.risk_flags))

# ✅ ADDITION: Show both confidence and uncertainty
        checkin_conf = getattr(adj, "confidence", None)
        cols = st.columns(2)
        with cols[0]:
            render_confidence(None, checkin_conf, key="checkin_conf")
        with cols[1]:
            render_uncertainty(checkin_conf, key="checkin_uncertainty", show_explanation=False)
        
        note = getattr(adj, "confidence_note", None)
        if note:
            st.caption(f"📊 {note}")
        
        # ✅ ADDITION: Show which coaching mode was used
        st.caption(
            f"**AI coaching style:** {autonomy_level} "
            "(change this anytime from the sidebar)."
        )
        
        # ✅ ADDITION: Explainability view
        with st.expander("🔍 Why these suggestions? (Explainability view)"):
            st.markdown(f"""
**Your inputs:**
- Mood: **{mood or '—'}**
- Sleep quality: **{sleep or '—'}**
- Energy: **{energy or '—'}**
- Workload: **{workload or '—'}**
            """)
            if notes:
                st.markdown(f"- Notes: _{notes}_")
            
            if getattr(adj, "risk_flags", None):
                st.markdown("**Risk flags detected:** " + ", ".join(adj.risk_flags))
            
            st.caption(
                "This view shows which signals influenced today's micro-plan. "
                "You stay in control—feel free to ignore or modify any suggestion."
            )
        
        # ✅ ADDITION: Trust caption
        st.markdown("#### 🔒 Trust and Predictability by Design")
        st.caption(
            "The Morning Wellness Check-In uses only the mood, sleep, energy, workload, "
            "and notes you provide above. The confidence bar reflects how reliable the "
            "mentor believes today's suggestions are; when confidence is lower, treat "
            "the plan as a gentle suggestion rather than a prescription."
        )
        mode_labels = {
        "passive": "Passive Observer – mostly reflective, very low autonomy.",
        "gentle": "Gentle Suggester – soft, optional suggestions (default).",
        "active": "Active Coach – concrete action steps, medium autonomy.",
        "directive": "Directive Guide – very structured plan, higher autonomy.",
        }

        st.caption(
        f"**AI coaching style:** {mode_labels.get(coach_mode, coach_mode)} "
        "(change this anytime from the sidebar)."
        )

        # 🔍 Explainability View – why these suggestions?
        with st.expander("🔍 Why these suggestions? (Explainability view)"):
            st.markdown(
                f"""
- Mood: **{mood or '—'}**
- Sleep quality: **{sleep or '—'}**
- Energy: **{energy or '—'}**
- Workload: **{workload or '—'}**
                """
            )
            if notes:
                st.markdown(f"- Notes: _{notes}_")

            if getattr(adj, "risk_flags", None):
                st.markdown(
                    "**Risk flags detected:** " + ", ".join(adj.risk_flags)
                )

            st.caption(
                "This view shows which signals influenced today's micro-plan. "
                "You stay in control — feel free to ignore or modify any suggestion."
            )




# ──────────────────────────────────────────────────────────────
# 🌬 Guided Meditations & Breathing Exercises
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🌬 Guided Meditations & Breathing Exercises")

init_session()

st.markdown("### 🧭 Personalize Your Session")
user_goal = st.text_input("Describe your current state or need:", "")
stress_level = st.slider("Stress Level", 0, 10, 0)

# 🔧 Autonomy control for this user story
autonomy_mode = st.radio(
    "Mentor autonomy level",
    ["Passive (show more options)", "Balanced guidance", "Directive (strong recommendation)"],
    index=1,
    horizontal=True,
    help=(
        "Passive: more options, you choose.\n"
        "Balanced: a small curated set.\n"
        "Directive: the mentor leans in and suggests one clear path."
    ),
)

if user_goal.strip() == "":
    st.warning("Please describe your current state or need before continuing.")
else:
    if st.button("💡 Get AI-Recommended Techniques"):
        with st.spinner("Analyzing your needs..."):
            result = run_mentor_cycle(
                user_goal=user_goal,
                stress_level=stress_level,
                profile={
                    "context": "corporate",
                    "session_type": "guided_practice",
                    # NEW: autonomy mode passed into the graph/LLM
                    "autonomy_mode": autonomy_mode,
                },
                history=st.session_state.get("session_history", []),
            )

            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception as e:
                    st.error(f"⚠️ Could not parse AI response: {e}")
                    st.stop()

            st.session_state["techniques"] = result.get("techniques", [])
            st.session_state["summary"] = result.get(
                "summary",
                "Stay mindful and consistent!",
            )

            if st.session_state["techniques"]:
                st.success("✨ Here are your personalized AI mindfulness techniques:")
                for technique in st.session_state["techniques"]:
                    st.markdown(
                        f"### 🧘 {technique['title']} · "
                        f"_{technique['duration_min']} min_"
                    )
                    st.write(technique["description"])
                    for step in technique["steps"]:
                        st.markdown(f"- {step}")
                st.markdown(
                    f"---\n**Reflection:** {st.session_state['summary']}"
                )

                # ─────────────────────────────────────────────
                # ✅ Uncertainty / Confidence visualization
                # ─────────────────────────────────────────────
                guided_conf = result.get("confidence")
                guided_conf_note = result.get("confidence_note")

                render_confidence(
                    provenance=None,
                    confidence=guided_conf,
                    key="guided_techniques_conf",
                )
                if guided_conf_note:
                    st.caption(f"Confidence note: {guided_conf_note}")

                # ─────────────────────────────────────────────
                # 🔍 Explainability View
                # ─────────────────────────────────────────────
                with st.expander("🔍 Why did the mentor pick these techniques?"):
                    explanation = (
                        result.get("explanation")
                        or result.get("reasoning")
                        or ""
                    )
                    if explanation:
                        st.write(explanation)
                    else:
                        # Safe fallback explanation if graph doesn’t return one
                        st.write(
                            f"These techniques were selected to match your goal "
                            f"('{user_goal}') and stress level ({stress_level}/10) "
                            f"under **{autonomy_mode}**."
                        )
                    st.caption(
                        "This view summarizes the mentor's internal reasoning so the "
                        "selection feels transparent instead of mysterious."
                    )

                # ─────────────────────────────────────────────
                # 🔒 Trust & Predictability by Design
                # ─────────────────────────────────────────────
                st.markdown("#### 🔒 Trust & Predictability by Design")
                st.info(
                    "• Your **stress level** influences how intense or gentle the practices are.\n"
                    "• Your **text description** shapes the focus (anxiety, focus, sleep, etc.).\n"
                    "• **Autonomy level** controls how strongly the mentor steers you:\n"
                    "  - Passive → more options, you stay in full control.\n"
                    "  - Balanced → a small, curated set.\n"
                    "  - Directive → one main path is highlighted.\n"
                    "These rules make the mentor’s behavior consistent and easier to predict."
                )

            else:
                st.warning(
                    "⚠️ No AI-generated techniques received. "
                    "Try again or check your API key."
                )





# ────────────────────────────────────────────────────────────────────
# 📊 Daily Productivity Tracking (NEW SECTION - Add after Morning Check-In)
# ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 📊 Daily Productivity Tracking")

st.caption(
    "Track your productivity each day to see how it relates to your stress levels. "
    "Rate your overall productivity and add optional notes about what helped or hindered you."
)

with st.form("productivity_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    
    with col1:
        prod_date = st.date_input(
            "Date",
            value=datetime.today(),
            help="Which day are you tracking?"
        )
        
        productivity_score = st.slider(
            "Productivity Score (0-10)",
            min_value=0,
            max_value=10,
            value=5,
            help="0 = Completely unproductive, 10 = Extremely productive"
        )
    
    with col2:
        # Visual productivity level indicator
        if productivity_score >= 8:
            prod_label = "🚀 Highly Productive"
            prod_color = "green"
        elif productivity_score >= 6:
            prod_label = "✅ Good Progress"
            prod_color = "blue"
        elif productivity_score >= 4:
            prod_label = "⚡ Moderate Output"
            prod_color = "orange"
        else:
            prod_label = "🐌 Low Productivity"
            prod_color = "red"
        
        st.markdown(f"### {prod_label}")
        st.progress(productivity_score / 10)
    
    prod_notes = st.text_area(
        "What helped or hindered your productivity today? (optional)",
        placeholder="e.g., 'Had a great focus session in the morning, but got interrupted by meetings in the afternoon'",
        height=100
    )
    
    submit_productivity = st.form_submit_button(
        "💾 Save Today's Productivity",
        use_container_width=True,
        type="primary"
    )

if submit_productivity:
    try:
        from services.productivity_storage import save_productivity
        
        save_productivity(
            date_iso=prod_date.isoformat(),
            productivity=float(productivity_score),
            notes=prod_notes.strip() or None
        )
        
        st.success(f"✅ Productivity logged for {prod_date.strftime('%B %d, %Y')}")
        st.balloons()
        
        # Show helpful tip after first few entries
        all_prod = load_productivity() if callable(load_productivity) else []
        if len(all_prod) >= 3:
            st.info(
                "💡 **Tip**: You now have enough data to see productivity trends! "
                "Scroll down to the 'Productivity vs. Stress Insights' section to analyze patterns."
            )
        
    except Exception as e:
        st.error(f"Could not save productivity: {e}")

# Show recent productivity entries
try:
    from services.productivity_storage import load_productivity
    recent_prod = load_productivity() or []
    
    if recent_prod:
        st.markdown("### 📋 Recent Productivity Logs")
        
        # Show last 5 entries
        for entry in reversed(recent_prod[-5:]):
            date_str = entry.get("date", "Unknown date")
            score = entry.get("productivity", 0)
            notes = entry.get("notes", "")
            
            with st.expander(f"📅 {date_str} — Score: {score}/10", expanded=False):
                st.progress(float(score) / 10)
                if notes:
                    st.caption(f"Notes: {notes}")
        
        # Show total count
        st.caption(f"Total productivity logs: {len(recent_prod)}")
        
except Exception as e:
    st.warning(f"Could not load recent productivity entries: {e}")


# ────────────────────────────────────────────────────────────────────
# 📊 Stress-Level Tracking Dashboard
# ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 📊 Stress-Level Tracking Dashboard")

all_checkins = load_checkins() if callable(load_checkins) else []

if not all_checkins:
    st.info(
        "No saved Morning Wellness Check-Ins yet. "
        "Log a few days above so we can show daily and weekly stress trends."
    )
else:
    # Build a DataFrame for charts
    df = build_stress_dataframe(all_checkins)
    max_date = df["date"].max()
    num_days = len(df)

    # Show data summary
    st.markdown(f"**📈 Tracking Data**: {num_days} days logged")
    
    if num_days < 3:
        st.warning(
            "⚠️ You have less than 3 days of data. Log at least 3-5 days "
            "for more reliable AI insights."
        )

    st.markdown("#### ⏱ Time Window for Daily Stress")
    days_window = st.radio(
        "How much recent history do you want to see?",
        options=[7, 14, 21, 30],
        index=1,
        format_func=lambda x: f"{x} days",
        horizontal=True,
        help="Controls how many days of stress check-ins are shown in the line chart.",
    )

    cutoff = max_date - pd.Timedelta(days=days_window - 1)
    df_window = df[df["date"] >= cutoff]

    col_daily, col_weekly = st.columns(2)

    # Daily stress line chart
    with col_daily:
        st.markdown("#### 📈 Daily Stress (0–100)")
        daily_chart = (
            alt.Chart(df_window)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("stress_score:Q", scale=alt.Scale(domain=[0, 100]), title="Stress Score"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("stress_score:Q", title="Stress Score"),
                    alt.Tooltip("mood:N", title="Mood"),
                    alt.Tooltip("sleep_quality:N", title="Sleep"),
                    alt.Tooltip("energy:N", title="Energy"),
                    alt.Tooltip("workload:N", title="Workload"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(daily_chart, use_container_width=True)

    # Weekly average bars
    with col_weekly:
        st.markdown("#### 📊 Weekly Average Stress")
        weekly = (
            df.groupby("week_start", as_index=False)["stress_score"]
            .mean()
            .rename(columns={"stress_score": "avg_stress"})
        )
        weekly_chart = (
            alt.Chart(weekly)
            .mark_bar()
            .encode(
                x=alt.X("week_start:T", title="Week Starting"),
                y=alt.Y("avg_stress:Q", scale=alt.Scale(domain=[0, 100]), title="Average Stress"),
                tooltip=[
                    alt.Tooltip("week_start:T", title="Week"),
                    alt.Tooltip("avg_stress:Q", title="Avg Stress", format=".1f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(weekly_chart, use_container_width=True)

    # Quick stats
    st.markdown("#### 📊 Quick Statistics")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    avg_stress = df["stress_score"].mean()
    max_stress = df["stress_score"].max()
    min_stress = df["stress_score"].min()
    latest_stress = df_window.iloc[-1]["stress_score"] if not df_window.empty else 0
    
    with col_stat1:
        st.metric("Average Stress", f"{avg_stress:.1f}")
    with col_stat2:
        st.metric("Latest Score", f"{latest_stress:.1f}")
    with col_stat3:
        st.metric("Peak Stress", f"{max_stress:.1f}")
    with col_stat4:
        st.metric("Lowest Stress", f"{min_stress:.1f}")

    # ────────────────────────────────────────────────────────────────
    # 🤖 AI ANALYSIS - Only run when user clicks button
    # ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Mentor's View of Your Stress")
    
    st.caption(
        "Get AI-powered insights about your stress patterns, main drivers, "
        "and personalized suggestions based on your check-in history."
    )

    # Check if analysis already exists in session
    analysis_exists = "last_stress_analysis" in st.session_state

    # ✅ BUTTON - Only analyze when clicked
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        analyze_clicked = st.button(
            "🔍 Analyze My Stress Patterns",
            key="analyze_stress_btn",
            type="primary",
            use_container_width=True
        )
    with col_btn2:
        if analysis_exists:
            st.caption("💡 Analysis already generated. Click to refresh with latest data.")

    if analyze_clicked:
        with st.spinner("🧠 Analyzing your stress trends..."):
            from graph.graph import create_stress_analytics_graph
            
            compiled = create_stress_analytics_graph()
            result = compiled.invoke({
                "checkins": all_checkins,
                "stress_analytics": {},
            })
            
            analytics_dict = result.get("stress_analytics") or {}

        # Display the summary
        summary_text = analytics_dict.get("summary", "No analysis available.")
        st.info(summary_text)
        
        # Show key drivers
        key_drivers = analytics_dict.get("key_drivers", [])
        if key_drivers:
            st.markdown("**Main stress drivers:**")
            for driver in key_drivers:
                st.markdown(f"• {driver}")
        
        # Show suggestions
        suggestions = analytics_dict.get("suggestions", [])
        if suggestions:
            st.markdown("**Suggestions for next week:**")
            for suggestion in suggestions:
                st.markdown(f"• {suggestion}")
        
        # 🎯 Display confidence
# ✅ ADDITION: Show both confidence and uncertainty
        stress_conf = analytics_dict.get("confidence")
        cols = st.columns(2)
        with cols[0]:
            render_confidence(None, stress_conf, key="stress_analytics_conf")
        with cols[1]:
            render_uncertainty(stress_conf, key="stress_analytics_uncertainty", show_explanation=False)
        
        conf_note = analytics_dict.get("confidence_note")
        if conf_note:
            st.caption(f"📊 {conf_note}")
        
        # ✅ ADDITION: Explainability view
        num_days = len(all_checkins)
        reasoning_steps = [
            {
                "title": "Data Collection",
                "description": f"Analyzed {num_days} check-ins to identify patterns",
                "confidence": 0.95
            },
            {
                "title": "Pattern Recognition",
                "description": "Identified stress triggers and timing patterns from mood, sleep, energy, and workload data",
                "confidence": stress_conf if stress_conf else 0.8
            },
            {
                "title": "Driver Analysis",
                "description": f"Extracted {len(analytics_dict.get('key_drivers', []))} main stress drivers",
                "confidence": stress_conf if stress_conf else 0.75
            },
            {
                "title": "Recommendation Generation",
                "description": f"Generated {len(analytics_dict.get('suggestions', []))} personalized suggestions",
                "confidence": stress_conf if stress_conf else 0.7
            }
        ]
        render_explainability_view(reasoning_steps, key="stress_analytics_reasoning")
        
        # ✅ ADDITION: Trust caption
        st.markdown("#### 🔒 Trust and Predictability by Design")
        st.caption(
            f"Stress analytics analyzes {num_days} days of your morning check-ins. "
            "It identifies patterns in mood, sleep, energy, and workload to find stress triggers. "
            "More days of data = higher confidence. Treat low-confidence insights as preliminary observations."
        )
        
        # Store in session state so it persists after button click
        st.session_state["last_stress_analysis"] = analytics_dict
        st.session_state["stress_analysis_timestamp"] = datetime.now()
        
        # ────────────────────────────────────────────────────────────
        # 💬 User feedback on stress analytics
        # ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💬 Was this stress analysis helpful?")
        
        col_fb1, col_fb2, col_fb3 = st.columns(3)
        with col_fb1:
            if st.button("👍 Helpful", key="stress_helpful", use_container_width=True):
                st.success("Thanks! We'll continue tracking your patterns.")
                # Optional: save feedback to file
                try:
                    feedback_data = {
                        "feature": "stress_analytics",
                        "feedback": "helpful",
                        "timestamp": datetime.now().isoformat(),
                    }
                    # You can implement save_feedback() later
                    st.session_state["stress_feedback"] = "helpful"
                except Exception:
                    pass
                    
        with col_fb2:
            if st.button("👎 Not helpful", key="stress_not_helpful", use_container_width=True):
                st.warning("Thanks for the feedback. Try logging more days for better insights.")
                st.session_state["stress_feedback"] = "not_helpful"
                
        with col_fb3:
            if st.button("💡 Suggest improvement", key="stress_suggest", use_container_width=True):
                st.session_state["show_stress_feedback_form"] = True
        
        # Show feedback form if user clicked suggest improvement
        if st.session_state.get("show_stress_feedback_form"):
            feedback_text = st.text_area(
                "What would make this analysis more useful?",
                key="stress_feedback_text",
                placeholder="e.g., 'I'd like to see specific time patterns', 'Compare with last month', etc."
            )
            if st.button("Submit Feedback", key="submit_stress_feedback"):
                if feedback_text.strip():
                    st.success(f"✅ Feedback recorded: {feedback_text}")
                    st.session_state["stress_feedback_detail"] = feedback_text
                    st.session_state["show_stress_feedback_form"] = False
                    st.rerun()
    
    # ────────────────────────────────────────────────────────────────
    # Show previous analysis if it exists (after button was clicked)
    # ────────────────────────────────────────────────────────────────
    elif analysis_exists:
        with st.expander("📋 View Last Analysis", expanded=False):
            analytics_dict = st.session_state["last_stress_analysis"]
            analysis_time = st.session_state.get("stress_analysis_timestamp")
            
            if analysis_time:
                st.caption(f"Generated: {analysis_time.strftime('%B %d, %Y at %I:%M %p')}")
            
            summary_text = analytics_dict.get("summary", "")
            if summary_text:
                st.info(summary_text)
            
            key_drivers = analytics_dict.get("key_drivers", [])
            if key_drivers:
                st.markdown("**Main stress drivers:**")
                for driver in key_drivers:
                    st.markdown(f"• {driver}")
            
            suggestions = analytics_dict.get("suggestions", [])
            if suggestions:
                st.markdown("**Suggestions:**")
                for suggestion in suggestions:
                    st.markdown(f"• {suggestion}")
            
            # Show confidence from cached analysis
            conf = analytics_dict.get("confidence")
            if conf is not None:
                st.progress(conf, text=f"AI Confidence: {int(conf * 100)}%")
                conf_note = analytics_dict.get("confidence_note")
                if conf_note:
                    st.caption(conf_note)
# ────────────────────────────────────────────────────────────────────
# 📈 Productivity vs. Stress Insights
# ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 📈 Productivity vs. Stress Insights")

st.caption(
    "Analyze how your stress levels correlate with your productivity. "
    "This helps identify when stress starts affecting your performance."
)

# Load productivity entries
try:
    from services.productivity_storage import load_productivity
    productivity_entries = load_productivity() or []
except Exception:
    productivity_entries = []

if not all_checkins or not productivity_entries:
    st.info(
        "📊 To see this comparison, you'll need both:\n\n"
        "1. **Morning Wellness Check-Ins** (for stress data)\n"
        "2. **Daily Productivity Tracking** (use the form above)\n\n"
        "Log both for at least 3-5 days to see meaningful patterns."
    )
    
    # Show what's missing
    if not all_checkins:
        st.warning("⚠️ Missing: Morning Check-Ins")
    if not productivity_entries:
        st.warning("⚠️ Missing: Productivity logs")
        
else:
    stress_df = build_stress_dataframe(all_checkins)
    prod_df = build_productivity_dataframe(productivity_entries)

    # Inner join on date (only days where we have BOTH)
    merged = stress_df.merge(
        prod_df[["date", "productivity", "prod_notes"]],
        on="date",
        how="inner",
        suffixes=("_stress", "_prod"),
    )

    if merged.empty:
        st.warning(
            "⚠️ We don't yet have any days where **both** stress and productivity were logged on the same day.\n\n"
            "**What to do:**\n"
            "1. Make sure you're logging Morning Check-Ins daily\n"
            "2. Also log your productivity score each day\n"
            "3. Come back after 3-5 days of consistent logging"
        )
    else:
        num_overlapping = len(merged)
        st.markdown(f"**📊 Overlapping Data**: {num_overlapping} days with both stress and productivity logged")
        
        if num_overlapping < 5:
            st.warning(
                f"⚠️ You have only {num_overlapping} days of overlapping data. "
                "Log at least 5-7 days for more reliable correlation insights."
            )

        # Scale productivity to 0–100 so it overlays nicely with stress
        merged["productivity_scaled"] = merged["productivity"] * 10.0

        st.markdown("#### 🔗 Daily Stress vs. Productivity Over Time")

        melted = merged.melt(
            id_vars="date",
            value_vars=["stress_score", "productivity_scaled"],
            var_name="metric",
            value_name="value",
        )

        # Create dual-axis chart
        chart = (
            alt.Chart(melted)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 100]), title="Score (0-100)"),
                color=alt.Color(
                    "metric:N",
                    title="Metric",
                    scale=alt.Scale(
                        domain=["stress_score", "productivity_scaled"],
                        range=["#d62728", "#1f77b4"],  # red vs blue
                    ),
                    legend=alt.Legend(
                        labelExpr="datum.label == 'stress_score' ? 'Stress' : 'Productivity (×10)'"
                    )
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("value:Q", title="Score", format=".1f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "📊 **Note**: Productivity is rescaled to 0–100 for visualization (multiply by 10). "
            "Look for periods where stress rises while productivity drops, or vice versa."
        )

        # Quick correlation stats
        st.markdown("#### 📊 Quick Statistics")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        avg_stress_overlap = merged["stress_score"].mean()
        avg_prod = merged["productivity"].mean()
        
        # Calculate simple correlation
        correlation = merged["stress_score"].corr(merged["productivity"])
        
        with col_stat1:
            st.metric("Avg Stress", f"{avg_stress_overlap:.1f}")
        with col_stat2:
            st.metric("Avg Productivity", f"{avg_prod:.1f}/10")
        with col_stat3:
            correlation_label = "Negative" if correlation < -0.2 else "Positive" if correlation > 0.2 else "Weak"
            st.metric("Correlation", correlation_label, f"{correlation:.2f}")
        with col_stat4:
            high_stress_days = len(merged[merged["stress_score"] >= 70])
            st.metric("High Stress Days", high_stress_days)

        # ────────────────────────────────────────────────────────────
        # 🤖 AI ANALYSIS - Only run when user clicks
        # ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Mentor's View of Stress vs Productivity")
        
        st.caption(
            "Get AI insights about how your stress and productivity interact, "
            "identify risk patterns, and receive personalized recommendations."
        )

        # Build the data payload
        joined_records = []
        for _, row in merged[["date", "stress_score", "productivity", "mood", "workload", "prod_notes"]].iterrows():
            record = {}
            for col in row.index:
                value = row[col]
                # Convert pandas Timestamp to ISO string
                if pd.api.types.is_datetime64_any_dtype(type(value)) or hasattr(value, 'isoformat'):
                    record[col] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                # Convert numpy/pandas numeric types to native Python types
                elif hasattr(value, 'item'):  # numpy scalar
                    record[col] = value.item()
                # Handle NaN/None
                elif pd.isna(value):
                    record[col] = None
                else:
                    record[col] = value
            
            joined_records.append(record)

        # Check if analysis exists
        analysis_exists = "last_productivity_analysis" in st.session_state

        # ✅ BUTTON - Only analyze when clicked
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            analyze_clicked = st.button(
                "🔍 Analyze Correlation",
                key="analyze_prod_btn",
                type="primary",
                use_container_width=True
            )
        with col_btn2:
            if analysis_exists:
                st.caption("💡 Analysis already generated. Click to refresh with latest data.")

        if analyze_clicked:
            with st.spinner("🧠 Analyzing how stress and productivity relate..."):
                from graph.graph import create_productivity_insights_graph
                
                compiled = create_productivity_insights_graph()
                result = compiled.invoke({
                    "records": joined_records,
                    "productivity_insights": {},
                })
                
                insights_dict = result.get("productivity_insights") or {}

            # Display correlation summary
            correlation_summary = insights_dict.get("correlation_summary", "No analysis available.")
            st.info(correlation_summary)
            
            # Show risk windows
            risk_windows = insights_dict.get("risk_windows", [])
            if risk_windows:
                st.markdown("**⚠️ High-risk patterns identified:**")
                for window in risk_windows:
                    st.markdown(f"• {window}")
            
            # Show suggestions
            suggestions = insights_dict.get("suggestions", [])
            if suggestions:
                st.markdown("**💡 Recommendations to protect performance:**")
                for suggestion in suggestions:
                    st.markdown(f"• {suggestion}")
            
      # ✅ ADDITION: Show both confidence and uncertainty
            prod_conf = insights_dict.get("confidence")
            cols = st.columns(2)
            with cols[0]:
                render_confidence(None, prod_conf, key="productivity_insights_conf")
            with cols[1]:
                render_uncertainty(prod_conf, key="productivity_insights_uncertainty", show_explanation=False)
            
            conf_note = insights_dict.get("confidence_note")
            if conf_note:
                st.caption(f"📊 {conf_note}")
            
            # ✅ ADDITION: Explainability view
            num_overlapping = len(merged)
            reasoning_steps = [
                {
                    "title": "Data Alignment",
                    "description": f"Matched {num_overlapping} days where both stress and productivity were logged",
                    "confidence": 0.95
                },
                {
                    "title": "Correlation Analysis",
                    "description": f"Calculated relationship between stress levels and productivity scores (correlation: {correlation:.2f})",
                    "confidence": 0.9 if num_overlapping >= 7 else 0.6
                },
                {
                    "title": "Risk Pattern Detection",
                    "description": f"Identified {len(insights_dict.get('risk_windows', []))} high-risk patterns where stress impacts performance",
                    "confidence": prod_conf if prod_conf else 0.75
                },
                {
                    "title": "Actionable Recommendations",
                    "description": f"Generated {len(insights_dict.get('suggestions', []))} strategies to protect productivity",
                    "confidence": prod_conf if prod_conf else 0.7
                }
            ]
            render_explainability_view(reasoning_steps, key="productivity_insights_reasoning")
            
            # ✅ ADDITION: Trust caption
            st.markdown("#### 🔒 Trust and Predictability by Design")
            st.caption(
                f"Productivity insights compare {num_overlapping} days where you logged both stress check-ins "
                "and productivity scores. The correlation analysis shows how they relate. "
                "Need 7+ overlapping days for reliable patterns. Lower confidence = treat as exploratory insights."
            )
            
            # Store in session state
            st.session_state["last_productivity_analysis"] = insights_dict
            st.session_state["productivity_analysis_timestamp"] = datetime.now()
            
            # ────────────────────────────────────────────────────────
            # 💬 User feedback
            # ────────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 💬 Did these insights help you?")
            
            col_fb1, col_fb2, col_fb3 = st.columns(3)
            with col_fb1:
                if st.button("👍 Helpful", key="prod_helpful", use_container_width=True):
                    st.success("Great! Keep logging both metrics to refine the analysis.")
                    st.session_state["prod_feedback"] = "helpful"
                    
            with col_fb2:
                if st.button("👎 Not helpful", key="prod_not_helpful", use_container_width=True):
                    st.warning("Thanks! More overlapping data will improve accuracy.")
                    st.session_state["prod_feedback"] = "not_helpful"
                    
            with col_fb3:
                if st.button("💡 Suggest improvement", key="prod_suggest", use_container_width=True):
                    st.session_state["show_prod_feedback_form"] = True
            
            # Show feedback form
            if st.session_state.get("show_prod_feedback_form"):
                feedback_text = st.text_area(
                    "How can we make this more actionable?",
                    key="prod_feedback_text",
                    placeholder="e.g., 'Show specific times when stress hurts productivity most', 'Compare weekdays vs weekends', etc."
                )
                if st.button("Submit Feedback", key="submit_prod_feedback"):
                    if feedback_text.strip():
                        st.success(f"✅ Feedback recorded: {feedback_text}")
                        st.session_state["prod_feedback_detail"] = feedback_text
                        st.session_state["show_prod_feedback_form"] = False
                        st.rerun()
        
        # ────────────────────────────────────────────────────────────
        # Show cached result if available
        # ────────────────────────────────────────────────────────────
        elif analysis_exists:
            with st.expander("📋 View Last Analysis", expanded=False):
                insights_dict = st.session_state["last_productivity_analysis"]
                analysis_time = st.session_state.get("productivity_analysis_timestamp")
                
                if analysis_time:
                    st.caption(f"Generated: {analysis_time.strftime('%B %d, %Y at %I:%M %p')}")
                
                correlation_summary = insights_dict.get("correlation_summary", "")
                if correlation_summary:
                    st.info(correlation_summary)
                
                risk_windows = insights_dict.get("risk_windows", [])
                if risk_windows:
                    st.markdown("**High-risk patterns:**")
                    for window in risk_windows:
                        st.markdown(f"• {window}")
                
                suggestions = insights_dict.get("suggestions", [])
                if suggestions:
                    st.markdown("**Recommendations:**")
                    for suggestion in suggestions:
                        st.markdown(f"• {suggestion}")
                
                # Show confidence
                conf = insights_dict.get("confidence")
                if conf is not None:
                    st.progress(conf, text=f"AI Confidence: {int(conf * 100)}%")
                    conf_note = insights_dict.get("confidence_note")
                    if conf_note:
                        st.caption(conf_note)

# ──────────────────────────────────────────────────────────────
# 📝 Weekly Reflection Journal
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📝 Weekly Reflection Journal")

st.caption(
    "Once a week, jot down how your stress felt, what you accomplished, "
    "and what was hard. The AI mentor will highlight patterns and growth over time."
)

JOURNAL_DIR = "data"
os.makedirs(JOURNAL_DIR, exist_ok=True)
JOURNAL_FILE = os.path.join(JOURNAL_DIR, "weekly_reflections.json")


def _load_weekly_reflections() -> list:
    if not os.path.exists(JOURNAL_FILE):
        return []
    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_weekly_reflection(entry: dict) -> None:
    entries = _load_weekly_reflections()
    entries.append(entry)
    with open(JOURNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


existing_entries = _load_weekly_reflections()
week_ending = st.date_input(
    "Week ending on",
    value=datetime.today(),
    help="Pick the week you are reflecting on (usually Friday or Sunday).",
    key="weekly_ref_date",
)

reflection_text = st.text_area(
    "Write about this week's stress patterns, accomplishments, and challenges:",
    placeholder=(
        "For example: This week I felt most stressed on Mon/Tue before stand-up. "
        "Wins: shipped the API refactor, stayed consistent with 10-min breaks. "
        "Challenges: skipped meditation on busy days."
    ),
    height=200,
    key="weekly_reflection_text",
)


def _normalize_text_block(value):
    """
    Normalize LLM outputs (list/dict/string) into clean text.
    - list -> bullet list
    - dict -> key: value lines
    - other -> string
    """
    if value is None:
        return ""

    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)

    if isinstance(value, dict):
        pairs = [f"{k}: {v}" for k, v in value.items() if v]
        return "\n".join(pairs)

    return str(value).strip()



analyze_week = st.button(
    "✨ Analyze and Save Weekly Reflection",
    key="analyze_week_btn",
    help="AI will analyze your reflection to identify stress patterns, accomplishments, and growth areas. Confidence score shows reliability of the analysis."
)

if analyze_week and reflection_text.strip():
    user_payload = {
        "current_week": {
            "week_ending": week_ending.isoformat(),
            "raw_text": reflection_text.strip(),
        },
        "recent_history": [
            {
                "week_ending": e.get("week_ending"),
                "summary": e.get("ai_summary", ""),
                "stress_pattern": e.get("stress_pattern", ""),
            }
            for e in existing_entries[-3:]
        ],
    }

    with st.spinner("Reflecting on your week..."):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a supportive corporate well-being coach. "
                            "Given a weekly reflection, plus a few recent summaries, "
                            "identify stress patterns, highlight accomplishments, "
                            "name key challenges, and describe any growth you see. "
                            "Then suggest 3–5 specific micro-actions for next week. "
                            "Reply ONLY as JSON with keys: "
                            "summary, stress_pattern, accomplishments, challenges, "
                            "growth_highlights, action_suggestions (list of strings), "
                            "confidence (float 0-1), confidence_note (string)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.6,
                max_tokens=700,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            st.error(f"Could not analyze weekly reflection: {e}")
        else:
            # Extract confidence
            confidence = data.get("confidence")
            if confidence is not None:
                try:
                    confidence = max(0.0, min(1.0, float(confidence)))
                except:
                    confidence = None
            confidence_note = data.get("confidence_note", "")

            entry = {
                "week_ending": week_ending.isoformat(),
                "raw_text": reflection_text.strip(),
                "ai_summary": _normalize_text_block(data.get("summary")),
                "stress_pattern":  _normalize_text_block(data.get("stress_pattern")),
                "accomplishments": _normalize_text_block(data.get("accomplishments")),
                "challenges":  _normalize_text_block(data.get("challenges")),
                "growth_highlights": _normalize_text_block(data.get("growth_highlights")),
                "action_suggestions": data.get("action_suggestions", []),
                "confidence": confidence,
                "confidence_note": confidence_note,
                "saved_at": datetime.now().isoformat(),
            }
            _save_weekly_reflection(entry)
            st.success("✅ Weekly reflection saved and analyzed.")

            st.markdown("### 🧠 AI Summary of Your Week")
            if entry["ai_summary"]:
                st.info(entry["ai_summary"])

            cols = st.columns(2)
            with cols[0]:
                st.markdown("#### 🔍 Stress Pattern")
                st.write(entry["stress_pattern"] or "No clear pattern extracted.")
                st.markdown("#### 🌟 Accomplishments")
                st.write(entry["accomplishments"] or "Write at least one win each week.")
            with cols[1]:
                st.markdown("#### ⚔️ Challenges")
                st.write(entry["challenges"] or "Note what felt hardest this week.")
                st.markdown("#### 🌱 Growth Highlights")
                st.write(
                    entry["growth_highlights"]
                    or "Growth will show up after a few weeks of journaling."
                )

            if entry["action_suggestions"]:
                st.markdown("#### 🎯 Suggestions for Next Week")
                for s in entry["action_suggestions"]:
                    st.write(f"- {s}")

            # Display confidence and uncertainty
            cols = st.columns(2)
            with cols[0]:
                render_confidence(
                    provenance=None,
                    confidence=entry.get("confidence"),
                    key="weekly_reflection_conf",
                )
            with cols[1]:
                render_uncertainty(
                    confidence=entry.get("confidence"),
                    key="weekly_reflection_uncertainty",
                    show_explanation=False,
                )
            if entry.get("confidence_note"):
                st.caption(f"📝 {entry['confidence_note']}")

            # Explainability view - show AI reasoning process
            reasoning_steps = [
                {
                    "title": "Text Analysis",
                    "description": f"Analyzed {len(reflection_text.strip().split())} words from your weekly reflection",
                    "confidence": 0.9
                },
                {
                    "title": "Pattern Recognition",
                    "description": f"Compared with {len(existing_entries[-3:])} recent reflections to identify trends",
                    "confidence": entry.get("confidence", 0.7) if existing_entries else 0.4
                },
                {
                    "title": "Stress Pattern Extraction",
                    "description": "Identified stress triggers, timing, and intensity from your writing",
                    "confidence": entry.get("confidence", 0.7)
                },
                {
                    "title": "Action Suggestions",
                    "description": f"Generated {len(entry.get('action_suggestions', []))} personalized micro-actions based on patterns",
                    "confidence": entry.get("confidence", 0.7)
                }
            ]
            render_explainability_view(reasoning_steps, key="weekly_reflection_reasoning")

             # ✅ ADDITION: Trust caption
            st.markdown("#### 🔒 Trust and Predictability by Design")
            st.caption(
                "Weekly reflection analysis uses your written reflection plus up to 3 recent weeks "
                "to identify patterns. The AI looks for stress triggers, accomplishments, and growth areas. "
                "More weeks logged = better trend detection. Low confidence = treat as preliminary insights."
            )

            # Feedback widgets
            st.markdown("---")
            render_feedback_widget(
                key="weekly_reflection_feedback",
                title="How helpful was this weekly analysis?",
                feedback_type="weekly_reflection",
                metadata={"confidence": entry.get("confidence")},
                compact=True
            )

            if entry.get("confidence") is not None:
                render_confidence_accuracy_check(
                    key="weekly_reflection_accuracy",
                    confidence=entry.get("confidence"),
                    feature_name="weekly reflection analysis"
                )

if existing_entries:
    st.markdown("### 📚 Recent Weekly Entries")
    for e in reversed(existing_entries[-4:]):
        with st.expander(
            f"Week ending {e.get('week_ending')} – click to view summary",
            expanded=False,
        ):
            st.markdown("**Summary**")
            st.write(e.get("ai_summary", ""))
            st.markdown("**Stress pattern**")
            st.write(e.get("stress_pattern", ""))
            st.markdown("**Accomplishments**")
            st.write(e.get("accomplishments", ""))
            st.markdown("**Challenges**")
            st.write(e.get("challenges", ""))
            st.markdown("**Growth highlights**")
            st.write(e.get("growth_highlights", ""))


# ──────────────────────────────────────────────────────────────
# 💬 AI Coaching & Support — Mentor Conversations
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 💬 AI Coaching & Support — Mentor Conversations")
# ✅ ADDITION: Trust caption explaining mentor behavior
st.caption(
    "💡 **How the mentor works:** The AI provides calm, evidence-based responses to your stress concerns. "
    "Confidence scores show how certain the mentor is about its advice. This is for general wellness support, "
    "not clinical treatment. For severe anxiety or depression, please contact a mental health professional."
)

# Keep mentor conversation history in session_state
if "mentor_history" not in st.session_state:
    st.session_state["mentor_history"] = []

# 1) Show previous conversation from history
for idx, h in enumerate(st.session_state["mentor_history"]):
    st.chat_message("user").markdown(h.get("user", ""))
    st.chat_message("assistant").markdown(h.get("assistant", ""))

    # show confidence per turn if present
# ✅ ADDITION: Show both confidence and uncertainty per turn
    if "confidence" in h:
        mentor_conf = h.get("confidence")
        cols = st.columns(2)
        with cols[0]:
            render_confidence(None, mentor_conf, key=f"mentor_conf_{idx}")
        with cols[1]:
            render_uncertainty(mentor_conf, key=f"mentor_uncertainty_{idx}", show_explanation=False)
        
        if h.get("confidence_note"):
            st.caption(f"📊 {h['confidence_note']}")

# 2) Input form directly under the heading
with st.form("mentor_form", clear_on_submit=True):
    user_msg = st.text_area(
        "Tell me what's stressing you out today...",
        key="mentor_input",
        placeholder="Type a short message about what's on your mind…",
    )
    send_clicked = st.form_submit_button("Send")

# 3) Only send to the mentor when the user clicks Send
if send_clicked and user_msg.strip():
    msg_text = user_msg.strip()
    try:
        from services.mentor_graph import run_mentor_conversation

        reply = run_mentor_conversation(
            user_message=msg_text,
            history=st.session_state["mentor_history"],
        )
    except Exception as e:
        reply = {
            "user": msg_text,
            "assistant": f"⚠️ Mentor error: {e}",
            "confidence": None,
            "confidence_note": None,
        }

    # Normalise to a dict
    if not isinstance(reply, dict):
        reply = {
            "user": msg_text,
            "assistant": str(reply),
            "confidence": None,
            "confidence_note": None,
        }

    # Append the full turn to history and rerun so it shows once
    st.session_state["mentor_history"].append(reply)
    st.rerun()



# ──────────────────────────────────────────────────────────────────────────────
# ⭐ Motivational Messaging (Standalone Section at the End)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## ⭐ Motivational Messaging")

task_completion = st.session_state.get("task_completion", {})
total = len(task_completion)
done = sum(1 for v in task_completion.values() if v)

if total == 0 or done == 0:
    st.info("Complete some daily practices to receive a tailored motivational message.")
else:
    completed_tasks = [task for task, finished in task_completion.items() if finished]

    # 🔘 Only generate when the user asks
    if st.button(
        "✨ Generate motivational message",
        help="AI creates personalized encouragement based on your task completion rate and consistency. Higher completion = higher confidence."
    ):
        try:
            result = run_motivation_message(
                completed=done,
                total=total,
                activities=completed_tasks,
            )
            msg = result.get("message", "")
            confidence = result.get("confidence")
            confidence_note = result.get("confidence_note", "")
        except Exception as e:
            msg = (
                "You're making meaningful progress. Even one completed practice is a real step "
                f"toward lower stress. (AI motivation unavailable: {e})"
            )
            confidence = None
            confidence_note = ""

        st.success(msg)

        # Display confidence and uncertainty
        cols = st.columns(2)
        with cols[0]:
            render_confidence(
                provenance=None,
                confidence=confidence,
                key="motivation_conf",
            )
        with cols[1]:
            render_uncertainty(
                confidence=confidence,
                key="motivation_uncertainty",
                show_explanation=False,
            )
        if confidence_note:
            st.caption(f"📝 {confidence_note}")

        # Explainability view - show AI reasoning process
        completion_ratio = done / total if total > 0 else 0
        reasoning_steps = [
            {
                "title": "Task Completion Analysis",
                "description": f"You completed {done} out of {total} practices ({int(completion_ratio * 100)}% completion rate)",
                "confidence": 0.95
            },
            {
                "title": "Consistency Assessment",
                "description": f"Evaluated consistency based on {len(completed_tasks)} completed activities",
                "confidence": 0.9 if completion_ratio >= 0.7 else 0.6
            },
            {
                "title": "Message Personalization",
                "description": "Generated encouragement tailored to your progress level and activity mix",
                "confidence": confidence if confidence else 0.7
            }
        ]
        render_explainability_view(reasoning_steps, key="motivation_reasoning")

                # ✅ ADDITION: Trust caption
        st.markdown("#### 🔒 Trust and Predictability by Design")
        st.caption(
            f"Motivational messages are based on your task completion rate ({done}/{total} completed). "
            "Higher completion rates get stronger encouragement. The confidence score reflects how well "
            "your pattern matches typical wellness journeys. This is purely supportive—you control your pace."
        )

        # Feedback widgets
        st.markdown("---")
        render_feedback_widget(
            key="motivation_feedback",
            title="Did this message motivate you?",
            feedback_type="motivational_message",
            metadata={"confidence": confidence, "message_length": len(msg)},
            compact=True
        )

        if confidence is not None:
            render_confidence_accuracy_check(
                key="motivation_accuracy",
                confidence=confidence,
                feature_name="motivational message"
            )

# ──────────────────────────────────────────────────────────────────────────────
# 🧑‍💼 HR Wellness Insights (Anonymized Stress Trends)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 🧑‍💼 HR Wellness Insights (Anonymized)")

all_checkins = load_checkins() if callable(load_checkins) else []

if not all_checkins:
    st.info("No check-in data available yet to generate HR insights.")
else:
    # 🔘 Only show charts + LLM summary when HR clicks the button
    if st.button(
        "📊 Generate HR wellness insights",
        help="Aggregates anonymized employee check-in data to identify workforce stress trends. More data days = higher confidence in patterns."
    ):
        df = build_stress_dataframe(all_checkins)

        # Weekly averages (same as personal dashboard)
        weekly = (
            df.groupby("week_start", as_index=False)["stress_score"]
            .mean()
            .rename(columns={"stress_score": "avg_stress"})
        )

        st.markdown("### 📊 Weekly Employee Stress Overview")

        weekly_chart = (
            alt.Chart(weekly)
            .mark_bar()
            .encode(
                x=alt.X("week_start:T", title="Week"),
                y=alt.Y("avg_stress:Q", title="Average Stress (0–100)"),
                tooltip=["week_start:T", "avg_stress:Q"],
            )
            .properties(height=240)
        )
        
        st.altair_chart(weekly_chart, use_container_width=True)

        # Stress distribution
        st.markdown("### 📈 Stress Level Distribution in the Workforce")

        df_copy = df.copy()
        df_copy["band"] = pd.cut(
            df_copy["stress_score"],
            bins=[0, 33, 66, 100],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )

        band_counts = (
            df_copy.groupby("band", as_index=False)["stress_score"]
            .count()
            .rename(columns={"stress_score": "count"})
        )

        dist_chart = (
            alt.Chart(band_counts)
            .mark_bar()
            .encode(
                x=alt.X("band:N", title="Stress Category"),
                y=alt.Y("count:Q", title="Number of Check-ins"),
                tooltip=["band:N", "count:Q"],
            )
            .properties(height=240)
        )

        st.altair_chart(dist_chart, use_container_width=True)

        # AI-generated HR summary (via LangGraph)
        st.markdown("### 🤖 AI Insight for HR Leaders")

        stress_series = [
            {
                "date": str(row["date"]),
                "stress_score": float(row["stress_score"]),
            }
            for _, row in df.iterrows()
        ]

        try:
            hr_result = run_hr_insights(stress_series)
            hr_summary = hr_result.get("summary", "")
            hr_confidence = hr_result.get("confidence")
            hr_confidence_note = hr_result.get("confidence_note", "")
        except Exception as e:
            hr_summary = f"Could not generate HR summary: {e}"
            hr_confidence = None
            hr_confidence_note = ""

        st.info(hr_summary)

        # Display confidence and uncertainty
        cols = st.columns(2)
        with cols[0]:
            render_confidence(
                provenance=None,
                confidence=hr_confidence,
                key="hr_insights_conf",
            )
        with cols[1]:
            render_uncertainty(
                confidence=hr_confidence,
                key="hr_insights_uncertainty",
                show_explanation=False,
            )
        if hr_confidence_note:
            st.caption(f"📝 {hr_confidence_note}")

        # Explainability view - show AI reasoning process
        num_days = len(set(s["date"] for s in stress_series))
        num_weeks = len(weekly)
        reasoning_steps = [
            {
                "title": "Data Collection",
                "description": f"Analyzed {len(stress_series)} check-ins across {num_days} unique days",
                "confidence": 0.95
            },
            {
                "title": "Trend Aggregation",
                "description": f"Computed weekly averages across {num_weeks} weeks for pattern recognition",
                "confidence": 0.9 if num_days >= 30 else 0.6
            },
            {
                "title": "Statistical Analysis",
                "description": "Identified stress distribution patterns (Low/Medium/High bands) across workforce",
                "confidence": 0.85 if num_days >= 60 else 0.5
            },
            {
                "title": "HR Recommendations",
                "description": f"Generated actionable insights based on {num_days}-day trend data",
                "confidence": hr_confidence if hr_confidence else 0.7
            }
        ]
        render_explainability_view(reasoning_steps, key="hr_insights_reasoning")

        # Feedback widgets
        st.markdown("---")
        render_feedback_widget(
            key="hr_insights_feedback",
            title="Are these HR insights actionable?",
            feedback_type="hr_insights",
            metadata={"confidence": hr_confidence, "data_points": len(stress_series)},
            compact=True
        )

        if hr_confidence is not None:
            render_confidence_accuracy_check(
                key="hr_insights_accuracy",
                confidence=hr_confidence,
                feature_name="HR wellness insights"
            )

        # Action tracker for HR recommendations
        if hr_summary:
            render_action_tracker(
                key="hr_insights_action",
                recommendation=hr_summary[:200],
                feature_name="HR insight"
            )
