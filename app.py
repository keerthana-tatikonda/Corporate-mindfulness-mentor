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

    render_confidence(
        provenance=None,
        confidence=getattr(dec, "confidence", None),
        key="decomposition_conf",
    )
    note = getattr(dec, "confidence_note", None)
    if note:
        st.caption(f"Confidence note: {note}")

# ──────────────────────────────────────────────────────────────
# Display Main Plan (Activities & Summary)
# ──────────────────────────────────────────────────────────────
result = st.session_state.get("result")

if result:
    st.markdown("---")

    st.markdown(f"## 🎯 Your Goal: {result.goal}")
    render_confidence(None, getattr(result, "confidence", None), key="plan_conf")
    plan_conf_note = getattr(result, "confidence_note", None)
    if plan_conf_note:
        st.caption(f"Confidence note: {plan_conf_note}")

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
            render_confidence(
                None,
                getattr(p, "confidence", None),
                key="profile_conf",
            )
            note = getattr(p, "confidence_note", None)
            if note:
                st.caption(f"Confidence note: {note}")
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
            render_confidence(
                provenance=None,
                confidence=getattr(adapted, "confidence", None),
                key="adapt_conf",
            )
            note = getattr(adapted, "confidence_note", None)
            if note:
                st.caption(f"Confidence note: {note}")
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
    llm_output = run_llm_break_workflow()
    st.markdown("### 🌼 AI-Powered Mindful Suggestion")
    st.info(llm_output["message"])
    st.success(llm_output["reflection"])
    st.caption(f"💡 {llm_output['recommendation']}")


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

    # 🔮 Pure AI-based adjustment with confidence (no rule-based fallback)
    try:
        adj = run_morning_checkin(ck)
    except Exception as e:
        st.error(f"Could not adjust based on check-in: {e}")
    else:
        # ✅ Save check-in + AI adjustment for dashboards
        try:
            save_checkin(ck.model_dump(), adj.model_dump())
        except Exception as e:
            st.warning(
                f"Check-in saved only for this session (storage error: {e})"
            )

        # 🤖 AI feedback section
        st.markdown("### 🤖 AI Feedback on Today’s Check-In")

        if getattr(adj, "summary", None):
            st.success(adj.summary)

        # Show AI confidence bar
        render_confidence(
            provenance=None,
            confidence=getattr(adj, "confidence", None),
            key="checkin_conf",
        )
        note = getattr(adj, "confidence_note", None)
        if note:
            st.caption(f"Confidence note: {note}")

        # Focus + risk flags
        if getattr(adj, "focus_for_today", None):
            st.markdown("**Focus for today**")
            for a in adj.focus_for_today:
                st.write(f"- {a}")

        if getattr(adj, "risk_flags", None):
            st.caption("Flags: " + ", ".join(adj.risk_flags))






# ──────────────────────────────────────────────────────────────
# 🌬 Guided Meditations & Breathing Exercises
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🌬 Guided Meditations & Breathing Exercises")

init_session()

st.markdown("### 🧭 Personalize Your Session")
user_goal = st.text_input("Describe your current state or need:", "")
stress_level = st.slider("Stress Level", 0, 10, 0)

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
        render_confidence(
            provenance=None,
            confidence=analytics_dict.get("confidence"),
            key="stress_analytics_conf",
        )
        conf_note = analytics_dict.get("confidence_note")
        if conf_note:
            st.caption(f"📊 Confidence note: {conf_note}")
        
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
            
            # 🎯 Display confidence
            render_confidence(
                provenance=None,
                confidence=insights_dict.get("confidence"),
                key="productivity_insights_conf",
            )
            conf_note = insights_dict.get("confidence_note")
            if conf_note:
                st.caption(f"📊 Confidence note: {conf_note}")
            
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



analyze_week = st.button("✨ Analyze and Save Weekly Reflection", key="analyze_week_btn")

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
                            "growth_highlights, action_suggestions (list of strings)."
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
            entry = {
                "week_ending": week_ending.isoformat(),
                "raw_text": reflection_text.strip(),
                "ai_summary": _normalize_text_block(data.get("summary")),
                "stress_pattern":  _normalize_text_block(data.get("stress_pattern")),
                "accomplishments": _normalize_text_block(data.get("accomplishments")),
                "challenges":  _normalize_text_block(data.get("challenges")),
                "growth_highlights": _normalize_text_block(data.get("growth_highlights")),
                "action_suggestions": data.get("action_suggestions", []),
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

# Keep mentor conversation history in session_state
if "mentor_history" not in st.session_state:
    st.session_state["mentor_history"] = []

# 1) Show previous conversation from history
for idx, h in enumerate(st.session_state["mentor_history"]):
    st.chat_message("user").markdown(h.get("user", ""))
    st.chat_message("assistant").markdown(h.get("assistant", ""))

    # show confidence per turn if present
    if "confidence" in h:
        render_confidence(
            provenance=None,
            confidence=h.get("confidence"),
            key=f"mentor_conf_{idx}",
        )
        if h.get("confidence_note"):
            st.caption(h["confidence_note"])

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
    if st.button("✨ Generate motivational message"):
        try:
            msg = run_motivation_message(
                completed=done,
                total=total,
                activities=completed_tasks,
            )
        except Exception as e:
            msg = (
                "You're making meaningful progress. Even one completed practice is a real step "
                f"toward lower stress. (AI motivation unavailable: {e})"
            )

        st.success(msg)

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
    if st.button("📊 Generate HR wellness insights"):
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
        # Streamlit: replace use_container_width with width="stretch"
        st.altair_chart(weekly_chart, width="stretch")

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

        st.altair_chart(dist_chart, width="stretch")

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
            hr_summary = run_hr_insights(stress_series)
        except Exception as e:
            hr_summary = f"Could not generate HR summary: {e}"

        st.info(hr_summary)
