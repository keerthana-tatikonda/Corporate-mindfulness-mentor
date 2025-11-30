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

# ──────────────────────────────────────────────────────────────
# Imports from your project
# ──────────────────────────────────────────────────────────────
from graph.graph import (
    run_goal_creation,
    run_goal_decomposition,
    run_personalized_goal,
    run_workload_adaptation,
    run_morning_checkin,
    run_stress_analytics,
    run_productivity_insights,
)

from services.llm import MODEL, client
from services.storage import save_plan
from services.break_agent import auto_mindfulness_reminder
from services.checkin_storage import save_checkin, load_checkins
from services.productivity_storage import save_productivity, load_productivity



from graph.break_graph import run_break_workflow, run_llm_break_workflow
from graph.schemas import CheckIn

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
# 📊 Stress-Level Tracking Dashboard
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Stress-Level Tracking Dashboard")

all_checkins = load_checkins() if callable(load_checkins) else []

if not all_checkins:
    st.info(
        "No saved Morning Wellness Check-Ins yet. "
        "Log a few days above so we can show daily and weekly stress trends."
    )
else:
    df = build_stress_dataframe(all_checkins)
    max_date = df["date"].max()

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

    with col_daily:
        st.markdown("#### 📈 Daily Stress (0–100)")
        daily_chart = (
            alt.Chart(df_window)
            .mark_line(point=True)
            .encode(
                x="date:T",
                y=alt.Y("stress_score:Q", scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    "date:T",
                    "stress_score:Q",
                    "mood:N",
                    "sleep_quality:N",
                    "energy:N",
                    "workload:N",
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(daily_chart, use_container_width=True)

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
                x="week_start:T",
                y=alt.Y("avg_stress:Q", scale=alt.Scale(domain=[0, 100])),
                tooltip=["week_start:T", "avg_stress:Q"],
            )
            .properties(height=260)
        )
          # 🤖 AI feedback on stress trends + confidence + stored feedback
    st.markdown("### 🤖 Mentor’s View of Your Stress")

    with st.spinner("Analyzing..."):
        # uses your helper from the bottom of the file
        stress_text, stress_conf, stress_conf_note = analyze_stress_trends_with_llm(
            all_checkins
        )

    st.info(stress_text)
    render_confidence(None, stress_conf, key="stress_dash_conf")
    if stress_conf_note:
        st.caption(f"Confidence note: {stress_conf_note}")

    # simple feedback radio – and store the answer
    stress_helpful = st.radio(
        "Was this stress insight helpful?",
        options=["Not sure yet", "Yes, helpful", "No, not really"],
        index=0,
        horizontal=True,
        key="stress_insight_feedback",
    )

    # only store if user actually chooses yes/no
    if stress_helpful != "Not sure yet":
        save_insight_feedback(
            kind="stress_trend",
            date_str=str(max_date.date()),
            helpful=stress_helpful,
        )


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
                "ai_summary": data.get("summary", ""),
                "stress_pattern": data.get("stress_pattern", ""),
                "accomplishments": data.get("accomplishments", ""),
                "challenges": data.get("challenges", ""),
                "growth_highlights": data.get("growth_highlights", ""),
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
# 📉 Productivity vs. Stress Insights
# ──────────────────────────────────────────────────────────────
# -------------------------------------------------------------------
# 📉 Productivity vs. Stress Insights (User Story)
# -------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📉 Productivity vs. Stress Insights")

# ----- Capture productivity for a given day -----
with st.form("productivity_form", clear_on_submit=False):
    colp1, colp2 = st.columns(2)
    with colp1:
        prod_date = st.date_input("Date", value=datetime.today())
    with colp2:
        prod_score = st.slider(
            "Productivity (0–10)",
            min_value=0,
            max_value=10,
            value=7,
            help="How productive did you feel overall on this day?",
        )
    prod_notes = st.text_input(
        "Notes (optional)",
        placeholder="e.g., Deep work in morning; many meetings; context switching, etc.",
    )
    prod_submitted = st.form_submit_button("Save today's productivity")

if prod_submitted:
    try:
        save_productivity(prod_date.isoformat(), prod_score, prod_notes)
        st.success("✅ Productivity entry saved.")
    except Exception as e:
        st.error(f"Could not save productivity entry: {e}")

# ----- Build dataframes + join stress & productivity -----
stress_df = build_stress_dataframe(
    load_checkins() if callable(load_checkins) else []
)
prod_df = build_productivity_dataframe(
    load_productivity() if callable(load_productivity) else []
)
joined_df = build_stress_productivity_join(stress_df, prod_df)

if joined_df.empty:
    st.info(
        "To see productivity vs stress, log at least one Morning Check-In "
        "and one productivity rating on the same date."
    )
else:
    col_scatter, col_time = st.columns(2)

    # Scatter plot: stress vs productivity
    with col_scatter:
        st.markdown("#### 🔍 Daily Stress vs. Productivity")
        scatter = (
            alt.Chart(joined_df)
            .mark_circle(size=80)
            .encode(
                x=alt.X("stress_score:Q", title="Stress (0–100)"),
                y=alt.Y("productivity:Q", title="Productivity (0–10)"),
                color="date:T",
                tooltip=[
                    "date:T",
                    "stress_score:Q",
                    "productivity:Q",
                    "mood:N",
                    "sleep_quality:N",
                    "energy:N",
                    "workload:N",
                    "prod_notes:N",
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(scatter, use_container_width=True)

    # Trend line over time
    with col_time:
        st.markdown("#### 📆 Stress & Productivity Over Time")
        long_df = pd.melt(
            joined_df[["date", "stress_score", "productivity"]],
            id_vars="date",
            value_vars=["stress_score", "productivity"],
            var_name="metric",
            value_name="value",
        )
        trend = (
            alt.Chart(long_df)
            .mark_line(point=True)
            .encode(
                x="date:T",
                y="value:Q",
                color="metric:N",
                tooltip=["date:T", "metric:N", "value:Q"],
            )
            .properties(height=260)
        )
        st.altair_chart(trend, use_container_width=True)

    # 🤖 AI feedback + confidence + user feedback
    st.markdown("### 🤖 Mentor’s Insight on Stress vs. Productivity")

    joined_records = joined_df.to_dict(orient="records")
    with st.spinner("Analyzing how stress is affecting your productivity..."):
        prod_text, prod_conf, prod_conf_note = analyze_productivity_vs_stress_with_llm(
            joined_records
        )

    st.info(prod_text)
    render_confidence(None, prod_conf, key="prod_vs_stress_conf")
    if prod_conf_note:
        st.caption(f"Confidence note: {prod_conf_note}")

    prod_helpful = st.radio(
        "Was this productivity insight helpful?",
        options=["Not sure yet", "Yes, helpful", "No, not really"],
        index=0,
        horizontal=True,
        key="prod_insight_feedback",
    )

    if prod_helpful != "Not sure yet":
        # use the most recent date present in the joined data
        latest_date = joined_df["date"].max()
        save_insight_feedback(
            kind="prod_vs_stress",
            date_str=str(latest_date.date()),
            helpful=prod_helpful,
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

# ──────────────────────────────────────────────────────────────
# 💬 AI Coaching & Support — Mentor Conversations
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 💬 AI Coaching & Support — Mentor Conversations")

if "mentor_history" not in st.session_state:
    st.session_state["mentor_history"] = []

for h in st.session_state["mentor_history"]:
    st.chat_message("user").markdown(h["user"])
    st.chat_message("assistant").markdown(h["assistant"])

user_msg = st.text_input("Tell me what's stressing you out today...")

if user_msg:
    st.chat_message("user").markdown(user_msg)

    try:
        from graph.mentor_graph import run_mentor_conversation

        reply = run_mentor_conversation(
            history=st.session_state["mentor_history"],
            user_message=user_msg,
        )
    except Exception as e:
        reply = f"⚠️ Mentor error: {e}"

    st.chat_message("assistant").markdown(reply)

    st.session_state["mentor_history"].append(
        {
            "user": user_msg,
            "assistant": reply,
        }
    )
