import os
import re
import json
from datetime import datetime
import streamlit as st

from graph.graph import run_goal_creation, run_goal_decomposition
from services.llm import MODEL
from services.storage import save_plan

# ──────────────────────────────────────────────────────────────────────────────
# Page Setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Corporate Mindfulness Mentor",
    page_icon="🧘",
    layout="centered",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session State Init
# ──────────────────────────────────────────────────────────────────────────────
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

init_session_state()

# ──────────────────────────────────────────────────────────────────────────────
# Persistence Helpers
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# UI: Title & Sidebar (CLEANED UP)
# ──────────────────────────────────────────────────────────────────────────────
st.title("🧘‍♀️ Corporate Mindfulness Mentor")
st.subheader("Goal Creation & Personalized Plan")

with st.sidebar:
    # Only show essential info to users
    st.markdown("### 📊 Your Progress")
    total_plans = len(st.session_state["goal_history"])
    st.metric("Total Plans Created", total_plans)
    
    # Only show API status if there's an issue
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ API Key Missing")
        st.caption("Please add OPENAI_API_KEY to your .env file")
    
    st.markdown("---")

    # Goal History (user-friendly)
    if st.session_state["goal_history"]:
        st.markdown("### 📚 Previous Goals")
        
        for plan in reversed(st.session_state["goal_history"][-10:]):
            with st.expander(f"📋 {plan['name'][:35]}...", expanded=False):
                st.caption(f"Created: {plan['timestamp']}")
                st.caption(f"Duration: {plan['duration'].title()}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📖 Load", key=f"load_{plan['id']}", use_container_width=True):
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
        
        # Export all option
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
    
    # Reset button
    if st.button("🔄 Start Fresh", help="Clear all saved plans", use_container_width=True):
        if st.session_state["goal_history"]:
            # Show confirmation
            st.session_state.clear()
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Main Form
# ──────────────────────────────────────────────────────────────────────────────
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
        placeholder="Share details about your schedule, work environment, or specific challenges...",
        help="More context helps us create a better personalized plan",
        height=100,
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        submitted = st.form_submit_button("✨ Generate My Plan", use_container_width=True, type="primary")
    with col_btn2:
        decompose_request = st.form_submit_button("📅 Break Into Steps", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Handle Form Actions
# ──────────────────────────────────────────────────────────────────────────────
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
                pass  # Silent fail for storage

            st.success("✅ Your personalized plan is ready!")
            st.balloons()

        except Exception as e:
            st.error("😔 Oops! Something went wrong creating your plan.")
            st.caption(f"Error details: {str(e)}")
            st.info("💡 Try again or contact support if the problem persists.")

if decompose_request:
    if st.session_state.get("result") is None:
        st.info("💡 First, generate a plan, then you can break it into weekly steps!")
    else:
        li = st.session_state.get("last_input") or {}
        try:
              li = st.session_state.get("last_input") or {}
              cadence = li.get("duration_type", "weekly")  # "daily" | "weekly" | "monthly"
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

# ──────────────────────────────────────────────────────────────────────────────
# Display Decomposition (Milestones)
# ──────────────────────────────────────────────────────────────────────────────
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
    
    if hasattr(dec, 'ai_summary') and dec.ai_summary:
        st.markdown("---")
        st.markdown("### 🌱 Why This Sequence?")
        st.info(dec.ai_summary)

# ──────────────────────────────────────────────────────────────────────────────
# Display Main Plan (Activities & Summary)
# ──────────────────────────────────────────────────────────────────────────────
result = st.session_state.get("result")
if result:
    st.markdown("---")

    # Header
    st.markdown(f"## 🎯 Your Goal: {result.goal}")
    
    current_id = st.session_state.get("current_goal_id")
    if current_id and current_id in st.session_state["saved_plans"]:
        st.caption(f"Saved as: {current_id}")

    # Activities
    st.markdown("### 📋 Your Daily Practices")
    st.caption("Check off activities as you complete them")
    
    for i, act in enumerate(result.suggested_activities, start=1):
        checkbox_key = f"activity_{current_id}_{i}" if current_id else f"activity_temp_{i}"
        st.checkbox(f"**{i}.** {act}", key=checkbox_key)

    # Summary
    st.markdown("---")
    st.markdown("### 💡 About Your Plan")
    clean_summary = re.sub(r"#+\s*", "", result.ai_summary)
    
    st.markdown(
        f"""<div style='
            background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e9 100%);
            padding: 24px;
            border-radius: 12px;
            border-left: 5px solid #4CAF50;
            font-size: 16px;
            line-height: 1.8;
            margin: 16px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        '>{clean_summary}</div>""",
        unsafe_allow_html=True,
    )

    # Action buttons
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

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
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