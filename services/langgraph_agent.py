# services/langgraph_agent.py

import json
from dotenv import load_dotenv
import os
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Add it to your .env file.")

# Configure LLM
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.8, openai_api_key=api_key)

# Define LangGraph state
class MentorState(TypedDict):
    user_goal: str
    user_profile: Dict[str, Any]
    current_stress_level: int
    session_history: List[Dict[str, Any]]
    subtasks: List[str]
    recommended_ids: List[str]
    motivation: str
    reasoning_summary: str

# ------------------------------------------------------
# 🧠 1. Analyze user needs + decompose into subtasks
# ------------------------------------------------------
def analyze_needs(state: MentorState) -> MentorState:
    llm = get_llm()
    prompt = f"""
    You are an AI wellness mentor for corporate employees.

    The user’s goal is: "{state['user_goal']}"
    Their stress level is {state['current_stress_level']}/10.
    Their work context: {state['user_profile']}.

    Decompose this high-level goal into 3 small, actionable subtasks that
    can be achieved with short relaxation techniques (like breathing, meditation,
    or mindfulness). Be concise.
    """
    reasoning = llm.invoke(prompt).content.strip()
    subtasks = [line.strip("-• ") for line in reasoning.split("\n") if line.strip()]
    state["reasoning_summary"] = reasoning
    state["subtasks"] = subtasks
    return state

# ------------------------------------------------------
# 🪷 2. Map subtasks → JSON techniques dynamically
# ------------------------------------------------------
def recommend_techniques(state: MentorState) -> MentorState:
    lib = load_library()
    all_techniques = []
    focus_text = " ".join(state.get("subtasks", [])).lower()
    stress = state["current_stress_level"]

    if "breathe" in focus_text or "anxiety" in focus_text or stress >= 7:
        keywords = ["breathing", "relax", "calm"]
    elif "focus" in focus_text or "concentration" in focus_text:
        keywords = ["focus", "meditation"]
    else:
        keywords = ["body", "awareness", "mindfulness"]

    # Match techniques by keyword
    for cat in lib.categories:
        for item in cat.items:
            text = (item.title + " " + item.description).lower()
            if any(k in text for k in keywords):
                all_techniques.append(item)

    if not all_techniques:
        all_techniques = [cat.items[0] for cat in lib.categories if cat.items]

    # Recommend top 3 techniques
    state["recommended_ids"] = [t.id for t in all_techniques[:3]]
    return state

# ------------------------------------------------------
# 📊 3. Track user progress and adapt the plan
# ------------------------------------------------------
def monitor_progress(state: MentorState) -> MentorState:
    history = state.get("session_history", [])
    if not history:
        state["motivation"] = "Let's start simple: try one short session daily."
        return state

    avg_after = sum(s.get("stress_after", 5) for s in history) / len(history)
    if avg_after > 6:
        state["motivation"] = "Your stress is still high. We'll try shorter sessions next."
    else:
        state["motivation"] = "Nice progress! You’re building a healthy routine."
    return state

# ------------------------------------------------------
# 🔄 4. Adapt plan based on feedback
# ------------------------------------------------------
def adapt_plan(state: MentorState) -> MentorState:
    if state["current_stress_level"] >= 8:
        # Simplify the plan — 1 short technique only
        state["recommended_ids"] = state["recommended_ids"][:1]
    elif state["current_stress_level"] <= 4 and len(state["recommended_ids"]) < 3:
        # Add a mindfulness technique as bonus if user is improving
        lib = load_library()
        for cat in lib.categories:
            for item in cat.items:
                if "mind" in item.title.lower() and item.id not in state["recommended_ids"]:
                    state["recommended_ids"].append(item.id)
                    break
    return state

# ------------------------------------------------------
# 💬 5. Provide motivational feedback (GPT generated)
# ------------------------------------------------------
def provide_motivation(state: MentorState) -> MentorState:
    llm = get_llm()
    prompt = f"""
    The user set a goal: {state['user_goal']}
    Subtasks: {state.get('subtasks', [])}
    Stress: {state['current_stress_level']}/10
    Based on their performance, write one encouraging message (1-2 sentences).
    """
    msg = llm.invoke(prompt).content.strip()
    state["motivation"] = msg
    return state

# ------------------------------------------------------
# ⚙️ 6. Build LangGraph
# ------------------------------------------------------
def build_graph():
    g = StateGraph(MentorState)
    g.add_node("analyze_needs", analyze_needs)
    g.add_node("recommend_techniques", recommend_techniques)
    g.add_node("monitor_progress", monitor_progress)
    g.add_node("adapt_plan", adapt_plan)
    g.add_node("provide_motivation", provide_motivation)

    g.set_entry_point("analyze_needs")
    g.add_edge("analyze_needs", "recommend_techniques")
    g.add_edge("recommend_techniques", "monitor_progress")
    g.add_edge("monitor_progress", "adapt_plan")
    g.add_edge("adapt_plan", "provide_motivation")
    g.add_edge("provide_motivation", END)
    return g.compile(checkpointer=MemorySaver())

_graph = None
def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

# ------------------------------------------------------
# 🚀 7. Entry point for Streamlit (app.py)
# ------------------------------------------------------



def run_mentor_cycle(user_goal: str, stress_level: int, profile: dict, history: list):
    """
    Generates unique mindfulness or breathing techniques dynamically using OpenAI.
    Always returns a Python dictionary (not raw string).

    Now also returns:
      - confidence: float 0..1 (how appropriate the mentor thinks this set is)
      - confidence_note: short explanation of that confidence
      - explanation: short reasoning about why these techniques were chosen
    """

    # --- Autonomy handling -------------------------------------------------
    autonomy_mode = profile.get("autonomy_mode", "Balanced guidance")
    # Normalize for the prompt
    if "Passive" in autonomy_mode:
        autonomy_instruction = (
            "Autonomy mode: PASSIVE — offer 3 distinct options and let the user choose. "
            "Use softer language and avoid sounding prescriptive."
        )
        target_min, target_max = 3, 3
    elif "Directive" in autonomy_mode:
        autonomy_instruction = (
            "Autonomy mode: DIRECTIVE — focus on 1–2 strong recommendations and clearly "
            "highlight what the user should start with first."
        )
        target_min, target_max = 1, 2
    else:
        # Balanced
        autonomy_instruction = (
            "Autonomy mode: BALANCED — suggest 2–3 options, gently steering the user "
            "toward one or two primary techniques."
        )
        target_min, target_max = 2, 3

    # Keep history compact for the model (optional, improves continuity)
    short_history = []
    try:
        for h in (history or [])[-3:]:
            # expect items like {"user": "...", "assistant": "..."} if present
            short_history.append(
                {
                    "user": h.get("user"),
                    "assistant": h.get("assistant"),
                }
            )
    except Exception:
        short_history = []

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0, top_p=0.9)

    prompt = f"""
You are an AI mindfulness mentor helping a corporate employee manage stress and improve focus.

User Goal: {user_goal}
Stress Level: {stress_level}/10
Profile Details: {profile}
Recent Session History (optional, may be empty): {short_history}

{autonomy_instruction}

Generate between {target_min} and {target_max} *unique and diverse* mindfulness or breathing techniques.

Guidelines:
- Make the techniques contextually relevant to the user's goal and stress level.
- Each run should offer different creative variations — avoid repeating the same names or steps.
- If the user goal includes emotional words (e.g., "anxious", "tired", "angry"), personalize the tone and purpose accordingly.
- Include a short motivational comment or reflection in each technique that connects to their goal (e.g., "Since you mentioned {user_goal.lower()}, try...").

Return ONLY valid JSON with this format:
{{
  "techniques": [
    {{
      "title": "string",
      "duration_min": int,
      "description": "string",
      "steps": ["step1", "step2", "step3"],
      "motivation": "string"
    }}
  ],
  "summary": "brief personalized reflection message summarizing the overall mindfulness advice",
  "confidence": 0.0_to_1.0_float_if_you_can_estimate,
  "confidence_note": "short reason for your confidence level",
  "explanation": "2–4 sentences explaining why these techniques were chosen given the goal, stress level, and autonomy mode"
}}
"""

    try:
        response = llm.invoke(prompt)
        text = (response.content or "").strip()

        # Extract JSON payload robustly
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON object found in LLM response.")
        json_text = text[start_idx : end_idx + 1]

        data = json.loads(json_text)

        techniques = data.get("techniques", []) or []
        summary = data.get("summary") or "Stay mindful and consistent!"

        # --- Uncertainty / confidence handling ----------------------------
        raw_conf = data.get("confidence", None)
        try:
            confidence = float(raw_conf) if raw_conf is not None else None
        except Exception:
            confidence = None

        # Heuristic fallback if LLM didn't provide confidence
        if confidence is None:
            # Simple heuristic: more techniques + mid-range stress → moderate confidence,
            # very high stress → slightly lower confidence.
            n = len(techniques)
            base = 0.6
            if target_min <= n <= target_max:
                base += 0.1
            if stress_level >= 8:
                base -= 0.1
            elif 3 <= stress_level <= 6:
                base += 0.05
            confidence = max(0.0, min(1.0, base))

        confidence_note = (data.get("confidence_note") or "").strip()
        if not confidence_note:
            if confidence >= 0.8:
                confidence_note = (
                    "High confidence: techniques match your goal, stress level, and autonomy setting."
                )
            elif confidence >= 0.5:
                confidence_note = (
                    "Moderate confidence: techniques are a good fit, but adjust based on what feels right for you."
                )
            else:
                confidence_note = (
                    "Lower confidence: your inputs are unusual or there is limited context; treat these as gentle suggestions."
                )

        # --- Explainability / reasoning -----------------------------------
        explanation = (data.get("explanation") or "").strip()
        if not explanation:
            # Fallback explanation if the model didn't send one
            explanation = (
                f"These techniques were selected to support your goal "
                f"('{user_goal}') at a stress level of {stress_level}/10 under "
                f"**{autonomy_mode}**. The mentor balances grounding practices "
                f"and gentle activation so they stay feasible during a workday."
            )

        # ✅ Always return a Python dictionary
        return {
            "techniques": techniques,
            "summary": summary,
            "confidence": confidence,
            "confidence_note": confidence_note,
            "explanation": explanation,
        }

    except Exception as e:
        print("⚠️ Error in run_mentor_cycle:", e)
        # Conservative fallback with lower confidence
        return {
            "techniques": [],
            "summary": "I wasn’t able to generate techniques right now. Please try again.",
            "confidence": 0.3,
            "confidence_note": "Fallback response due to an error while generating techniques.",
            "explanation": (
                "The mentor could not complete its normal reasoning process, so no "
                "specific techniques are shown. This is a safe fallback to avoid "
                "guessing when the model is uncertain."
            ),
        }
