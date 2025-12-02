from dotenv import load_dotenv
import os
from datetime import datetime
from typing import TypedDict

load_dotenv()

# LangGraph
from langgraph.graph import StateGraph, END

# Project services/utilities (unchanged)
from services.break_agent import MindfulBreakAgent
from utils.messages import get_random_break_message
from services.llm import client, MODEL

# Optional: try importing coaching_style_instructions if it exists
try:
    from services.llm import coaching_style_instructions
except ImportError:
    def coaching_style_instructions(mode: str) -> str:
        """Fallback if the function doesn't exist in services.llm"""
        instructions = {
            "passive": "Provide gentle observations without prescribing actions.",
            "gentle": "Offer soft, collaborative suggestions.",
            "active": "Give clear, confident recommendations.",
            "directive": "Provide specific, structured instructions.",
        }
        return instructions.get(mode, instructions["gentle"])

# Rest of imports and code...

class BreakState(TypedDict, total=False):
    message: str
    log_path: str
    result: str
    decision: str
    coach_mode: str  # ✅ ADD THIS

class BreakLLMState(dict):
    """State representation for LLM workflow."""
    message: str
    reflection: str
    recommendation: str
    coach_mode: str

def suggest_break(state):
    """Step 1: Suggest a basic mindfulness message."""
    msg = get_random_break_message()
    coach_mode = state.get("coach_mode", "gentle")  # ✅ EXTRACT FROM STATE
    return {"message": msg, "coach_mode": coach_mode}

def reflect_on_break(state):
    coach_mode = state.get("coach_mode", "gentle")  # ✅ EXTRACT FROM STATE
    style_text = coaching_style_instructions(coach_mode)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        # Fallback shim
        class ChatOpenAI:
            def __init__(self, model: str, temperature: float = 0.7, openai_api_key: str | None = None, **kwargs):
                self._model = model
                self._temperature = temperature
            
            def invoke(self, text: str):
                resp = client.chat.completions.create(
                    model=self._model if self._model else MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": text},
                    ],
                    temperature=self._temperature,
                    max_tokens=180,
                )
                class _R:
                    content = resp.choices[0].message.content or ""
                return _R()
        
        class ChatPromptTemplate:
            def __init__(self, template: str):
                self._template = template
            
            @classmethod
            def from_template(cls, template: str):
                return cls(template)
            
            def format(self, **kwargs) -> str:
                text = self._template
                for k, v in kwargs.items():
                    text = text.replace("{" + k + "}", str(v))
                return text

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_template(
        "You are a workplace mindfulness coach.\n"
        "Current coaching style: {mode}.\n"
        "Style instructions: {style_text}\n\n"
        "The user received this break reminder: '{message}'. "
        "Explain briefly why this is beneficial for mental health and focus."
    )
    reflection = llm.invoke(
        prompt.format(
            mode=coach_mode,
            style_text=style_text,
            message=state["message"],
        )
    )
    return {
        "reflection": getattr(reflection, "content", "") or "",
        "coach_mode": coach_mode,
        "message": state["message"],
    }
# -----------------------------------------------------------------------------
# Runner (single definition; kept teammate behavior)
# -----------------------------------------------------------------------------
def run_break_workflow(scheduled_time: str | None = None) -> str:
    """
    Orchestrates a break:
    - Generates a short suggestion,
    - Records the break at `scheduled_time` (auto) or now (manual).
    """
    agent = MindfulBreakAgent()
    msg = get_random_break_message()  # decoupled from agent methods
    timestamp = scheduled_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return agent.record_break(msg, timestamp=timestamp)

def personalized_recommendation(state):
    coach_mode = state.get("coach_mode", "gentle")  # ✅ EXTRACT FROM STATE
    style_text = coaching_style_instructions(coach_mode)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        # Fallback shim (same as above)
        class ChatOpenAI:
            def __init__(self, model: str, temperature: float = 0.7, openai_api_key: str | None = None, **kwargs):
                self._model = model
                self._temperature = temperature
            
            def invoke(self, text: str):
                resp = client.chat.completions.create(
                    model=self._model if self._model else MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": text},
                    ],
                    temperature=self._temperature,
                    max_tokens=180,
                )
                class _R:
                    content = resp.choices[0].message.content or ""
                return _R()
        
        class ChatPromptTemplate:
            def __init__(self, template: str):
                self._template = template
            
            @classmethod
            def from_template(cls, template: str):
                return cls(template)
            
            def format(self, **kwargs) -> str:
                text = self._template
                for k, v in kwargs.items():
                    text = text.replace("{" + k + "}", str(v))
                return text

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_template(
        "You are a workplace mindfulness coach.\n"
        "Current coaching style: {mode}.\n"
        "Style instructions: {style_text}\n\n"
        "Given this break message '{message}' and reflection '{reflection}', "
        "suggest one short mindfulness activity the user can try now.\n"
        "Keep it consistent with the coaching style."
    )
    rec = llm.invoke(
        prompt.format(
            mode=coach_mode,
            style_text=style_text,
            message=state["message"],
            reflection=state["reflection"],
        )
    )
    return {"recommendation": getattr(rec, "content", "") or ""}

def create_llm_break_graph():
    """Create and compile the LangGraph-based LLM workflow."""
    graph = StateGraph(BreakLLMState)
    graph.add_node("suggest_break", suggest_break)
    graph.add_node("reflect_on_break", reflect_on_break)
    graph.add_node("personalized_recommendation", personalized_recommendation)

    graph.set_entry_point("suggest_break")
    graph.add_edge("suggest_break", "reflect_on_break")
    graph.add_edge("reflect_on_break", "personalized_recommendation")
    graph.add_edge("personalized_recommendation", END)

    return graph.compile()

def run_llm_break_workflow(coach_mode: str = "gentle"):
    """Run the compiled LLM workflow and return final results with confidence.

    Returns a dict like:
    {
        "message": "...",              # brief break reminder
        "reflection": "...",           # why this matters
        "recommendation": "...",       # concrete activity
        "confidence": 0.0-1.0,         # heuristic confidence
        "confidence_note": "..."       # plain-language explanation
    }
    """
    compiled = create_llm_break_graph()
    raw = compiled.invoke({"coach_mode": coach_mode}) or {}  # ✅ PASS coach_mode HERE

    # --- Heuristic confidence so we can show a bar in the UI ---
    def _heuristic_conf_break(out: dict) -> tuple[float, str]:
        msg_len = len((out.get("message") or "").split())
        refl_len = len((out.get("reflection") or "").split())
        rec_len = len((out.get("recommendation") or "").split())

        # Simple scoring: more complete explanations -> higher confidence
        score = 0.35
        if msg_len >= 4:
            score += 0.15
        if refl_len >= 20:
            score += 0.25
        if rec_len >= 12:
            score += 0.15

        # clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        if score >= 0.8:
            note = (
                "High confidence: the reminder, explanation, and activity are all "
                "well-formed and consistent with standard short-break guidance."
            )
        elif score >= 0.5:
            note = (
                "Moderate confidence: this looks like a reasonable suggestion, "
                "but feel free to adapt or ignore it."
            )
        else:
            note = (
                "Lower confidence: treat this as a light suggestion rather than "
                "a strong recommendation."
            )

        return score, note

    conf, note = _heuristic_conf_break(raw if isinstance(raw, dict) else {})

    if not isinstance(raw, dict):
        raw = {}

    raw.setdefault("message", "")
    raw.setdefault("reflection", "")
    raw.setdefault("recommendation", "")
    raw["confidence"] = conf
    raw["confidence_note"] = note
    return raw