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

# -----------------------------------------------------------------------------
# Optional LangChain imports with safe shims (no logic change if LC is present)
# -----------------------------------------------------------------------------
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    _HAVE_LC = True
except Exception:
    # SHIMS: keep import-time happy if LC isn't installed locally.
    _HAVE_LC = False

    class ChatOpenAI:  # type: ignore
        """
        Minimal shim so this module can import without langchain_openai.
        Matches the interface used here: ChatOpenAI(...).invoke(text) -> object with .content
        We delegate to the existing OpenAI client to keep behavior as close as possible.
        """
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

    class _PromptTemplateShim:  # type: ignore
        def __init__(self, template: str):
            self._template = template

        @classmethod
        def from_template(cls, template: str):
            return cls(template)

        def format(self, **kwargs) -> str:
            # very simple {var} replacement compatible with how it's used here
            text = self._template
            for k, v in kwargs.items():
                text = text.replace("{" + k + "}", str(v))
            return text

    ChatPromptTemplate = _PromptTemplateShim  # type: ignore

# -----------------------------------------------------------------------------
# LangGraph State for non-LLM break workflow
# -----------------------------------------------------------------------------
class BreakState(TypedDict, total=False):
    message: str
    log_path: str
    result: str
    decision: str  # added as optional because check_focus_node writes it

# -----------------------------------------------------------------------------
# Nodes (unchanged logic)
# -----------------------------------------------------------------------------
def generate_break_node(state: BreakState):
    agent = MindfulBreakAgent()
    msg = state.get("message") or get_random_break_message()
    result = agent.record_break(msg)

    # Optional motivational message using OpenAI (existing behavior)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a mindfulness coach offering encouraging messages.",
                },
                {
                    "role": "user",
                    "content": f"The user just completed a break: '{msg}'. Give a short encouragement.",
                },
            ],
            max_tokens=50,
            temperature=0.8,
        )
        encouragement = (resp.choices[0].message.content or "").strip()
        if encouragement:
            result += f" 💬 {encouragement}"
    except Exception as e:
        print(f"Motivation generation failed: {e}")

    return {**state, "result": result}


def check_focus_node(state: BreakState):
    """Ask LLM whether user should take a break based on last activity."""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a mindfulness productivity coach."},
                {
                    "role": "user",
                    "content": (
                        "The user has been working continuously for an hour on computer tasks. "
                        "Should they take a short mindful break now? Reply yes or no with one reason."
                    ),
                },
            ],
            max_tokens=30,
        )
        decision = (resp.choices[0].message.content or "").lower()
        state["decision"] = "yes" if "yes" in decision else "no"
    except Exception as e:
        state["decision"] = "yes"
        print("check_focus_node fallback:", e)
    return state

# -----------------------------------------------------------------------------
# Workflow Builder (unchanged)
# -----------------------------------------------------------------------------
def create_break_graph():
    workflow = StateGraph(BreakState)
    workflow.add_node("check_focus", check_focus_node)
    workflow.add_node("generate_break", generate_break_node)
    workflow.set_entry_point("check_focus")
    workflow.add_conditional_edges(
        "check_focus",
        lambda s: "generate_break" if s.get("decision") == "yes" else END,
        {"generate_break": "generate_break", "END": END},
    )
    workflow.add_edge("generate_break", END)
    return workflow.compile()

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

# -----------------------------------------------------------------------------
# 🤖 LLM-Powered Mindful Break Workflow (unchanged logic; shims only if LC missing)
# -----------------------------------------------------------------------------
class BreakLLMState(dict):
    """State representation for LLM workflow."""
    message: str
    reflection: str
    recommendation: str

def suggest_break(state):
    """Step 1: Suggest a basic mindfulness message."""
    msg = get_random_break_message()
    return {"message": msg}

def reflect_on_break(state):
    """Step 2: Reflect on the benefit of this break using LLM."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_template(
        "The user received this break reminder: '{message}'. "
        "Explain briefly why this is beneficial for mental health and focus."
    )
    # In LC: .format returns a string; in shim it's the same.
    reflection = llm.invoke(prompt.format(message=state["message"]))
    return {"reflection": getattr(reflection, "content", "") or ""}

def personalized_recommendation(state):
    """Step 3: Suggest a simple, actionable activity."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_template(
        "Given this break message '{message}' and reflection '{reflection}', "
        "suggest one short mindfulness activity the user can try now."
    )
    rec = llm.invoke(prompt.format(**state))
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

def run_llm_break_workflow():
    """Run the compiled LLM workflow and return final results."""
    compiled = create_llm_break_graph()
    return compiled.invoke({})
