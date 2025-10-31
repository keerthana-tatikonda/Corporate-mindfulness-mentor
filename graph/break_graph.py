from dotenv import load_dotenv
import os
load_dotenv()
from langgraph.graph import StateGraph, END
from typing import TypedDict
from services.break_agent import MindfulBreakAgent
from utils.messages import get_random_break_message
from services.llm import client, MODEL
from datetime import datetime
# LLM imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate






# --- LangGraph State ---
class BreakState(TypedDict):
    message: str
    log_path: str
    result: str


# --- Node Logic ---
def generate_break_node(state: BreakState):
    agent = MindfulBreakAgent()
    msg = state.get("message") or get_random_break_message()
    result = agent.record_break(msg)

    # optional motivational message using OpenAI
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a mindfulness coach offering encouraging messages."
                },
                {
                    "role": "user",
                    "content": f"The user just completed a break: '{msg}'. Give a short encouragement."
                }
            ],
            max_tokens=50,
            temperature=0.8
        )
        encouragement = resp.choices[0].message.content.strip()
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
                {"role": "user", "content": (
                    "The user has been working continuously for an hour on computer tasks. "
                    "Should they take a short mindful break now? Reply yes or no with one reason."
                )}
            ],
            max_tokens=30
        )
        decision = resp.choices[0].message.content.lower()
        state["decision"] = "yes" if "yes" in decision else "no"
    except Exception as e:
        state["decision"] = "yes"
        print("check_focus_node fallback:", e)
    return state



# --- Workflow Builder ---
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



# --- Runner Function (used in app.py) ---
def run_break_workflow(scheduled_time=None):
    """
    Executes a break workflow and records the break with the correct timestamp.
    If scheduled_time is provided (auto reminder), it logs that time;
    otherwise, it logs the current time (manual break).
    """
def run_break_workflow(scheduled_time: str | None = None) -> str:
    """
    Orchestrates a break:
    - Generates a short suggestion,
    - Records the break at `scheduled_time` (auto) or now (manual).
    """
    agent = MindfulBreakAgent()
    msg = get_random_break_message()  # decouple from agent methods

    timestamp = scheduled_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return agent.record_break(msg, timestamp=timestamp)

# ──────────────────────────────────────────────────────────────
# 🤖 LLM-Powered Mindful Break Workflow
# ──────────────────────────────────────────────────────────────

class BreakLLMState(dict):
    """State representation for LLM workflow."""
    message: str
    reflection: str
    recommendation: str


def suggest_break(state):
    """Step 1: Suggest a basic mindfulness message."""
    from utils.messages import get_random_break_message
    msg = get_random_break_message()
    return {"message": msg}


def reflect_on_break(state):
    """Step 2: Reflect on the benefit of this break using LLM."""
    llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template(
        "The user received this break reminder: '{message}'. "
        "Explain briefly why this is beneficial for mental health and focus."
    )
    reflection = llm.invoke(prompt.format(message=state["message"]))
    return {"reflection": reflection.content}


def personalized_recommendation(state):
    """Step 3: Suggest a simple, actionable activity."""
    llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template(
        "Given this break message '{message}' and reflection '{reflection}', "
        "suggest one short mindfulness activity the user can try now."
    )
    rec = llm.invoke(prompt.format(**state))
    return {"recommendation": rec.content}


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



