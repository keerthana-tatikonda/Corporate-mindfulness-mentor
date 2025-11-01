# graph/nodes.py
from typing import List, Dict, Any, Tuple, Optional
import json
import re
import hashlib, random

from services.llm import client, MODEL
from .schemas import Goal, PlanResponse, SubGoal, DecomposedPlan
from datetime import datetime
import random
import hashlib
from .schemas import DayAdjustment
from services.llm import client, MODEL

from pydantic import ValidationError
import logging

log = logging.getLogger(__name__)

def _to_list(value):
    """Coerce arbitrary JSON-ish values into List[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        # accept comma or newline separated text
        parts = [p.strip() for p in value.replace("\r", "").split("\n")]
        flat = []
        for p in parts:
            flat += [q.strip() for q in p.split(",")]
        return [x for x in flat if x]
    # fallback: single value -> one-item list
    return [str(value).strip()] if str(value).strip() else []



# --- Constants ---
TIME_UNITS: Dict[str, str] = {
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
}

DEFAULT_MILESTONE_TITLES = {
    1: "Foundation & Awareness",
    2: "Building Skills & Consistency",
    3: "Deepening Practice",
    4: "Integration & Mastery",
    5: "Advanced Techniques",
    6: "Sustainable Habits",
}

def _parse_bullets(text: str) -> tuple[List[str], str]:
    """Extract bullet points and summary from an LLM reply."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    bullets = []
    summary_lines = []
    
    for line in lines:
        bullet_match = re.match(r'^(?:[-•*]|\d+\.)\s+(.+)$', line)
        if bullet_match:
            bullets.append(bullet_match.group(1).strip())
        else:
            if bullets or not any(char in line for char in ['-', '•', '*']):
                summary_lines.append(line)
    
    summary = " ".join(summary_lines).strip()
    return bullets, summary

def _parse_structured_response(content: str) -> tuple[List[str], str]:
    """Parse JSON structured output, handling both string arrays and object arrays."""
    try:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            data = json.loads(content)
        
        raw_activities = data.get('activities', [])
        summary = data.get('summary', '')
        
        activities = []
        for item in raw_activities:
            if isinstance(item, str):
                activities.append(item)
            elif isinstance(item, dict):
                if 'activity' in item:
                    activities.append(str(item['activity']))
                elif 'description' in item:
                    activities.append(str(item['description']))
                else:
                    for value in item.values():
                        if isinstance(value, str):
                            activities.append(value)
                            break
            else:
                activities.append(str(item))
        
        return activities, summary
    except (json.JSONDecodeError, AttributeError):
        return _parse_bullets(content)

def _get_fallback_activities(duration_type: str) -> List[str]:
    """Provide context-appropriate fallback activities."""
    base_activities = {
        "daily": [
            "5-minute box breathing exercise before your first meeting",
            "10-minute mindful walk during lunch break",
            "3-minute desk stretching routine mid-afternoon",
            "5-minute body scan meditation before leaving work",
        ],
        "weekly": [
            "20-minute guided meditation session twice per week",
            "30-minute nature walk on weekends",
            "Evening journaling practice (15 minutes, 3x per week)",
            "Weekly digital detox hour (no screens)",
        ],
        "monthly": [
            "Monthly mindfulness workshop or webinar attendance",
            "Weekly stress assessment and reflection (30 minutes)",
            "Bi-weekly yoga or tai chi class",
            "Monthly wellness goal review and adjustment session",
        ]
    }
    return base_activities.get(duration_type, base_activities["daily"])

def _get_fallback_summary(duration_type: str) -> str:
    """Provide context-appropriate fallback summary."""
    summaries = {
        "daily": "A structured daily routine combining breathwork, movement, and mindfulness to build stress resilience.",
        "weekly": "A balanced weekly plan incorporating meditation, nature connection, and reflection for sustained wellbeing.",
        "monthly": "A comprehensive monthly framework for developing long-term mindfulness habits and stress management skills.",
    }
    return summaries.get(duration_type, summaries["daily"])

HORIZON_RE = re.compile(r"\b(\d+)\s*(day|days|week|weeks|month|months)\b", re.IGNORECASE)

def _infer_horizon_from_goal_text(text: str) -> Optional[Tuple[int, str]]:
    """Return (number, unit) if the goal text contains something like '3 weeks'."""
    m = HORIZON_RE.search(text or "")
    if not m:
        return None
    count = int(m.group(1))
    unit_raw = m.group(2).lower()
    unit = TIME_UNITS.get(unit_raw)
    if not unit:
        return None
    return count, unit

def _unit_from_cadence(duration_type: str) -> str:
    """Map dropdown to unit name."""
    if duration_type in ("daily", "day"):
        return "day"
    if duration_type in ("weekly", "week"):
        return "week"
    if duration_type in ("monthly", "month"):
        return "month"
    return "week"

def _label_for(index: int, unit: str) -> str:
    """Return label like 'Week 1', 'Day 3', 'Month 2'."""
    base = {"day": "Day", "week": "Week", "month": "Month"}[unit]
    return f"{base} {index}"

def _get_default_title(week_number: int, total_weeks: int) -> str:
    """Get a meaningful default title based on week number and total."""
    if week_number in DEFAULT_MILESTONE_TITLES:
        return DEFAULT_MILESTONE_TITLES[week_number]
    
    if total_weeks <= 2:
        return ["Getting Started", "Building Momentum"][min(week_number - 1, 1)]
    elif total_weeks == 3:
        return ["Foundation", "Skill Building", "Integration"][min(week_number - 1, 2)]
    elif total_weeks == 4:
        titles = ["Foundation & Awareness", "Building Habits", "Deepening Practice", "Integration"]
        return titles[min(week_number - 1, 3)]
    else:
        third = total_weeks // 3
        if week_number <= third:
            return "Foundation Phase"
        elif week_number <= 2 * third:
            return "Development Phase"
        else:
            return "Mastery Phase"

def _create_unique_seed(goal_name: str, description: str) -> str:
    """Create a unique seed based on goal content to ensure variety."""
    content = f"{goal_name}|{description}|{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    hash_obj = hashlib.md5(content.encode())
    return hash_obj.hexdigest()[:8]

def generate_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a mindfulness plan with unique activities."""
    goal_name = state["goal_name"]
    duration_type = state["duration_type"]
    description = state.get("description", "")
    
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    unique_seed = _create_unique_seed(goal_name, description)
    
    system = (
        "You are a creative, evidence-informed mindfulness coach for corporate employees. "
        "You provide diverse, personalized mindfulness plans in JSON format. "
        "IMPORTANT: Generate unique, specific activities tailored to the user's exact goal. "
        "Consider the user's specific context and create a truly customized plan. "
        "Always respond with valid JSON containing 'activities' (array of strings) and 'summary' (string)."
    )
    
    user_prompt = (
        f'User\'s Specific Goal: "{goal_name}"\n'
        f"Practice Frequency: {duration_type}\n"
    )
    
    if description:
        user_prompt += f"User's Situation: {description}\n"
    
    user_prompt += (
        "\n⚠️ CRITICAL: Create a plan SPECIFICALLY for this goal.\n"
        "Do NOT give generic mindfulness advice.\n\n"
        "Response format:\n"
        '{\n'
        '  "activities": ["Activity 1", "Activity 2", "Activity 3"],\n'
        '  "summary": "Brief explanation"\n'
        '}\n\n'
        f"Session: {unique_seed}-{timestamp}\n"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            top_p=0.95,
            max_tokens=600,
            presence_penalty=0.6,
            frequency_penalty=0.6,
            response_format={"type": "json_object"}
        )

        content = resp.choices[0].message.content or ""
        
        try:
            data = json.loads(content)
            raw_activities = data.get('activities', [])
            summary = data.get('summary', '')
            
            activities = []
            for item in raw_activities:
                if isinstance(item, str):
                    activities.append(item)
                elif isinstance(item, dict):
                    if 'activity' in item:
                        activities.append(str(item['activity']))
                    elif 'description' in item:
                        activities.append(str(item['description']))
                    else:
                        for value in item.values():
                            if isinstance(value, str):
                                activities.append(value)
                                break
                else:
                    activities.append(str(item))
            
        except json.JSONDecodeError:
            activities, summary = _parse_structured_response(content)

        if not activities or len(activities) < 3:
            activities = _get_fallback_activities(duration_type)
        
        if not summary:
            summary = _get_fallback_summary(duration_type)

        activities = [str(act).strip() for act in activities if act][:5]

    except Exception as e:
        print(f"Error generating plan: {e}")
        activities = _get_fallback_activities(duration_type)
        summary = _get_fallback_summary(duration_type)
    
    return {
        **state,
        "activities": activities,
        "summary": summary.strip()
    }

def _fallback_subgoals(goal: Goal, total_count: int = 3) -> List[SubGoal]:
    """Generate meaningful fallback subgoals SPECIFIC to the goal."""
    unit = _unit_from_cadence(goal.duration_type)
    goal_lower = goal.goal_name.lower()
    
    if total_count == 2:
        titles = ["Getting Started", "Building Momentum"]
    elif total_count == 3:
        titles = ["Foundation", "Skill Building", "Integration"]
    elif total_count == 4:
        titles = ["Foundation & Awareness", "Building Habits", "Deepening Practice", "Integration"]
    else:
        titles = [_get_default_title(i+1, total_count) for i in range(total_count)]
    
    # Goal-specific activity patterns
    if "stress" in goal_lower or "anxiety" in goal_lower:
        activity_sets = [
            ["Track your stress triggers in a journal (5 min/day)", "Practice 4-7-8 breathing when stressed (2 minutes)"],
            ["15-minute guided body scan meditation each morning", "Take mindful breaks between tasks (3 minutes each)"],
            ["Progressive muscle relaxation before bed (10 minutes)", "Create a worry-time routine (evening, 15 min)"],
        ]
    elif "focus" in goal_lower or "concentration" in goal_lower or "attention" in goal_lower:
        activity_sets = [
            ["Single-tasking practice: one task at a time (work hours)", "Mindful transition ritual between tasks (2 min)"],
            ["Pomodoro with mindful breaks: 25 work / 5 mindful rest", "Meditation to improve attention (10 min morning)"],
            ["Deep work blocks with intention setting (90 minutes)", "Digital minimalism: scheduled phone checks only"],
        ]
    elif "sleep" in goal_lower or "rest" in goal_lower:
        activity_sets = [
            ["Establish consistent bedtime routine (30 min before sleep)", "No screens 1 hour before bed"],
            ["Body scan meditation for sleep (15 min in bed)", "Gratitude journaling before sleep (5 minutes)"],
            ["Progressive muscle relaxation nightly", "Sleep-friendly environment audit and adjustments"],
        ]
    else:
        activity_sets = [
            [f"Identify specific triggers related to '{goal.goal_name}' (daily observation)", "2-minute mindful pause when trigger occurs"],
            [f"Practice targeted technique for '{goal.goal_name}' (10-15 min daily)", "Reflect on progress in journal (weekly)"],
            [f"Integrate practices into your routine for '{goal.goal_name}'", "Adjust and refine based on what works"],
        ]
    
    subgoals = []
    for i in range(total_count):
        title = titles[i] if i < len(titles) else _get_default_title(i+1, total_count)
        acts = activity_sets[i % len(activity_sets)]
        
        subgoals.append(SubGoal(
            title=title,
            timeframe=_label_for(i+1, unit),
            activities=acts
        ))
    
    return subgoals

def generate_decomposed_plan(goal: Goal, base: PlanResponse) -> DecomposedPlan:
    """
    Create milestone plan with FAST generation and goal-specific activities.
    OPTIMIZED: Reduced token count and simplified prompt for faster response.
    """
    
    # 1) Infer horizon
    inferred = _infer_horizon_from_goal_text(goal.goal_name or "")
    if inferred:
        total_count, horizon_unit = inferred
    else:
        total_count, horizon_unit = 3, _unit_from_cadence(goal.duration_type)

    cadence_unit = _unit_from_cadence(goal.duration_type)
    unique_seed = _create_unique_seed(goal.goal_name, goal.description or "")

    # 2) OPTIMIZED: Shorter, more direct prompt for faster response
    prompt = f"""Create a {total_count}-{horizon_unit} mindfulness plan for: "{goal.goal_name}"

Requirements:
- {total_count} milestones, each with a descriptive title
- 3-4 specific activities per milestone
- Activities must be UNIQUE to this goal
- Progressive difficulty (basic → intermediate → advanced)

Format:
{_label_for(1, horizon_unit)}: <Title>
- Activity 1
- Activity 2
- Activity 3

{_label_for(2, horizon_unit)}: <Title>
- Activity 1
- Activity 2
- Activity 3

End with one sentence explaining the progression.

Goal: "{goal.goal_name}"
ID: {unique_seed}
"""

    try:
        # OPTIMIZED: Reduced max_tokens and timeout for faster response
        chat = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a concise mindfulness coach. Create goal-specific, progressive plans quickly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,  # Slightly lower for faster, more focused responses
            max_tokens=800,   # Reduced from 1200 for speed
            presence_penalty=0.6,
            frequency_penalty=0.6,
            timeout=30,  # Add timeout to prevent hanging
        )
        text = chat.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in decomposition: {e}")
        return DecomposedPlan(
            goal=goal.goal_name,
            subgoals=_fallback_subgoals(goal, total_count),
            duration_type=goal.duration_type,
            ai_summary=f"A progressive plan specifically designed to help you achieve: {goal.goal_name}"
        )

    # 3) Quick parsing
    blocks: List[Tuple[str, List[str]]] = []
    current_title: Optional[str] = None
    current_acts: List[str] = []

    def flush_block():
        nonlocal current_title, current_acts
        if current_title:
            blocks.append((current_title, current_acts[:]))
        current_title, current_acts = None, []

    unit_word = _label_for(1, horizon_unit).split()[0]
    heading_re = re.compile(rf"^(?:#{1,6}\s*)?(?:{unit_word})\s*\d+\s*[:\-—–]\s*(.+)$", re.I)
    bullet_re = re.compile(r"^(?:-|\*|•|\d+[.)])\s+(.+)$")

    explanation_lines: List[str] = []
    in_expl = False

    for ln in [l.strip() for l in text.splitlines() if l.strip()]:
        m = heading_re.match(ln)
        if m:
            if current_title:
                flush_block()
            current_title = m.group(1).strip()
            continue

        b = bullet_re.match(ln)
        if b and current_title:
            current_acts.append(b.group(1).strip())
            continue

        if current_title:
            flush_block()
            in_expl = True
            explanation_lines.append(ln)
        elif in_expl:
            explanation_lines.append(ln)

    if current_title:
        flush_block()

    # 4) Ensure exact count
    if len(blocks) > total_count:
        blocks = blocks[:total_count]
    elif len(blocks) < total_count:
        for i in range(len(blocks), total_count):
            blocks.append((_get_default_title(i + 1, total_count), []))

    # 5) Build SubGoals (simplified deduplication for speed)
    subgoals: List[SubGoal] = []
    all_activities_lower = set()
    
    for i, (title, acts) in enumerate(blocks, start=1):
        # Quick deduplication
        unique = []
        for a in acts:
            a2 = str(a).strip()
            if a2 and a2.lower() not in all_activities_lower:
                unique.append(a2)
                all_activities_lower.add(a2.lower())
        
        # Use smart fallbacks if needed (but prefer LLM output)
        if len(unique) < 3:
            # Get goal-specific fallback activities
            goal_lower = goal.goal_name.lower()
            
            # Define specific activities based on goal type and week
            fallback_activities = []
            
            if "mindfulness" in goal_lower or "plan" in goal_lower:
                fallback_activities = [
                    [f"Morning mindfulness meditation ({5*i} minutes)", 
                     f"Mindful breathing during breaks (3-5 minutes, {i+1}x daily)",
                     f"Evening reflection on mindful moments ({3*i} minutes)"],
                    [f"Body scan practice ({10*i} minutes)", 
                     f"Mindful walking in nature ({15*i} minutes)",
                     f"Gratitude journaling ({5*i} minutes)"],
                    [f"Extended meditation session ({20*i} minutes)", 
                     f"Integrate mindfulness into daily activities (throughout day)",
                     f"Teach someone else a mindfulness technique"],
                ]
            elif "stress" in goal_lower or "anxiety" in goal_lower:
                fallback_activities = [
                    [f"Track stress triggers in a journal ({5*i} min daily)",
                     f"Practice 4-7-8 breathing when stressed (2-3 minutes)",
                     f"Progressive muscle relaxation ({10*i} minutes)"],
                    [f"Guided stress-relief meditation ({15*i} minutes)",
                     f"Take mindful breaks between tasks ({i*2} times daily, 5 min each)",
                     f"Create a worry time routine (evening, {10*i} min)"],
                    [f"Advanced stress management techniques ({20*i} minutes)",
                     f"Develop personal stress-relief toolkit",
                     f"Practice stress reframing exercises"],
                ]
            elif "focus" in goal_lower or "concentration" in goal_lower:
                fallback_activities = [
                    [f"Single-tasking practice (choose {i} task(s) daily)",
                     f"Mindful transition ritual between tasks ({i*2} minutes)",
                     f"Attention training meditation ({10*i} minutes)"],
                    [f"Pomodoro sessions: {20+i*5} min focus / 5 min break",
                     f"Digital minimalism practice (limit phone checks to {4-i} times daily)",
                     f"Deep work block ({60+i*30} minutes)"],
                    [f"Extended focus sessions ({90+i*30} minutes)",
                     f"Eliminate all distractions for focused work periods",
                     f"Flow state cultivation exercises"],
                ]
            elif "sleep" in goal_lower:
                fallback_activities = [
                    [f"Establish consistent bedtime routine ({20+i*10} min before sleep)",
                     f"No screens {i} hour(s) before bed",
                     f"Gentle stretching or yoga ({10*i} minutes)"],
                    [f"Body scan for sleep in bed ({15*i} minutes)",
                     f"Gratitude journaling before sleep ({5*i} minutes)",
                     f"Progressive relaxation technique"],
                    [f"Advanced sleep hygiene practices",
                     f"Sleep environment optimization",
                     f"Consistent wake/sleep schedule (7 days/week)"],
                ]
            else:
                # Generic but still specific fallbacks
                fallback_activities = [
                    [f"Identify specific patterns related to '{goal.goal_name}' (daily observation)",
                     f"Practice targeted technique for your goal ({10*i} min daily)",
                     f"Journal about progress and insights ({5*i} minutes)"],
                    [f"Deepen your practice for '{goal.goal_name}' ({15*i} min daily)",
                     f"Experiment with variations of techniques",
                     f"Reflect on what works best for you"],
                    [f"Integrate '{goal.goal_name}' practices into daily life",
                     f"Make your practice automatic and effortless",
                     f"Share your progress with others or mentor someone"],
                ]
            
            # Get activities for this week (cycle through if needed)
            week_activities = fallback_activities[min(i-1, len(fallback_activities)-1)]
            
            # Add missing activities
            for filler in week_activities:
                if len(unique) >= 5:
                    break
                if filler.lower() not in all_activities_lower:
                    unique.append(filler)
                    all_activities_lower.add(filler.lower())

        final_title = title.strip() if title and title.strip() else _get_default_title(i, total_count)

        subgoals.append(
            SubGoal(
                timeframe=_label_for(i, horizon_unit),
                title=final_title,
                activities=unique[:5],
            )
        )

    ai_summary = " ".join(explanation_lines).strip() or (
        f"Progressive {total_count}-{horizon_unit} plan for '{goal.goal_name}'"
    )

    return DecomposedPlan(
        goal=goal.goal_name, 
        subgoals=subgoals,
        duration_type=goal.duration_type,
        ai_summary=ai_summary,
    )

# --- Morning Check-In Node (additive) ---
# --- Morning Check-In Node (robust & personalized fallback) ---
# # --- Morning Check-In (helpers + node) ---
import os
import json
import logging
import hashlib
import random
from typing import Dict, Any, List
from pydantic import ValidationError

from services.llm import client, MODEL
from .schemas import DayAdjustment

log = logging.getLogger(__name__)

CHECKIN_SYS_PROMPT = (
    "You are a corporate mindfulness mentor. Given a user's morning check-in, "
    "summarize their state, flag risks, and suggest 3-5 concrete, brief actions for today. "
    "Actions must be short, feasible at work, and tied to stress management (breathwork, micro-breaks, reframing). "
    "Reply in strict JSON with keys: summary (str), focus_for_today (list[str]), risk_flags (list[str])."
)

def _to_list(value):
    """Coerce arbitrary JSON-ish values into List[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("\r", "").split("\n")]
        flat: List[str] = []
        for p in parts:
            flat += [q.strip() for q in p.split(",")]
        return [x for x in flat if x]
    s = str(value).strip()
    return [s] if s else []

def _rule_based_fallback(ck: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic, input-sensitive fallback:
    - Seed from (mood, sleep, energy, workload) so ANY change alters the selection.
    - Pick different micro-actions from per-factor pools.
    - Ensure 3–5 total items and at least one per provided field.
    """
    mood = (ck.get("mood") or "").lower().strip()
    sleep = (ck.get("sleep_quality") or "").lower().strip()
    energy = (ck.get("energy") or "").lower().strip()
    workload = (ck.get("workload") or "").lower().strip()
    notes = (ck.get("notes") or "").strip()

    seed_str = f"{mood}|{sleep}|{energy}|{workload}"
    seed = int(hashlib.sha1(seed_str.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)

    mood_pool = {
        "calm": [
            "1-min grounding before first meeting",
            "gratitude note at lunch",
            "slow nasal breathing for 2 min mid-morning",
        ],
        "neutral": [
            "3 mindful breaths before each call",
            "2-min posture reset hourly",
            "note 1 intention for the afternoon",
        ],
        "anxious": [
            "2-min box breathing before hard tasks",
            "3-min cognitive reframe (worry → counter-fact)",
            "progressive muscle relax for 2 min at desk",
        ],
        "frustrated": [
            "4-7-8 breath before sending emails",
            "2-min walk to reset after blockers",
            "write-and-hold draft for 5 min before sending",
        ],
    }
    sleep_pool = {
        "poor": [
            "4-7-8 breathing before first meeting",
            "10-min earlier wind-down tonight",
            "skip caffeine after 2pm",
        ],
        "ok": [
            "1-min mindful breath before lunch",
            "light stretch mid-afternoon",
            "screen break: 20-20-20 twice today",
        ],
        "great": [
            "10-min deep work sprint early morning",
            "2-min energizer breath (in fast, out slow)",
            "share one quick win with teammate",
        ],
    }
    energy_pool = {
        "low": [
            "5-min brisk walk at lunch",
            "water + light snack mid-afternoon",
            "sunlight break for 3 min",
        ],
        "medium": [
            "2×45-min focus sprints (timer on)",
            "micro-stretch every hour",
            "check in with posture at 3pm",
        ],
        "high": [
            "tackle the hardest task first",
            "mentor a teammate for 10 min",
            "end-of-day reflect on 1 learning",
        ],
    }
    workload_pool = {
        "light": [
            "batch messages twice today",
            "finish one small backlog item",
            "organize tomorrow's top 3",
        ],
        "normal": [
            "plan 3 priority tasks (timeboxed)",
            "schedule 3-min break after each block",
            "say no to 1 low-impact ask",
        ],
        "heavy": [
            "90-min deep focus (no notifications)",
            "break tasks into 25-min pomodoros",
            "delegate or defer 1 item",
        ],
    }

    def pick_from(pool_map, key, k=1):
        if not key or key not in pool_map:
            return []
        items = pool_map[key][:]
        rng.shuffle(items)
        return items[:k]

    actions: List[str] = []
    flags: List[str] = []

    actions += pick_from(mood_pool, mood, 1)
    actions += pick_from(sleep_pool, sleep, 1)
    actions += pick_from(energy_pool, energy, 1)
    actions += pick_from(workload_pool, workload, 1)

    if sleep == "poor":
        flags.append("poor sleep")
    if mood in {"anxious", "frustrated"}:
        flags.append("elevated stress")
    if energy == "low":
        flags.append("low energy")
    if workload == "heavy":
        flags.append("heavy workload")

    candidate_pool = (
        mood_pool.get(mood, []) +
        sleep_pool.get(sleep, []) +
        energy_pool.get(energy, []) +
        workload_pool.get(workload, [])
    )
    seen = set()
    dedup: List[str] = []
    for a in actions + candidate_pool:
        key = a.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(a)

    target_min, target_max = 3, 5
    while len(dedup) < target_min and candidate_pool:
        rng.shuffle(candidate_pool)
        cand = candidate_pool.pop(0)
        if cand.lower() not in seen:
            seen.add(cand.lower())
            dedup.append(cand)
    dedup = dedup[:target_max]

    summary_bits: List[str] = []
    if flags:
        summary_bits.append(" • ".join(flags))
    if notes:
        summary_bits.append(f"note: {notes[:80]}")
    summary = "Plan tuned for today" + (f" ({'; '.join(summary_bits)})" if summary_bits else ".")

    return {
        "summary": summary,
        "focus_for_today": dedup,
        "risk_flags": flags
    }

def morning_checkin_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ck = state.get("checkin")
    if not ck:
        return state  # nothing to do

    # If API key is missing, return personalized rule-based fallback
    if not os.environ.get("OPENAI_API_KEY"):
        data = _rule_based_fallback(ck)
        safe_payload = {
            "summary": (str(data.get("summary")).strip() if data.get("summary") is not None else None),
            "focus_for_today": _to_list(data.get("focus_for_today")),
            "risk_flags": _to_list(data.get("risk_flags")),
        }
        try:
            state["day_adjustment"] = DayAdjustment(**safe_payload).model_dump()
        except ValidationError as ve:
            log.error("DayAdjustment validation failed (no API key): %s | payload=%r", ve, safe_payload)
            state["day_adjustment"] = DayAdjustment(
                summary=safe_payload.get("summary") or "Plan adjusted for today.",
                focus_for_today=safe_payload.get("focus_for_today") or ["1-min mindful breath before each meeting"],
                risk_flags=safe_payload.get("risk_flags") or [],
            ).model_dump()
        return state

    user_prompt = (
        f"mood={ck.get('mood')}, sleep_quality={ck.get('sleep_quality')}, "
        f"energy={ck.get('energy')}, workload={ck.get('workload')}\n"
        f"notes={ck.get('notes') or ''}\n\n"
        "Return JSON with: summary, focus_for_today (3-5 items), risk_flags.\n"
        "Each action must explicitly reflect at least one of: mood, sleep_quality, energy, or workload, "
        "and differ if any of these inputs change."
    )

    # Try 1: strict JSON mode
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": CHECKIN_SYS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)

        # Validate the JSON-mode payload; if invalid/empty, force fallback to free-form
        focus_list = _to_list(data.get("focus_for_today"))
        risk_list = _to_list(data.get("risk_flags"))
        summary_txt = (str(data.get("summary")).strip() if data.get("summary") is not None else "")

        if not focus_list:  # key requirement for your UI/tests
            raise ValueError("JSON-mode payload missing non-empty 'focus_for_today'")

        safe_payload = {
            "summary": summary_txt or None,
            "focus_for_today": focus_list,
            "risk_flags": risk_list,
        }

        try:
            state["day_adjustment"] = DayAdjustment(**safe_payload).model_dump()
        except ValidationError as ve:
            log.error("DayAdjustment validation failed (json mode): %s | payload=%r", ve, safe_payload)
            state["day_adjustment"] = DayAdjustment(
                summary=safe_payload.get("summary") or "Plan adjusted for today.",
                focus_for_today=safe_payload.get("focus_for_today") or ["1-min mindful breath before each meeting"],
                risk_flags=safe_payload.get("risk_flags") or [],
            ).model_dump()
        return state

    except Exception as e:
        log.warning("Morning check-in JSON mode failed: %s", e)

    # Try 2: free-form + best-effort JSON extraction
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": CHECKIN_SYS_PROMPT},
                {"role": "user", "content": user_prompt + "\nReply in JSON only."},
            ],
        )
        content = resp.choices[0].message.content or ""
        # handle ```json fenced output
        if content.strip().startswith("```"):
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                content = content[start:end+1]

        data = json.loads(content)
        safe_payload = {
            "summary": (str(data.get("summary")).strip() if data.get("summary") is not None else None),
            "focus_for_today": _to_list(data.get("focus_for_today")),
            "risk_flags": _to_list(data.get("risk_flags")),
        }

        try:
            state["day_adjustment"] = DayAdjustment(**safe_payload).model_dump()
        except ValidationError as ve:
            log.error("DayAdjustment validation failed (free-form): %s | payload=%r", ve, safe_payload)
            state["day_adjustment"] = DayAdjustment(
                summary=safe_payload.get("summary") or "Plan adjusted for today.",
                focus_for_today=safe_payload.get("focus_for_today") or ["1-min mindful breath before each meeting"],
                risk_flags=safe_payload.get("risk_flags") or [],
            ).model_dump()
        return state

    except Exception as e:
        log.error("Morning check-in free-form mode failed: %s", e)

    # Final safety net: deterministic rule-based plan
    data = _rule_based_fallback(ck)
    state["day_adjustment"] = DayAdjustment(
        summary=(str(data.get("summary")).strip() if data.get("summary") is not None else "Plan adjusted for today."),
        focus_for_today=_to_list(data.get("focus_for_today")) or ["1-min mindful breath before each meeting"],
        risk_flags=_to_list(data.get("risk_flags")),
    ).model_dump()
    return state
