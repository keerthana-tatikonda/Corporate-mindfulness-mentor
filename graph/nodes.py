# graph/nodes.py
from typing import List, Dict, Any, Tuple, Optional
import json
import re
import hashlib, random, uuid
import os

from services.llm import client, MODEL
#Gola creation, Profile personalisation and Workload adaptation
from .schemas import (
    Goal, PlanResponse, SubGoal, DecomposedPlan,
    UserProfile, PersonalizedPlanRequest, PersonalizedPlanResponse,
    WorkloadReport, AdaptedPlanResponse
)


from datetime import datetime, time
from .schemas import DayAdjustment
from services.llm import client, MODEL

from pydantic import ValidationError
import logging

log = logging.getLogger(__name__)

def _clip01(x) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))

def _extract_llm_conf(d: dict, key: str = "confidence") -> Optional[float]:
    """Pull a 0..1 confidence if present & sane."""
    if not isinstance(d, dict):
        return None
    v = d.get(key, None)
    if v is None:
        return None
    try:
        return _clip01(v)
    except Exception:
        return None

_TIME_24 = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")

def _heuristic_conf_list(items: List[str], min_ok: int = 3, max_ok: int = 6) -> float:
    """Simple heuristic: more valid, varied items = higher confidence."""
    if not items:
        return 0.15
    n = len(items)
    score = 0.4
    # count “looks like time — ...”
    time_hits = 0
    for s in items:
        s = (s or "").strip()
        # expect 'HH:MM —'
        parts = s.split("—", 1)
        if parts and _TIME_24.match(parts[0].strip()):
            time_hits += 1
    score += 0.15 * min(time_hits, 3)  # up to +0.45 across 3 timed items
    if min_ok <= n <= max_ok:
        score += 0.15
    return _clip01(score)

def _heuristic_conf_subgoals(subgoals: List["SubGoal"]) -> float:
    if not subgoals:
        return 0.2
    n = len(subgoals)
    activity_density = sum(len(sg.activities or []) for sg in subgoals) / max(1, n)
    score = 0.35
    if 3 <= n <= 6:
        score += 0.2
    score += 0.05 * min(activity_density, 6)  # reward populated milestones
    return _clip01(score)


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
    """Generate a mindfulness plan with unique activities (now with confidence)."""
    goal_name = state["goal_name"]
    duration_type = state["duration_type"]
    description = state.get("description", "")
    
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    unique_seed = _create_unique_seed(goal_name, description)
    
    system = (
        "You are a creative, evidence-informed mindfulness coach for corporate employees. "
        "You provide diverse, personalized mindfulness plans in JSON format. "
        "Always respond with valid JSON containing: "
        "'activities' (array of 3-5 strings) and 'summary' (string). "
        "IMPORTANT: Generate unique, specific activities tailored to the user's exact goal. "
        "Consider the user's specific context and create a truly customized plan. "
        "OPTIONAL (if you can estimate): include "
        "'confidence' (float 0..1: how confident you are this plan is appropriate) and "
        "'confidence_note' (short reason for your confidence)."
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
        '  "summary": "Brief explanation",\n'
        '  "confidence": 0.72,                // optional\n'
        '  "confidence_note": "Reason..."     // optional\n'
        '}\n\n'
        f"Session: {unique_seed}-{timestamp}\n"
    )

    activities: List[str] = []
    summary: str = ""
    confidence: Optional[float] = None
    confidence_note: str = ""

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
            response_format={"type": "json_object"},
        )

        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)

            # activities
            raw_activities = data.get('activities', [])
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

            # summary
            summary = data.get('summary', '') or ''

            # confidence (optional from LLM)
            confidence = _extract_llm_conf(data)  # uses your global helper
            confidence_note = (data.get("confidence_note") or "").strip()

        except json.JSONDecodeError:
            # Fallback parse for free-form text
            activities, summary = _parse_structured_response(content)

        # Ensure minimum viable output
        if not activities or len(activities) < 3:
            activities = _get_fallback_activities(duration_type)
        if not summary:
            summary = _get_fallback_summary(duration_type)

        # Heuristic confidence if LLM didn't provide one
        if confidence is None:
            confidence = _heuristic_conf_list(activities, min_ok=3, max_ok=5)

        # Normalize activities
        activities = [str(act).strip() for act in activities if act][:5]

    except Exception as e:
        log.warning("Error generating plan: %s", e)
        activities = _get_fallback_activities(duration_type)
        summary = _get_fallback_summary(duration_type)
        # conservative confidence when fully fallback
        confidence = _heuristic_conf_list(activities, min_ok=3, max_ok=5)
        confidence_note = ""

    return {
        **state,
        "activities": activities,
        "summary": summary.strip(),
        "confidence": confidence,            # float 0..1
        "confidence_note": confidence_note,  # may be ""
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
        log.warning("Error in decomposition: %s", e)
        # Fallback subgoals + confidence (heuristic)
        fb_subgoals = _fallback_subgoals(goal, total_count)
        fb_conf = _heuristic_conf_subgoals(fb_subgoals)
        fb_note = "Confidence estimated from milestone count and activity density."
        try:
            return DecomposedPlan(
                goal=goal.goal_name,
                subgoals=fb_subgoals,
                duration_type=goal.duration_type,
                ai_summary=f"A progressive plan specifically designed to help you achieve: {goal.goal_name}",
                confidence=fb_conf,
                confidence_note=fb_note,
            )
        except TypeError:
            # If your schema doesn't have confidence fields
            return DecomposedPlan(
                goal=goal.goal_name,
                subgoals=fb_subgoals,
                duration_type=goal.duration_type,
                ai_summary=f"A progressive plan specifically designed to help you achieve: {goal.goal_name}",
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

    # --- NEW: confidence (no behavior change to your plan logic) ---
    conf = _heuristic_conf_subgoals(subgoals)
    conf_note = "Confidence estimated from number of milestones and activity density."

    try:
        return DecomposedPlan(
            goal=goal.goal_name, 
            subgoals=subgoals,
            duration_type=goal.duration_type,
            ai_summary=ai_summary,
            confidence=conf,
            confidence_note=conf_note,
        )
    except TypeError:
        # If your DecomposedPlan schema doesn't have confidence fields, fall back to original return
        return DecomposedPlan(
            goal=goal.goal_name, 
            subgoals=subgoals,
            duration_type=goal.duration_type,
            ai_summary=ai_summary,
        )


#Profile personalisation and Workload adaptation


# HH or H, optional :MM, optional am/pm
_TIME_RX = re.compile(r"(?i)\b(\d{1,2})(?::?(\d{2}))?\s*(am|pm)?\b")


def _parse_work_schedule(txt: str) -> tuple[list[str], time, time]:
    """
    Returns (days, start_time, end_time). Defaults to Mon–Fri 09:00–17:00.
    Understands:
      - "Mon-Fri 9-5"
      - "Mon–Fri 09:00-17:30"
      - "Mon-Fri 6pm to 3am"
      - "Mon,Wed,Fri 10-6"
    """
    if not txt:
        return (["Mon","Tue","Wed","Thu","Fri"], time(9,0), time(17,0))

    days = ["Mon","Tue","Wed","Thu","Fri"]
    lower = txt.lower()

    # crude day extraction
    day_tokens = [("mon","Mon"),("tue","Tue"),("tues","Tue"),("wed","Wed"),
                  ("thu","Thu"),("thur","Thu"),("fri","Fri"),("sat","Sat"),("sun","Sun")]
    found = [v for k,v in day_tokens if k in lower]
    if found:
        # keep ordering Mon..Sun but only those found
        order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        days = [d for d in order if d in set(found)]

    # time window – find two times in the string
    # accept separators '-', '–', '—', 'to'
    sep = " to " if " to " in lower else "-"
    if "–" in lower: sep = "–"
    if "—" in lower: sep = "—"

    start, end = time(9,0), time(17,0)
    parts = re.split(r"\s*(?:to|–|—|-)\s*", lower, maxsplit=1)
    if len(parts) == 2:
        a, b = parts
    else:
        a, b = lower, ""

    def _match_time(chunk: str) -> time | None:
        m = _TIME_RX.search(chunk)
        if not m:
            return None
        hh = int(m.group(1))
        mm = int(m.group(2) or 0)
        ampm = (m.group(3) or "").lower()
        if ampm == "pm" and hh < 12:
            hh += 12
        if ampm == "am" and hh == 12:
            hh = 0
        # if no am/pm and looks like “9-5”, guess typical office hours
        return time(hh % 24, mm % 60)

    t1 = _match_time(a)
    t2 = _match_time(b)
    if t1: start = t1
    if t2: end = t2

    return (days, start, end)

# --- label pools and label picker ---
LABEL_POOLS = {
    "breathing": [
        "5-min box breathing",
        "4-7-8 breathing (5 min)",
        "coherent breathing (5 min)",
        "paced breathing (5 min)"
    ],
    "nature": [
        "2-min window gaze + 3-min walk",
        "plant-focus micro-break (5 min)",
        "fresh-air loop (5 min)",
        "brief nature visualization (5 min)"
    ],
    "body_scan": [
        "quiet body scan (5 min)",
        "top-to-toe tension release (5 min)",
        "progressive muscle relax (5 min)",
        "micro body check (5 min)"
    ],
    "journaling": [
        "gratitude journaling (5 min)",
        "3 wins today (5 min)",
        "mind dump + reframe (5 min)",
        "brief reflection (5 min)"
    ],
    "movement": [
        "mindful stretch (5 min)",
        "posture reset + shoulder rolls (5 min)",
        "neck/back mobility (5 min)",
        "desk yoga flow (5 min)"
    ],
}

def _pick_label(modality: str, seed: int) -> str:
    pool = LABEL_POOLS.get(modality, ["5-min mindful pause"])
    rnd = random.Random(seed)
    return rnd.choice(pool)



def _slot(hh: int, mm: int, label: str) -> str:
    return f"{hh:02d}:{mm:02d} — {label}"

def _choose_modalities(prefs: list[str], constraints: list[str]) -> list[str]:
    base = ["breathing","movement","nature","body_scan","journaling"]
    prefs_norm = [p.strip().lower() for p in (prefs or []) if p.strip()]
    # order by user preference first
    ordered = sorted(base, key=lambda m: (prefs_norm.index(m) if m in prefs_norm else 99))
    # simple constraint: “no audio” → avoid body_scan
    if any("no audio" in c.lower() for c in (constraints or [])):
        ordered = [m for m in ordered if m != "body_scan"]
    return ordered or base



def _min_of_day(hh: int, mm: int) -> int:
    return (hh % 24) * 60 + (mm % 60)

def _hm_from_min(m: int) -> tuple[int, int]:
    m = m % (24 * 60)
    return m // 60, m % 60

def _deterministic_timeline(
    schedule: str,
    prefs: List[str],
    constraints: List[str],
    typical_stress: int | None,
    variation_salt: str,
) -> List[str]:
    """
    Build 3–6 time-stamped micro-activities inside the work window with slight randomness.
    - honors 'no audio' by avoiding body_scan
    - jitters times so they don't look identical across runs
    - biases activity type based on stress and prefs
    """
    rnd = random.Random(variation_salt)  # stable randomness per run

    # 1) Parse schedule
    days, start, end = _parse_work_schedule(schedule)
    s = start.hour * 60 + start.minute
    e = end.hour * 60 + end.minute
    span = max(0, (e - s) % (24 * 60)) or 8 * 60  # assume 8h if couldn't parse

    # 2) Pick 4 anchor slots that make sense across the day
    anchors = [s + 45, s + span//3, s + (2*span)//3, s + span - 30]
    # jitter +/- up to 12 min (more jitter if stress higher)
    jitter_max = 6 + int((typical_stress or 5) * 1.2)  # up to ~18 minutes
    anchors = [max(0, a + rnd.randint(-jitter_max, jitter_max)) for a in anchors]

    # 3) Choose modalities using prefs & constraints
    mod_order = _choose_modalities(prefs, constraints)
    # Slight shuffle but keep preference priority
    tail = mod_order[1:]
    rnd.shuffle(tail)
    mod_mix = [mod_order[0]] + tail

    # 4) Labels pool (no audio -> drop body_scan)
    allowed = set(["breathing","nature","body_scan","journaling","movement"])
    if any("no audio" in c.lower() for c in constraints):
        allowed.discard("body_scan")
    mod_mix = [m for m in mod_mix if m in allowed] or ["breathing","movement","nature"]

    out = []
    for i, a in enumerate(anchors):
        hh = (a // 60) % 24
        mm = a % 60
        modality = mod_mix[i % len(mod_mix)]
        label = _pick_label(modality, seed=rnd.randint(0, 10_000))

        # stress-aware duration selection (3–10 min)
        base = 5 + rnd.randint(-1, 2)
        if (typical_stress or 0) >= 7:
            base = min(10, base + 2)  # slightly longer if very stressed

        instruction = {
            "breathing": "Breathe slowly. Inhale 4, hold 4, exhale 6. Keep shoulders relaxed.",
            "nature": "Look away from screens, focus on depth/green tones or take a short walk.",
            "body_scan": "Scan head to toe and relax small areas of tension.",
            "journaling": "Write 1–2 sentences: one feeling + one action you’ll try today.",
            "movement": "Stand up, roll shoulders, neck mobility, soften jaw. Move gently.",
        }.get(modality, "Pause and breathe. Keep it easy and brief.")

        out.append(f"{hh:02d}:{mm:02d} — {base}-min {label}. {instruction}")

    # Occasionally add a 5th item near mid afternoon
    if rnd.random() < 0.35:
        mid = s + (2 * span) // 3 + rnd.randint(-10, 10)
        hh, mm = (mid // 60) % 24, mid % 60
        modality = rnd.choice(mod_mix)
        label = _pick_label(modality, seed=rnd.randint(0, 10_000))
        out.append(f"{hh:02d}:{mm:02d} — 5-min {label}. Keep it light and mindful.")

    return out[:6]

# --- DROP-IN REPLACEMENT: schedule-aware, constraint-aware (more varied) ---
def personalize_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input state:
      - goal_name, duration_type, description
      - profile: UserProfile(work_schedule, typical_stress_level, preferences[], constraints[])
    Output state adds:
      - p_activities: List[str]   (e.g., '09:30 — 5-min box breathing …')
      - p_summary: str
      - source: "LLM" | "fallback"
    """

    goal_name = state["goal_name"]
    duration_type = state["duration_type"]
    description = state.get("description", "")
    profile: UserProfile = state["profile"]

    prefs_list = profile.preferences or []
    cons_list  = profile.constraints or []
    sched = profile.work_schedule or "Mon–Fri 09:00-17:00"
    stress = getattr(profile, "typical_stress_level", None)

    # better salt + varied writing styles to avoid repetitive outputs
    salt  = uuid.uuid4().hex[:8]
    styles = [
        "very concise and practical",
        "gentle and encouraging",
        "energetic and motivational",
        "clinically neutral",
    ]
    style = random.choice(styles)

    system = (
        "You are a corporate mindfulness coach. "
        "Always return JSON with 'timeline' (array of 3–6 items) and 'summary' (string). "
        "Each timeline item MUST include time (HH:MM 24-hour), duration (3–10 minutes), "
        "a short label, and an instruction sentence. "
        "All items must be inside the user's work schedule and must honor constraints "
        "(e.g., if 'no audio', avoid audio-guided activities)."
    )

    user = f"""
Goal: {goal_name}
Cadence: {duration_type}
Context: {description or "-"}
Work schedule: {sched}
Typical stress (0–10): {stress if stress is not None else "-"}
Preferences: {", ".join(prefs_list) or "-"}
Constraints: {", ".join(cons_list) or "-"}
Writing style: {style}
Randomness hint: {salt}

Return ONLY a JSON object like:
{{
  "timeline": [
    {{"time":"09:30","duration_min":5,"label":"box breathing","instructions":"Inhale 4, hold 4, exhale 6…"}}
  ],
  "summary":"Why these fit the profile."
}}
"""

    # --- helper: basic validation/cleanup of LLM items ---
    def _clean_items(timeline: List[Dict[str, Any]]) -> List[str]:
        seen = set()
        cleaned: List[str] = []
        for item in timeline:
            t = str(item.get("time", "")).strip()
            d = item.get("duration_min")
            lbl = str(item.get("label", "")).strip()
            instr = str(item.get("instructions", "")).strip()
            if not t or not lbl:
                continue
            # normalize & dedupe on time+label
            key = (t, lbl.lower())
            if key in seen:
                continue
            seen.add(key)
            dur_txt = f"{int(d)}-min " if isinstance(d, int) else ""
            cleaned.append(f"{t} — {dur_txt}{lbl}. {instr}".strip())
        return cleaned

    # --- Try LLM path first: ask for multiple candidates and pick one at random ---
    data = {}
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.9,          # ↑ variety
            top_p=0.95,               # ↑ diversity
            n=3,                      # request 3 variants
            response_format={"type": "json_object"},
            max_tokens=900,
        )
        # choose one random candidate
        choice = random.choice(resp.choices)
        data = json.loads(choice.message.content or "{}")
    except Exception:
        data = {}

    timeline = data.get("timeline") or []
    summary  = (data.get("summary") or "").strip()

    items = _clean_items(timeline)

    # If model gave too little or nothing, use deterministic timeline (your existing helper)
    if len(items) < 3:
        items = _deterministic_timeline(
            schedule=sched,
            prefs=prefs_list,
            constraints=cons_list,
            typical_stress=stress,
            variation_salt=salt,
        )
        if not summary:
            summary = (
                "A compact timeline inside your work hours, aligned with your "
                "preferences and constraints."
            )

    return {
        **state,
        "p_activities": items[:6],
        "p_summary": summary or "A personalized timeline for your schedule.",
    }



def adapt_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input state:
      - goal_name, duration_type
      - base_activities: List[str]  (what you'd normally do)
      - workload: WorkloadReport {date, meetings, busy_hours, fatigue, blockers[]}
    Output:
      - adapted_plan: List[str] (2–5 items)
      - adapted_rationale: str
      - adapted_source: "LLM" | "fallback"  (for observability)
    """
    goal = state.get("goal_name", "")
    cadence = state.get("duration_type", "")
    base_activities = state.get("base_activities", []) or []
    workload: WorkloadReport = state.get("workload")

    # --- helpers for deterministic fallback (kept local so this function is self-contained) ---
    def _tier(busy_hours: float) -> tuple[int, int]:
        # (target_count, max_dur_min)
        if busy_hours >= 7:  return (2, 5)
        if busy_hours >= 5:  return (3, 7)
        return (5, 10)

    _POOLS = {
        "calm": [
            "3–5 min box breathing (inhale 4, hold 4, exhale 6)",
            "3–5 min grounding (feel both feet, scan contact points)",
            "2–3 min soft-gaze eye break",
            "brief body scan (3–5 min), release jaw/neck/shoulders",
        ],
        "focus": [
            "90-sec pre-task reset: 3 slow breaths + write one next action",
            "2-min posture reset + 60-sec eye rest",
            "2-min single-task pledge: silence notifications and set a 20-min timer",
        ],
        "movement_light": [
            "3–5 min desk stretch: neck, wrists, traps",
            "3–5 min doorway pec stretch + shoulder rolls",
            "3–5 min stand, shake out arms, hip circles",
        ],
        "movement_walk": [
            "5–10 min nature micro-walk",
            "5–8 min corridor loop (nose breathing)",
            "7–10 min outdoor fresh-air walk",
        ],
        "reflection": [
            "2–3 min ‘what went well’ note",
            "3–5 min gratitude jot (3 items)",
            "2–3 min quick de-brief: one lesson, one win",
        ],
    }

    def _modalities(fatigue: str, blockers: List[str]) -> List[str]:
        b = [x.lower() for x in (blockers or [])]
        f = (fatigue or "").lower()
        if f == "high":
            base = ["calm", "focus", "reflection"]
        elif f == "low":
            base = ["movement_walk", "movement_light", "focus", "calm"]
        else:
            base = ["movement_light", "calm", "focus", "reflection"]
        if any("oncall" in x for x in b):
            base = [m for m in base if m != "movement_walk"]
            if "movement_light" not in base:
                base.insert(0, "movement_light")
        return base

    def _hm(mins: int) -> tuple[int, int]:
        mins %= (24 * 60)
        return mins // 60, mins % 60

    def _slot_text(mins: int, label: str) -> str:
        hh, mm = _hm(mins)
        return f"{hh:02d}:{mm:02d} — {label}"

    def _fallback_plan(w: WorkloadReport) -> tuple[List[str], str]:
        busy = float(getattr(w, "busy_hours", 0.0) or 0.0)
        meetings = int(getattr(w, "meetings", 0) or 0)
        fatigue = getattr(w, "fatigue", "") or ""
        blockers = getattr(w, "blockers", []) or []

        target_count, max_dur = _tier(busy)
        mods = _modalities(fatigue, blockers)

        now = datetime.now()
        seed = int(now.strftime("%Y%m%d%H"))
        rnd = random.Random(seed)

        # position items later in the day to respect workload
        anchors = []
        base_m = now.hour * 60 + now.minute
        step = rnd.randint(60, 110)
        for _ in range(target_count):
            base_m += step + rnd.randint(10, 40)
            anchors.append(base_m)

        items = []
        for i, a in enumerate(anchors):
            mod = mods[i % len(mods)]
            label = rnd.choice(_POOLS.get(mod, _POOLS["calm"]))
            if "min" not in label:
                label = f"{max_dur}-min {label}"
            items.append(_slot_text(a, label))

        rationale = (
            f"Given {int(busy)} busy hours, {meetings} meetings, fatigue: {fatigue or 'n/a'}, "
            f"and blockers: {', '.join(blockers) or 'none'}, "
            f"the plan keeps total effort to {target_count} items (≤{max_dur} min each) "
            f"and places them later in the workday. Modalities: {', '.join(mods)}."
        )
        return items, rationale

    # -------------------- LLM-first attempt --------------------
    busy = float(getattr(workload, "busy_hours", 0.0) or 0.0)
    meetings = int(getattr(workload, "meetings", 0) or 0)
    fatigue = getattr(workload, "fatigue", "") or ""
    blockers = getattr(workload, "blockers", []) or []

    target_count, max_dur = _tier(busy)

    system = (
        "You are a structured corporate wellness coach. "
        "Return valid JSON only with keys 'day_plan' (array of 2–6 items) and 'rationale' (string). "
        "Each item must include time (HH:MM 24h), duration_min (int), label (short), and instructions (one sentence). "
        "Use base_activities as inspiration but right-size to today's workload."
    )
    user = {
        "goal": goal,
        "cadence": cadence,
        "busy_hours": busy,
        "meetings": meetings,
        "fatigue": fatigue,
        "blockers": blockers,
        "target_items": target_count,
        "max_duration_min": max_dur,
        "base_activities": base_activities[:6],
        "instruction": "Prefer placing items after midday when hours are heavy."
    }

    adapted_plan: List[str] = []
    adapted_rationale = ""
    source = "fallback"

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
            ],
            temperature=0.8,
            top_p=0.95,
            response_format={"type": "json_object"},
            max_tokens=900,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        js_items = data.get("day_plan") or []
        for it in js_items:
            t   = str(it.get("time", "")).strip()
            dur = it.get("duration_min")
            lab = str(it.get("label", "")).strip()
            ins = str(it.get("instructions", "")).strip()
            if t and lab:
                dur_txt = f"{int(dur)}-min " if isinstance(dur, int) else ""
                adapted_plan.append(f"{t} — {dur_txt}{lab}. {ins}".strip())
        adapted_rationale = (data.get("rationale") or "").strip()
    except Exception:
        # fall through to deterministic
        pass

    # -------------------- deterministic fallback --------------------
    # if the model didn’t give enough to use, fall back
    if len(adapted_plan) < 2:
        adapted_plan, adapted_rationale = _fallback_plan(workload)


    return {
        **state,
        "adapted_plan": adapted_plan[:5],
        "adapted_rationale": adapted_rationale or "Adjusted to match today's workload.",
    }



# --- Morning Check-In Node (additive) ---
# --- Morning Check-In Node (robust & personalized fallback) ---
# # --- Morning Check-In (helpers + node) ---

import logging
import hashlib
from pydantic import ValidationError
from .schemas import DayAdjustment

log = logging.getLogger(__name__)

CHECKIN_SYS_PROMPT = (
    "You are a corporate mindfulness mentor. Given a user's morning check-in, "
    "summarize their state, flag risks, and suggest 3-5 concrete, brief actions for today. "
    "Actions must be short, feasible at work, and tied to stress management (breathwork, micro-breaks, reframing). "
    "Reply in strict JSON with keys: summary (str), focus_for_today (list[str]), risk_flags (list[str])."
)

def _to_list_checkin(value):
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
            "focus_for_today": _to_list_checkin(data.get("focus_for_today")),
            "risk_flags": _to_list_checkin(data.get("risk_flags")),
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
        focus_list = _to_list_checkin(data.get("focus_for_today"))
        risk_list = _to_list_checkin(data.get("risk_flags"))
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
            "focus_for_today": _to_list_checkin(data.get("focus_for_today")),
            "risk_flags": _to_list_checkin(data.get("risk_flags")),
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
