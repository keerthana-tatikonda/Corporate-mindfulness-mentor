# graph/nodes.py
from typing import List, Dict, Any, Tuple, Optional
import json
import re
from services.llm import client, MODEL
from .schemas import Goal, PlanResponse, SubGoal, DecomposedPlan
from datetime import datetime
import random
import hashlib

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