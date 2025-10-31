from datetime import datetime, timedelta
from pathlib import Path
import json
import streamlit as st

# Optional desktop notification (safe import)
try:
    from plyer import notification
except Exception:
    notification = None


class MindfulBreakAgent:
    def __init__(self, log_path="data/break_log.json"):
        self.log_path = Path(log_path)
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w") as f:
                json.dump([], f, indent=4)

    def record_break(self, message: str, timestamp: str | None = None) -> str:
        """
        Record a mindful break.
        - If `timestamp` is provided (auto reminder), use it.
        - Otherwise, use the current time (manual break).
        """
        scheduled_time = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Load logs
        try:
            with open(self.log_path, "r") as f:
                logs = json.load(f)
        except FileNotFoundError:
            logs = []

        # Append entry
        logs.append({"scheduled_time": scheduled_time, "message": message})

        # Save
        with open(self.log_path, "w") as f:
            json.dump(logs, f, indent=4)

        return f"✅ Break recorded at {scheduled_time}"


def auto_mindfulness_reminder(interval_minutes: int = 60, enable_sound: bool = True):
    """
    Automatically reminds the user every `interval_minutes`.
    - Shows the next scheduled time only after the first manual break.
    - Logs the *scheduled* time when the reminder fires.
    """
    # Lazy import to avoid circular import
    from graph.break_graph import run_break_workflow

    now = datetime.now()

    # Require a first manual break to start the timer
    if "last_reminder_time" not in st.session_state:
        st.caption("💡 Take your first mindful break to start automatic reminders.")
        return

    last_time = st.session_state["last_reminder_time"]
    next_time = last_time + timedelta(minutes=interval_minutes)

    # Show next scheduled time
    st.info(f"⏰ Next break scheduled at **{next_time.strftime('%H:%M')}**")

    # If it’s time, trigger the reminder and log the scheduled time
    if now >= next_time:
        st.warning("🌼 It's time for a mindful break!")
        run_break_workflow(scheduled_time=next_time.strftime("%Y-%m-%d %H:%M:%S"))
        st.session_state["last_reminder_time"] = next_time

        # Optional desktop notification
        if notification:
            try:
                notification.notify(
                    title="Mindful Break",
                    message="Take a short break 🌿 Stretch, breathe, and relax.",
                    timeout=5,
                )
            except Exception:
                pass

        # Optional sound
        if enable_sound:
            st.audio("https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg")
