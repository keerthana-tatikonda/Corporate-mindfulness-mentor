import random

def get_random_break_message():
    messages = [
        "🌼 Take a mindful moment — close your eyes and breathe deeply for 1 minute.",
        "💧 Hydrate yourself and stretch your neck and shoulders!",
        "🌤 Step away from your screen — a quick walk can refresh your focus.",
        "🧘 Time for a short breathing exercise — inhale calm, exhale stress.",
        "☕ Stand up, stretch your arms, and let your mind reset."
    ]
    return random.choice(messages)
