# services/llm.py
import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini")

# Validate API key
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Please add it to your .env file:\n"
        "OPENAI_API_KEY=your-api-key-here"
    )


# services/llm.py  (add near the top or bottom)

def coaching_style_instructions(mode: str) -> str:
    mode = (mode or "gentle").lower()
    if mode == "passive":
        return (
            "Use a low-autonomy style. Mostly reflect and validate the user's feelings. "
            "Offer at most one optional suggestion, and avoid telling the user what they "
            "must do."
        )
    if mode == "gentle":
        return (
            "Use a gentle, collaborative style. Offer a small number of suggestions "
            "framed as options the user can accept or ignore."
        )
    if mode == "active":
        return (
            "Use an active coaching style. Provide a clear, concrete plan with several "
            "specific steps, while staying respectful of the user's autonomy."
        )
    if mode == "directive":
        return (
            "Use a directive, structured style. Provide a very specific plan with "
            "prioritized steps and timeboxing, but remain kind and non-judgmental."
        )
    return (
        "Use a gentle, collaborative style. Offer a small number of suggestions framed "
        "as options the user can accept or ignore."
    )


def create_http_client() -> httpx.Client:
    """
    Create HTTP client with appropriate configuration.
    
    Note: trust_env=False is used to prevent httpx from reading proxy
    settings from environment variables, which can cause connection issues
    in certain deployment environments (e.g., Streamlit Cloud, corporate networks).
    
    If you need proxy support, set trust_env=True or configure httpx.Client
    with explicit proxy settings.
    """
    return httpx.Client(
        timeout=30.0,  # 30 second timeout
        trust_env=False,  # Ignore system proxy settings
        follow_redirects=True
    )

# Create OpenAI client
_httpx_client = create_http_client()
client = OpenAI(
    api_key=OPENAI_API_KEY, 
    http_client=_httpx_client
)

# Optional: Add client validation
def validate_client():
    """Test the OpenAI client connection."""
    try:
        # Simple test call
        client.models.list()
        return True
    except Exception as e:
        print(f"OpenAI client validation failed: {e}")
        return False