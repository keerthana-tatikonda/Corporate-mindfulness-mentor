# tests/conftest.py
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import graph` works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide dummy env so services.llm doesn't raise at import time
os.environ.setdefault("OPENAI_API_KEY", "test-key")   # fake key for tests
os.environ.setdefault("MODEL", "gpt-4o-mini")         # any valid model name
