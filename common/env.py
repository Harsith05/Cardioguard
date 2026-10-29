"""Load settings from the project's .env file.

Docker Compose reads .env by itself; this makes the Python code read the same file,
so you don't have to set DATABASE_URL in every terminal.

Variables already set in the terminal (or by tests / Docker) take priority over .env.
"""
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv not installed: rely on real environment variables
        return
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)