# Read .env before any module reads environment variables.
from common.env import load_env

load_env()
