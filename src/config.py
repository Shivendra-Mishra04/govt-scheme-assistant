"""Configuration and path management."""
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Windows console UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables
load_dotenv()

# Paths
ROOT = Path(__file__).parent.parent
PDF_PATH = ROOT / "data" / "raw" / "pm-usp-csss-guidelines.pdf"
INDEX_DIR = ROOT / "data" / "index"
SCHEMES_JSON = ROOT / "data" / "schemes.json"

# LLM settings
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:20128/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy-key")
LLM_MODEL = os.getenv("LLM_MODEL", "shiven-ai")
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "shiven-eval")

# Embedding settings
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Retrieval settings
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.35"))
