"""
novaprospect/config/settings.py
================================
Central configuration. ALL tunables live here.
Reads from environment variables (via .env in Docker) with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env (Docker mounts it; override=False so explicit env vars win) ────
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)

# ── Paths (Docker: /app is workdir; cache is a named volume) ─────────────────
DATA_DIR       = ROOT_DIR / "data"
BOUNDARIES_DIR = DATA_DIR / "boundaries"
DB_DIR         = ROOT_DIR / "db"
LOGS_DIR       = ROOT_DIR / "logs"
OUTPUTS_DIR    = ROOT_DIR / "outputs"

# Local cache — ephemeral, Wasabi is source of truth
LOCAL_CACHE_DIR = Path(
    os.path.expanduser(os.environ.get("LOCAL_CACHE_DIR", "~/.sierra-cache"))
)
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Key files ─────────────────────────────────────────────────────────────────
SIERRA_BOUNDARY_GEOJSON = ROOT_DIR / "sierra_nevada.geojson"
BBOX_PATH               = ROOT_DIR / "bbox.geojson"
MAIN_DB_PATH            = DB_DIR / "prospector.duckdb"
COMMS_DB_PATH           = DB_DIR / "comms.duckdb"

# ── CRS ───────────────────────────────────────────────────────────────────────
CRS_WGS84    = "EPSG:4326"
CRS_ANALYSIS = "EPSG:32610"

# ── Grid levels (cell size in metres) ────────────────────────────────────────
RESOLUTION_LEVELS = {
    0: 50_000, 1: 25_000, 2: 10_000, 3: 5_000,
    4:  2_000, 5:  1_000, 6:    500, 7:   250,
    8:    100, 9:     50, 10:    30, 11:   20, 12: 10,
}
DRILL_THRESHOLD     = 0.45
EXCLUSION_THRESHOLD = 0.08

# ── Landsat bands ─────────────────────────────────────────────────────────────
LANDSAT_BANDS = {
    "coastal": 1, "blue": 2, "green": 3, "red": 4, "nir": 5,
    "swir1": 6, "swir2": 7, "pan": 8, "cirrus": 9, "tirs1": 10, "tirs2": 11,
}
SPECTRAL_INDICES = {
    "iron_oxide":      {"bands": ("red", "blue"),    "high_is_anomalous": True,  "threshold": 2.0},
    "hydroxyl":        {"bands": ("swir1", "swir2"), "high_is_anomalous": True,  "threshold": 1.5},
    "ferric_iron":     {"bands": ("swir1", "nir"),   "high_is_anomalous": True,  "threshold": 1.3},
    "gossan":          {"bands": ("red", "green"),   "high_is_anomalous": True,  "threshold": 1.8},
    "ndvi":            {"bands": ("nir", "red"),     "high_is_anomalous": False, "threshold": 0.4},
    "clay_alteration": {"bands": ("swir1", "nir"),   "high_is_anomalous": True,  "threshold": 1.2},
}

# ══════════════════════════════════════════════════════════════════════════════
# MASTER AGENT CONTROL PANEL
# ══════════════════════════════════════════════════════════════════════════════
# Change AGENT_PHASE to switch ALL agents at once.
#   "testing"    → low cost, strict limits, no web search
#   "production" → full capability
#
# Model notes (xAI Grok 4.20 via Responses API):
#   grok-4.20-reasoning                → latest stable single-agent reasoning
#   grok-4.20-multi-agent-beta-0309    → multi-agent orchestration mode
#   grok-4.20-beta-0309-non-reasoning  → fast, no chain-of-thought
#
# The Responses API is at /v1/responses (NOT /v1/chat/completions).
# Use the OpenAI SDK pointed at https://api.x.ai/v1.
# Grok 4.x REJECTS the reasoning_effort parameter — do not send it.
# Grok 4.x is unified multimodal — same model handles text + vision.
# ══════════════════════════════════════════════════════════════════════════════

AGENT_PHASE = os.environ.get("AGENT_PHASE", "testing")

AGENT_CONFIG = {
    "testing": {
        # Single-agent reasoning for all data-collection agents
        "model":             "grok-4.20-reasoning",
        "vision_model":      "grok-4.20-reasoning",
        # Multi-agent orchestrator for the top-level reasoning agent
        "orchestrator_model": "grok-4.20-multi-agent-beta-0309",
        "max_tokens":        16000,    # Reasoning models burn tokens on thought
        "web_search":        False,
        "image_size_px":     512,
        "max_api_calls":     200,
        "verbose_reasoning": False,
        "confidence_cutoff": 0.3,
    },
    "production": {
        "model":             "grok-4.20-reasoning",
        "vision_model":      "grok-4.20-reasoning",
        "orchestrator_model": "grok-4.20-multi-agent-beta-0309",
        "max_tokens":        16000,
        "web_search":        True,
        "image_size_px":     1024,
        "max_api_calls":     99999,
        "verbose_reasoning": True,
        "confidence_cutoff": 0.2,
    },
}
ACTIVE_CONFIG = AGENT_CONFIG[AGENT_PHASE]

# ── Grok API (xAI via Responses API) ─────────────────────────────────────────
GROK_API_KEY  = os.environ.get("GROK_API_KEY", "")
GROK_API_BASE = os.environ.get("GROK_API_BASE", "https://api.x.ai/v1")
LLM_TIMEOUT_S = int(os.environ.get("LLM_TIMEOUT_S", "600"))

# ── Wasabi S3 (source of truth for ALL data) ─────────────────────────────────
# Region is us-west-1. NOT us-east-1. This caused months of bugs.
WASABI_ACCESS_KEY  = os.environ.get("WASABI_ACCESS_KEY", "")
WASABI_SECRET_KEY  = os.environ.get("WASABI_SECRET_KEY", "")
WASABI_REGION      = os.environ.get("WASABI_REGION", "us-west-1")
WASABI_BUCKET_RAW  = os.environ.get("WASABI_BUCKET_RAW", "sierra-geo-harvester")
WASABI_BUCKET_DATA = os.environ.get("WASABI_BUCKET_DATA", "sierra-cell-data")
WASABI_BUCKET_LOGS = os.environ.get("WASABI_BUCKET_LOGS", "sierra-logs")

EMIT_MINERAL_CSV_S3_KEY = os.environ.get(
    "EMIT_MINERAL_CSV_S3_KEY",
    "reference/mineral_grouping_matrix_20230503.csv",
)

# ── Agent runtime ─────────────────────────────────────────────────────────────
MAX_WORKERS          = int(os.environ.get("MAX_WORKERS", "4"))
MAX_CACHE_GB         = int(os.environ.get("MAX_CACHE_GB", "100"))
DB_BATCH_SIZE        = 500
ON_DEMAND_TIMEOUT    = 120
BROKER_POLL_INTERVAL = 1.0
MAX_TASK_RETRIES     = 3

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL       = os.environ.get("LOG_LEVEL", "INFO")
LOG_TO_FILE     = True
LOG_TO_CONSOLE  = True
LOG_JSON_FORMAT = True

# ── Ensure directories exist ─────────────────────────────────────────────────
for d in (DB_DIR, LOGS_DIR, OUTPUTS_DIR, LOCAL_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
