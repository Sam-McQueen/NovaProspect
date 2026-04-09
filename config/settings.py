"""
sierra_prospector/config/settings.py
=====================================
Central configuration. ALL tunables live here.
"""

import os
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = Path("/mnt/c/Geodata/remote_sensing")
PROCESSED_DIR = DATA_DIR / "processed"
BOUNDARIES_DIR= DATA_DIR / "boundaries"
DB_DIR        = ROOT_DIR / "db"
LOGS_DIR      = ROOT_DIR / "logs"
OUTPUTS_DIR   = ROOT_DIR / "outputs"

# ── Key files ─────────────────────────────────────────────────────────────────
SIERRA_BOUNDARY_GEOJSON = ROOT_DIR / "sierra_nevada.geojson"
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
    "coastal":1,"blue":2,"green":3,"red":4,"nir":5,
    "swir1":6,"swir2":7,"pan":8,"cirrus":9,"tirs1":10,"tirs2":11,
}
SPECTRAL_INDICES = {
    "iron_oxide":      {"bands":("red","blue"),   "high_is_anomalous":True,  "threshold":2.0},
    "hydroxyl":        {"bands":("swir1","swir2"),"high_is_anomalous":True,  "threshold":1.5},
    "ferric_iron":     {"bands":("swir1","nir"),  "high_is_anomalous":True,  "threshold":1.3},
    "gossan":          {"bands":("red","green"),  "high_is_anomalous":True,  "threshold":1.8},
    "ndvi":            {"bands":("nir","red"),    "high_is_anomalous":False, "threshold":0.4},
    "clay_alteration": {"bands":("swir1","nir"),  "high_is_anomalous":True,  "threshold":1.2},
}

# ══════════════════════════════════════════════════════════════════════════════
# MASTER AGENT CONTROL PANEL
# Change AGENT_PHASE to switch ALL agents at once.
#   "testing"    → low cost, strict limits, no web search
#   "production" → full capability
# ══════════════════════════════════════════════════════════════════════════════
AGENT_PHASE = "testing"

AGENT_CONFIG = {
    "testing": {
        "model":             "grok-4.20-0309-reasoning",
        "vision_model":      "grok-4.20-0309-reasoning",
        "max_tokens":        800,
        "reasoning_tokens":  32000,
        "web_search":        False,
        "image_size_px":     512,
        "max_api_calls":     200,
        "verbose_reasoning": False,
        "confidence_cutoff": 0.3,
    },
    "production": {
        "model":             "grok-4.20-0309-reasoning",
        "vision_model":      "grok-4.20-0309-reasoning",
        "max_tokens":        2000,
        "reasoning_tokens":  32000,
        "web_search":        True,
        "image_size_px":     1024,
        "max_api_calls":     99999,
        "verbose_reasoning": True,
        "confidence_cutoff": 0.2,
    },
}
ACTIVE_CONFIG = AGENT_CONFIG[AGENT_PHASE]

# ── Grok API ──────────────────────────────────────────────────────────────────
GROK_API_KEY  = os.environ.get("GROK_API_KEY", "")
GROK_API_BASE = "https://api.x.ai/v1"

# ── Agent runtime ─────────────────────────────────────────────────────────────
MAX_WORKERS          = 4
DB_BATCH_SIZE        = 500
ON_DEMAND_TIMEOUT    = 120
BROKER_POLL_INTERVAL = 1.0
MAX_TASK_RETRIES     = 3

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL       = "INFO"
LOG_TO_FILE     = True
LOG_TO_CONSOLE  = True
LOG_JSON_FORMAT = True
