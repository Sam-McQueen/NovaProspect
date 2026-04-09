"""
sierra_prospector/agents/vision_agent.py
=====================================
Vision agent — renders geospatial data as images and sends them to Grok
for expert geological interpretation.

What it does per cell:
  1. Clips DEM to cell bounds
  2. Renders three visualisations:
       - Hillshade        (sun-lit terrain — reveals structure, lineaments, ridges)
       - Slope map        (gradient intensity — identifies benches, fault scarps)
       - Elevation colour (hypsometric tint — altitude zones at a glance)
  3. Sends all three images to Grok with a tightly engineered geology prompt
  4. Writes Grok's interpretation back to the cell as an LLM note

Why three images:
  A single hillshade misses things that slope or colour maps reveal.
  Sending all three gives Grok the same multi-view toolkit a human
  photogeologist would use.

Model behaviour is controlled entirely by ACTIVE_CONFIG in settings.py.
Change AGENT_PHASE from "testing" to "production" to unlock full capability.

Grok model: grok-4-0709 (reasoning model with vision)
API: OpenAI-compatible endpoint at api.x.ai
"""

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import requests
from PIL import Image

from config.settings import (
    GROK_API_KEY, GROK_API_BASE, ACTIVE_CONFIG, RAW_DIR
)
from core.ontology import GridCell, CellStatus, GeologyNote
from core.logger import get_logger
from agents.base_agent import BaseAgent
from agents.terrain_agent import find_dem_files, load_dem_for_bounds

log = get_logger("vision_agent")


# ── API call tracking (cost guard) ───────────────────────────────────────────
_api_call_count = 0

def _check_api_budget():
    global _api_call_count
    max_calls = ACTIVE_CONFIG.get("max_api_calls", 50)
    if _api_call_count >= max_calls:
        raise RuntimeError(
            f"API call budget exhausted ({max_calls} calls). "
            f"Increase max_api_calls in settings.py AGENT_CONFIG or switch to production phase."
        )
    _api_call_count += 1
    log.debug("API call",
              call_number=_api_call_count,
              budget=max_calls,
              remaining=max_calls - _api_call_count)


# ── Image rendering ───────────────────────────────────────────────────────────

def render_hillshade(
    elev: np.ndarray,
    azimuth_deg: float = 315.0,   # NW light source — standard cartographic
    altitude_deg: float = 45.0,
) -> np.ndarray:
    """
    Render a hillshade (simulated sunlit terrain) from elevation data.
    Returns uint8 array (0-255) ready for PIL.

    Hillshade reveals:
    - Structural lineaments (faults, contacts)
    - Ridge and valley morphology
    - Drainage patterns
    - Topographic benches (ancient placer terraces)
    """
    if elev.shape[0] < 3 or elev.shape[1] < 3:
        return np.zeros(elev.shape, dtype=np.uint8)

    # Fill NaN with mean for gradient calculation
    elev_filled = elev.copy()
    elev_filled[~np.isfinite(elev_filled)] = np.nanmean(elev_filled[np.isfinite(elev_filled)])

    # Compute gradients
    dy, dx = np.gradient(elev_filled)

    # Convert sun position to radians
    az_rad  = np.radians(360 - azimuth_deg + 90)
    alt_rad = np.radians(altitude_deg)

    # Hillshade formula
    slope   = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect  = np.arctan2(-dy, dx)
    shaded  = (np.sin(alt_rad) * np.cos(slope) +
               np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))

    # Normalise to 0-255
    shaded = np.clip(shaded, 0, 1)
    return (shaded * 255).astype(np.uint8)


def render_slope_map(elev: np.ndarray, cell_size_m: float) -> np.ndarray:
    """
    Render slope as a greyscale image.
    Dark = flat, bright = steep.

    Slope map reveals:
    - Fault scarps (sharp linear brightness changes)
    - Bench gravels (flat zones mid-slope)
    - Debris fans (radiating slope patterns)
    """
    if elev.shape[0] < 3 or elev.shape[1] < 3:
        return np.zeros(elev.shape, dtype=np.uint8)

    pixel_size = cell_size_m / max(elev.shape)
    dy, dx = np.gradient(
        np.where(np.isfinite(elev), elev, 0),
        pixel_size, pixel_size
    )
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Cap at 60° for display — cliffs are all the same beyond that
    slope_deg = np.clip(slope_deg, 0, 60)
    normalised = slope_deg / 60.0
    return (normalised * 255).astype(np.uint8)


def render_hypsometric(elev: np.ndarray) -> np.ndarray:
    """
    Render elevation as an RGB hypsometric (altitude-coloured) image.

    Colour scheme tuned for Sierra Nevada:
    - Deep blue/green  → low elevation (valleys, foothill rivers)
    - Yellow/tan       → mid elevation (gold belt, 500-1500m)
    - Orange/brown     → upper elevation (granite, 1500-2500m)
    - White/grey       → high peaks (above 2500m)

    This colour scheme visually highlights the 500-1500m gold belt.
    """
    if elev.size == 0:
        return np.zeros((*elev.shape, 3), dtype=np.uint8)

    valid = elev[np.isfinite(elev)]
    if len(valid) == 0:
        return np.zeros((*elev.shape, 3), dtype=np.uint8)

    elev_min = float(np.percentile(valid, 2))
    elev_max = float(np.percentile(valid, 98))

    if elev_max <= elev_min:
        return np.zeros((*elev.shape, 3), dtype=np.uint8)

    # Normalise 0-1
    norm = np.clip((elev - elev_min) / (elev_max - elev_min), 0, 1)
    norm = np.where(np.isfinite(elev), norm, 0)

    # Colour ramp: low=teal, mid-low=green, mid=yellow, mid-high=orange, high=white
    r = np.zeros_like(norm)
    g = np.zeros_like(norm)
    b = np.zeros_like(norm)

    # Low zone (0-0.25): teal → green
    m = (norm >= 0) & (norm < 0.25)
    t = norm[m] / 0.25
    r[m] = 0 + t * 50
    g[m] = 128 + t * 100
    b[m] = 128 - t * 100

    # Mid-low zone (0.25-0.5): green → yellow  ← gold belt
    m = (norm >= 0.25) & (norm < 0.5)
    t = (norm[m] - 0.25) / 0.25
    r[m] = 50 + t * 200
    g[m] = 228 + t * 27
    b[m] = 28 - t * 28

    # Mid-high zone (0.5-0.75): yellow → orange
    m = (norm >= 0.5) & (norm < 0.75)
    t = (norm[m] - 0.5) / 0.25
    r[m] = 250 - t * 50
    g[m] = 200 - t * 140
    b[m] = 0

    # High zone (0.75-1.0): orange → white (peaks)
    m = (norm >= 0.75) & (norm <= 1.0)
    t = (norm[m] - 0.75) / 0.25
    r[m] = 200 + t * 55
    g[m] = 60 + t * 195
    b[m] = 0 + t * 255

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def array_to_png_base64(
    arr: np.ndarray,
    target_size: int = 512,
    mode: str = "L",    # "L" = greyscale, "RGB" = colour
) -> str:
    """
    Convert a numpy array to a base64-encoded PNG string for API transmission.
    Resizes to target_size × target_size for consistent API costs.
    """
    if mode == "RGB":
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")

    # Resize to consistent size
    img = img.resize((target_size, target_size), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Grok API call ─────────────────────────────────────────────────────────────

GEOLOGY_SYSTEM_PROMPT = """You are an expert geologist and remote sensing specialist.

You will receive terrain imagery of one geographic cell rendered from a Digital Elevation Model.
You will see up to three images:
1. HILLSHADE — sun-lit terrain revealing surface morphology
2. SLOPE MAP — gradient intensity across the cell
3. HYPSOMETRIC — elevation zones by colour

Describe what you observe about the terrain — landforms, structural features, drainage patterns, evidence of erosion or deposition, and the likely geological setting. Be factual and concise."""


def call_grok_vision(
    images_b64: List[Dict],   # [{"label": "hillshade", "data": "base64..."}]
    cell: GridCell,
) -> Optional[str]:
    """
    Send rendered images to Grok for geological interpretation.
    Returns the geological analysis as a plain text string.
    """
    if not GROK_API_KEY:
        raise ValueError(
            "GROK_API_KEY not set. Run: export GROK_API_KEY='xai-your-key'\n"
            "Or add it permanently: echo 'export GROK_API_KEY=xai-...' >> ~/.bashrc"
        )

    _check_api_budget()

    cfg = ACTIVE_CONFIG

    # Build the user message with images and cell context
    content = []

    # Cell context — give Grok the numbers the other agents already computed
    context_lines = [
        f"CELL: {cell.tile_id}",
        f"Location: {cell.centroid_lat:.4f}°N, {cell.centroid_lon:.4f}°W",
        f"Cell size: {cell.cell_size_m/1000:.1f}km × {cell.cell_size_m/1000:.1f}km",
    ]
    if cell.terrain:
        t = cell.terrain
        if t.mean_elevation_m:
            context_lines.append(f"Elevation: {t.mean_elevation_m:.0f}m mean ({t.min_elevation_m:.0f}–{t.max_elevation_m:.0f}m)")
        if t.mean_slope_deg:
            context_lines.append(f"Slope: {t.mean_slope_deg:.1f}° mean")
        if t.dominant_aspect:
            context_lines.append(f"Aspect: {t.dominant_aspect}")
    if cell.probability_score is not None:
        context_lines.append(f"Current probability score: {cell.probability_score:.3f}")

    context_lines.append("\nAnalyse the following images for gold prospecting significance:")
    content.append({"type": "text", "text": "\n".join(context_lines)})

    # Add each image
    for img in images_b64:
        content.append({
            "type": "text",
            "text": f"\n[{img['label'].upper()}]"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img['data']}",
                "detail": "high"
            }
        })

    content.append({
        "type": "text",
        "text": (
            "\nDescribe the terrain and geological setting of this cell.\n"
            "LANDFORMS: [physical features visible]\n"
            "STRUCTURE: [linear features, contacts, structural patterns]\n"
            "DRAINAGE: [drainage pattern and characteristics]\n"
            "GEOLOGY: [likely lithology and geological setting]\n"
            "CONFIDENCE: [0.0-1.0]"
        )
    })

    # Build request payload
    payload = {
        "model": cfg["vision_model"],
        "max_tokens": cfg["max_tokens"],
        "messages": [
            {"role": "system", "content": GEOLOGY_SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
    }

    # Web search tool (disabled in testing phase)
    if cfg.get("web_search"):
        payload["tools"] = [{"type": "web_search_preview"}]

    log.info("Calling Grok vision API",
             tile_id=cell.tile_id,
             model=cfg["vision_model"],
             images=len(images_b64),
             max_tokens=cfg["max_tokens"],
             web_search=cfg.get("web_search"))

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type":  "application/json",
    }

    try:
        t0 = time.time()
        response = requests.post(
            f"{GROK_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        elapsed = round(time.time() - t0, 2)

        if response.status_code != 200:
            log.error("Grok API error",
                      status=response.status_code,
                      response=response.text[:500],
                      tile_id=cell.tile_id)
            return None

        data     = response.json()
        analysis = data["choices"][0]["message"]["content"]
        usage    = data.get("usage", {})

        log.info("Grok vision response received",
                 tile_id=cell.tile_id,
                 elapsed_s=elapsed,
                 input_tokens=usage.get("prompt_tokens"),
                 output_tokens=usage.get("completion_tokens"),
                 reasoning_tokens=usage.get("reasoning_tokens"))

        return analysis

    except requests.exceptions.Timeout:
        log.error("Grok API timeout", tile_id=cell.tile_id)
        return None
    except Exception as e:
        log.exception("Grok API call failed", exc=e, tile_id=cell.tile_id)
        return None


def parse_confidence_from_response(text: str) -> float:
    """Extract confidence score from Grok's structured response."""
    import re
    match = re.search(r"CONFIDENCE:\s*([0-9.]+)", text, re.IGNORECASE)
    if match:
        try:
            return min(max(float(match.group(1)), 0.0), 1.0)
        except ValueError:
            pass
    return 0.5


# ── Main agent class ──────────────────────────────────────────────────────────

class VisionAgent(BaseAgent):
    """
    Vision agent — renders DEM data as geological images and
    sends them to Grok for expert interpretation.

    Only runs on cells above ACTIVE_CONFIG["confidence_cutoff"] to control costs.
    All API behaviour is controlled by ACTIVE_CONFIG in settings.py.
    """

    agent_name  = "vision_agent"
    description = "Renders terrain images and gets geological interpretation from Grok"

    def __init__(self, dem_dir: Path = None):
        super().__init__()
        self._dem_files = find_dem_files(dem_dir)
        self._cfg = ACTIVE_CONFIG
        self.log.info("VisionAgent ready",
                      dem_count=len(self._dem_files),
                      phase=self._cfg.get("phase", "testing"),
                      model=self._cfg["vision_model"],
                      max_tokens=self._cfg["max_tokens"],
                      web_search=self._cfg["web_search"],
                      api_budget=self._cfg["max_api_calls"])

        if not GROK_API_KEY:
            self.log.warning(
                "GROK_API_KEY not set — vision agent will fail on API calls. "
                "Run: export GROK_API_KEY='xai-your-key-here'"
            )

    def process_cell(self, cell: GridCell, **kwargs) -> GridCell:
        """
        For one GridCell:
          1. Check if cell is worth the API cost
          2. Load DEM and render three images
          3. Send to Grok with geology prompt
          4. Parse response and write back to cell
        """
        cfg = self._cfg
        bounds = (cell.min_lon, cell.min_lat, cell.max_lon, cell.max_lat)

        # ── Cost gate — skip low-probability cells ────────────────────────────
        cutoff = cfg.get("confidence_cutoff", 0.3)
        if cell.probability_score is not None and cell.probability_score < cutoff:
            self.log.debug("Cell below confidence cutoff — skipping vision",
                           tile_id=cell.tile_id,
                           probability=cell.probability_score,
                           cutoff=cutoff)
            return cell

        self.log.info("Running vision analysis",
                      tile_id=cell.tile_id,
                      probability=cell.probability_score)

        # ── Load DEM ──────────────────────────────────────────────────────────
        elev = load_dem_for_bounds(self._dem_files, bounds)

        if elev is None:
            self.log.warning("No DEM data for vision rendering", tile_id=cell.tile_id)
            return cell

        target_size = cfg.get("image_size_px", 512)

        # ── Render three images ───────────────────────────────────────────────
        images = []

        try:
            hillshade = render_hillshade(elev)
            images.append({
                "label": "hillshade",
                "data":  array_to_png_base64(hillshade, target_size, mode="L")
            })
            self.log.debug("Hillshade rendered", tile_id=cell.tile_id, size=target_size)
        except Exception as e:
            self.log.exception("Hillshade render failed", exc=e, tile_id=cell.tile_id)

        try:
            slope_map = render_slope_map(elev, cell.cell_size_m)
            images.append({
                "label": "slope_map",
                "data":  array_to_png_base64(slope_map, target_size, mode="L")
            })
            self.log.debug("Slope map rendered", tile_id=cell.tile_id)
        except Exception as e:
            self.log.exception("Slope map render failed", exc=e, tile_id=cell.tile_id)

        try:
            hypsometric = render_hypsometric(elev)
            images.append({
                "label": "hypsometric",
                "data":  array_to_png_base64(hypsometric, target_size, mode="RGB")
            })
            self.log.debug("Hypsometric rendered", tile_id=cell.tile_id)
        except Exception as e:
            self.log.exception("Hypsometric render failed", exc=e, tile_id=cell.tile_id)

        if not images:
            self.log.error("No images rendered — skipping API call", tile_id=cell.tile_id)
            return cell

        # ── Call Grok ─────────────────────────────────────────────────────────
        try:
            analysis = call_grok_vision(images, cell)
        except RuntimeError as e:
            # Budget exhausted
            self.log.warning("API budget exhausted", error=str(e))
            return cell

        if not analysis:
            self.log.warning("No analysis returned from Grok", tile_id=cell.tile_id)
            return cell

        # ── Parse and write back ──────────────────────────────────────────────
        confidence = parse_confidence_from_response(analysis)

        note = {
            "note":      analysis,
            "confidence": confidence,
            "timestamp":  __import__("datetime").datetime.utcnow().isoformat(),
            "model":      cfg["vision_model"],
            "agent":      "vision_agent",
        }
        cell.llm_notes.append(note)
        cell.llm_reasoning = analysis

        # Blend vision confidence into probability score
        if cell.probability_score is not None:
            cell.probability_score = round(
                cell.probability_score * 0.5 + confidence * 0.5, 4
            )
        else:
            cell.probability_score = confidence

        cell.opportunity_score = cell.probability_score
        cell.status = CellStatus.COMPLETE

        self.log.info("Vision analysis complete",
                      tile_id=cell.tile_id,
                      confidence=confidence,
                      probability=cell.probability_score,
                      response_length=len(analysis))

        # Print to console so you can read it during testing
        print(f"\n{'='*60}")
        print(f"GROK VISION ANALYSIS — {cell.tile_id}")
        print(f"Location: {cell.centroid_lat:.4f}°N, {cell.centroid_lon:.4f}°W")
        print(f"{'='*60}")
        print(analysis)
        print(f"{'='*60}\n")

        return cell
