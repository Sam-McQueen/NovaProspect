"""
sierra_prospector/agents/structural_agent.py
=====================================
Structural agent — Round 2 pure synthesis agent.

Reads all Round 1 data from the DB for a cell and calls Grok once
for a rigorous geological interpretation. No investigative loops —
that belongs in the top reasoning agent.

Job: take all the numbers and turn them into one coherent geological
     description that the brain can read quickly.
"""

import re
import time
import requests
from datetime import datetime, timezone
from typing import Optional

from config.settings import GROK_API_KEY, GROK_API_BASE, ACTIVE_CONFIG
from core.ontology import GridCell, CellStatus, StructuralSummary
from core.logger import get_logger
from agents.base_agent import BaseAgent

log = get_logger("structural_agent")

_api_call_count = 0

STRUCTURAL_SYSTEM_PROMPT = """You are an expert structural geologist.

You will receive all available geological data for one geographic cell:
terrain morphology, mineral alteration, geochemistry, geophysics,
fault data, historical mine records, and well data.

Your task: synthesise all of this into a concise, rigorous geological
interpretation. Describe what the data collectively suggests about the
geological setting and processes. Be precise and objective.

Respond in exactly this format:
STRUCTURAL_SETTING: [tectonic/structural context in one phrase]
DEFORMATION_STYLE: [dominant structures present]
KEY_OBSERVATIONS: [2-3 sentences synthesising the multi-dataset picture]
NOTABLE_FEATURES: [anything anomalous or worth examining at finer resolution]
STRUCTURAL_SCORE: [0.0-1.0 reflecting structural complexity and data confidence]"""


def build_cell_summary(cell: GridCell) -> str:
    """Build a complete data summary for Grok from all Round 1 fields."""
    lines = [
        f"CELL: {cell.tile_id}",
        f"Location: {cell.centroid_lat:.4f}°N {cell.centroid_lon:.4f}°W",
        f"Cell size: {cell.cell_size_m/1000:.0f}km × {cell.cell_size_m/1000:.0f}km",
    ]

    # Terrain
    if cell.terrain:
        t = cell.terrain
        lines.append("\n[Terrain]")
        if t.mean_elevation_m:
            lines.append(f"  Elevation: {t.mean_elevation_m:.0f}m mean "
                         f"({t.min_elevation_m:.0f}–{t.max_elevation_m:.0f}m)")
        if t.mean_slope_deg:
            lines.append(f"  Slope: {t.mean_slope_deg:.1f}° mean")
        if t.dominant_aspect:
            lines.append(f"  Aspect: {t.dominant_aspect}")
        if t.drainage_density:
            lines.append(f"  Drainage density: {t.drainage_density:.3f}")

    # Hyperspectral mineralogy
    if cell.hyperspectral:
        h = cell.hyperspectral
        lines.append("\n[Mineralogy — EMIT L2B]")
        if h.dominant_mineral_1:
            lines.append(f"  Dominant mineral: {h.dominant_mineral_1} "
                         f"(abundance {h.dominant_mineral_1_abundance})")
        if h.dominant_mineral_2:
            lines.append(f"  Secondary mineral: {h.dominant_mineral_2} "
                         f"(abundance {h.dominant_mineral_2_abundance})")
        if h.alteration_class:
            lines.append(f"  Alteration class: {h.alteration_class}")
        for mineral, attr in [
            ("goethite",  "goethite_score"),
            ("jarosite",  "jarosite_score"),
            ("alunite",   "alunite_score"),
            ("kaolinite", "kaolinite_score"),
            ("chlorite",  "chlorite_score"),
        ]:
            val = getattr(h, attr, None)
            if val:
                lines.append(f"  {mineral}: {val:.3f}")
        if h.grok_note:
            lines.append(f"  Grok note: {h.grok_note}")

    # Point data (geophysics, mines, wells, faults)
    if cell.point_data:
        pd = cell.point_data
        lines.append("\n[Geophysics]")
        if pd.gravity_bouguer_mgal is not None:
            lines.append(f"  Bouguer gravity: {pd.gravity_bouguer_mgal:.2f} mGal "
                         f"({pd.gravity_sample_count} samples)")
        if pd.magnetic_intensity_nt is not None:
            lines.append(f"  Magnetic intensity: {pd.magnetic_intensity_nt:.2f}")
        if pd.fault_count:
            lines.append(f"\n[Faults]")
            lines.append(f"  Faults in cell: {pd.fault_count}")
            if pd.nearest_fault_name:
                lines.append(f"  Nearest: {pd.nearest_fault_name} "
                             f"({pd.nearest_fault_type or '?'}, {pd.nearest_fault_age or '?'})")
            if pd.nearest_fault_m:
                lines.append(f"  Distance: {pd.nearest_fault_m:.0f}m")
            if pd.fault_density_km_per_km2:
                lines.append(f"  Density: {pd.fault_density_km_per_km2:.3f} km/km²")
        if pd.historic_mine_count:
            lines.append(f"\n[Historical Records]")
            lines.append(f"  Historic mines: {pd.historic_mine_count}")
            if pd.nearest_mine_name:
                lines.append(f"  Nearest mine: {pd.nearest_mine_name} "
                             f"({pd.nearest_mine_commodity or 'unknown'})")
            if pd.depletion_score is not None:
                lines.append(f"  Depletion score: {pd.depletion_score:.2f}")
        if pd.borehole_count:
            lines.append(f"\n[Boreholes / Wells]")
            lines.append(f"  Count: {pd.borehole_count}")
            if pd.max_borehole_depth_m:
                lines.append(f"  Max depth: {pd.max_borehole_depth_m:.0f}m")

    # Vector notes (roads, GNIS)
    import json as _json
    for note in cell.llm_notes:
        try:
            data = _json.loads(note.get("note", "{}"))
            if data.get("agent") == "vector_agent":
                gnis = data.get("gnis", {})
                if gnis.get("stream_count"):
                    lines.append(f"\n[Named Features]")
                    lines.append(f"  Streams: {gnis['stream_count']}")
                    if gnis.get("nearest_stream"):
                        lines.append(f"  Nearest stream: {gnis['nearest_stream']}")
                roads = data.get("roads", {})
                if roads.get("paved_road_m"):
                    lines.append(f"\n[Access]")
                    lines.append(f"  Nearest paved road: {roads['paved_road_m']:.0f}m")
                    if roads.get("trail_m"):
                        lines.append(f"  Nearest trail: {roads['trail_m']:.0f}m")
        except Exception:
            pass

    # Prior LLM notes (hyperspectral grok analysis etc)
    agent_notes = [
        n for n in cell.llm_notes
        if n.get("agent") in ("hyperspectral_agent", "vision_agent")
        and n.get("note")
    ]
    if agent_notes:
        lines.append("\n[Prior Agent Analyses]")
        for n in agent_notes[-2:]:   # Last 2 only
            lines.append(f"  [{n.get('agent')}]: {n['note'][:200]}")

    return "\n".join(lines)


def call_grok_structural(summary: str, cell: GridCell) -> Optional[str]:
    global _api_call_count
    if _api_call_count >= ACTIVE_CONFIG.get("max_api_calls", 200):
        log.warning("API budget exhausted")
        return None
    _api_call_count += 1
    if not GROK_API_KEY:
        log.error("GROK_API_KEY not set")
        return None

    try:
        t0 = time.time()
        r  = requests.post(
            f"{GROK_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GROK_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model":      ACTIVE_CONFIG["model"],
                "max_tokens": ACTIVE_CONFIG["max_tokens"],
                "messages": [
                    {"role": "system", "content": STRUCTURAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": summary},
                ],
            },
            timeout=60,
        )
        elapsed = round(time.time() - t0, 2)
        if r.status_code != 200:
            log.error("Grok error", status=r.status_code,
                      response=r.text[:200], tile_id=cell.tile_id)
            return None
        text = r.json()["choices"][0]["message"]["content"]
        log.info("Grok response", tile_id=cell.tile_id, elapsed_s=elapsed,
                 tokens=r.json().get("usage", {}).get("completion_tokens"))
        return text
    except Exception as e:
        log.exception("Grok call failed", exc=e)
        return None


def parse_structural_response(text: str) -> StructuralSummary:
    summary = StructuralSummary(model=ACTIVE_CONFIG["model"])
    for field, attr in [
        ("STRUCTURAL_SETTING", "structural_setting"),
        ("DEFORMATION_STYLE",  "deformation_style"),
        ("KEY_OBSERVATIONS",   "key_observations"),
        ("NOTABLE_FEATURES",   "notable_features"),
    ]:
        m = re.search(rf"{field}:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            setattr(summary, attr, m.group(1).strip())
    m = re.search(r"STRUCTURAL_SCORE:\s*([0-9.]+)", text, re.IGNORECASE)
    if m:
        try:
            summary.structural_score = min(max(float(m.group(1)), 0.0), 1.0)
        except ValueError:
            pass
    return summary


class StructuralAgent(BaseAgent):
    """
    Round 2 pure synthesis agent.
    One Grok call per cell. Reads all Round 1 data, writes one clean summary.
    No investigative loops — that's the brain's job.
    """

    agent_name  = "structural_agent"
    description = "Synthesises all Round 1 data into one geological interpretation per cell"

    def __init__(self):
        super().__init__()
        self.log.info("StructuralAgent ready",
                      model=ACTIVE_CONFIG["model"],
                      phase=ACTIVE_CONFIG.get("verbose_reasoning"))

    def process_cell(self, cell: GridCell, **kwargs) -> GridCell:

        has_data = any([
            cell.terrain       is not None,
            cell.spectral      is not None,
            cell.point_data    is not None,
            cell.hyperspectral is not None,
        ])
        if not has_data:
            self.log.debug("No Round 1 data — skipping", tile_id=cell.tile_id)
            return cell

        summary  = build_cell_summary(cell)
        response = call_grok_structural(summary, cell)

        if not response:
            self.log.warning("No response from Grok", tile_id=cell.tile_id)
            return cell

        structural = parse_structural_response(response)
        cell.structural = structural

        if structural.structural_score is not None:
            prior = cell.probability_score or 0.0
            cell.probability_score = round(
                prior * 0.6 + structural.structural_score * 0.4, 4
            )
            cell.opportunity_score = cell.probability_score

        cell.llm_notes.append({
            "note":       response,
            "confidence": structural.structural_score or 0.5,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "model":      ACTIVE_CONFIG["model"],
            "agent":      "structural_agent",
        })

        print(f"\n{'='*60}")
        print(f"STRUCTURAL — {cell.tile_id} | score={structural.structural_score}")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")

        self.log.info("Complete",
                      tile_id=cell.tile_id,
                      setting=structural.structural_setting,
                      score=structural.structural_score)
        return cell
