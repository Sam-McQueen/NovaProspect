"""
sierra_prospector/agents/history_agent.py
=====================================
History agent — FINAL agent in the pipeline.

Runs after ALL other agents. Reads:
    - Complete cell ontology (all prior agent outputs)
    - 1800s historical mining documents (PDFs, text files)
    - GLO survey records
    - MLRS mining claim data

Produces:
    - Final depletion score (combining point_data depletion + historical evidence)
    - Historical context note (what was found here, when, how much)
    - Adjusted opportunity score

Why last:
    Depletion assessment is meaningless without knowing what the probability
    score is first. A cell with probability 0.9 and depletion 0.8 still has
    opportunity score 0.18 — worth noting but not drilling.
    A cell with probability 0.3 and depletion 0.9 is truly dead.
    The history agent makes the final call on opportunity.

Historical document processing:
    Uses Grok to read 1800s PDFs and extract:
    - Location references (creek names, township/range)
    - Production estimates (ounces, dollars at period prices)
    - Mining method (hydraulic, drift, lode, placer)
    - Depletion indicators ("worked out", "abandoned", "low grade")
"""

import json
import time
import requests
from pathlib import Path
from typing import Optional, List

from config.settings import GROK_API_KEY, GROK_API_BASE, ACTIVE_CONFIG, RAW_DIR
from core.ontology import GridCell, CellStatus, HistorySummary
from core.logger import get_logger
from agents.base_agent import BaseAgent

log = get_logger("history_agent")

_api_call_count = 0

HISTORY_SYSTEM_PROMPT = """You are a historical researcher and geologist specialising in California mining history and land use records.

You will receive a geological summary of a geographic cell and excerpts from historical documents relevant to that location.

Your job is to extract factual historical information about this area — what activities occurred here, when, and to what extent. Report what the records actually say without interpretation or speculation.

Respond in this format:
HISTORICAL_ACTIVITY: [what activities are recorded — mining, agriculture, logging, other]
TIME_PERIOD: [when activities occurred]
EXTENT: [scale of recorded activity — localised/moderate/extensive/unknown]
LAND_STATUS: [current or historical land status if noted]
HISTORICAL_NOTES: [2-3 sentences summarising what the records show]"""


def find_historical_documents(data_dir: Path = None) -> List[Path]:
    """Find historical mining documents."""
    search = data_dir or RAW_DIR
    docs = []
    for pattern in ["**/*1800*.txt", "**/*historical*.txt", "**/*mining_history*.txt",
                    "**/*GLO*.txt", "**/*survey*.txt", "**/*report*.txt",
                    "historical/**/*.txt", "historical/**/*.pdf"]:
        docs.extend(search.glob(pattern))
    log.info("Historical documents found", count=len(docs))
    return docs


def _read_text_file(path: Path) -> Optional[str]:
    """Read a text file safely."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.warning("Could not read file", file=str(path), error=str(e))
        return None


def _find_relevant_excerpts(
    docs: List[Path],
    cell_lat: float,
    cell_lon: float,
    max_chars: int = 2000
) -> str:
    """
    Extract relevant excerpts from historical documents.
    Naive approach: look for place names and coordinate references.
    Returns up to max_chars of relevant text.
    """
    # Creek/place names near this lat/lon — very rough heuristic
    # A proper implementation would use NER and geocoding
    excerpts = []
    total = 0

    for doc in docs[:10]:   # Limit to 10 docs for token budget
        text = _read_text_file(doc)
        if not text:
            continue

        # Look for latitude-nearby content (crude proximity filter)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in
                   ["gold","placer","hydraulic","drift","lode","mine",
                    "ounce","oz","oz.","worked","abandoned","rich","poor"]):
                excerpt = "\n".join(lines[max(0,i-1):i+3])
                excerpts.append(f"[{doc.name}]: {excerpt}")
                total += len(excerpt)
                if total > max_chars:
                    break
        if total > max_chars:
            break

    return "\n\n".join(excerpts) if excerpts else "No historical documents available."


def call_grok_history(cell_summary: str, historical_text: str,
                      cell: GridCell) -> Optional[str]:
    global _api_call_count
    max_calls = ACTIVE_CONFIG.get("max_api_calls", 200)
    if _api_call_count >= max_calls:
        return None
    _api_call_count += 1

    cfg = ACTIVE_CONFIG
    if not GROK_API_KEY:
        log.error("GROK_API_KEY not set")
        return None

    user_content = (
        f"CURRENT GEOLOGICAL ASSESSMENT:\n{cell_summary}\n\n"
        f"HISTORICAL DOCUMENT EXCERPTS:\n{historical_text}"
    )

    payload = {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "messages": [
            {"role": "system", "content": HISTORY_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    }

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type":  "application/json",
    }

    try:
        response = requests.post(
            f"{GROK_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if response.status_code != 200:
            log.error("Grok API error",
                      status=response.status_code,
                      tile_id=cell.tile_id)
            return None
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.exception("Grok history call failed", exc=e, tile_id=cell.tile_id)
        return None


def parse_history_response(text: str) -> HistorySummary:
    import re
    summary = HistorySummary(model=ACTIVE_CONFIG["model"])

    m = re.search(r"DEPLETION_SCORE:\s*([0-9.]+)", text, re.IGNORECASE)
    if m:
        try:
            summary.depletion_score = min(max(float(m.group(1)), 0.0), 1.0)
        except ValueError:
            pass

    m = re.search(r"DEPLETION_REASON:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        summary.depletion_reason = m.group(1).strip()

    m = re.search(r"HISTORICAL_NOTES:\s*(.+?)(?:OPPORTUNITY|$)", text,
                  re.IGNORECASE | re.DOTALL)
    if m:
        summary.historical_notes = m.group(1).strip()

    return summary


class HistoryAgent(BaseAgent):
    """
    Final pipeline agent. Reads all prior data + historical documents.
    Sets the definitive depletion score and opportunity score.
    """

    agent_name  = "history_agent"
    description = "Final agent — historical context, depletion assessment, opportunity scoring"

    def __init__(self, data_dir: Path = None):
        super().__init__()
        self._historical_docs = find_historical_documents(data_dir)
        self.log.info("HistoryAgent ready",
                      historical_docs=len(self._historical_docs),
                      model=ACTIVE_CONFIG["model"])

    def process_cell(self, cell: GridCell, **kwargs) -> GridCell:
        # ── Start with point_data depletion if available ──────────────────────
        base_depletion = 0.0
        if cell.point_data and cell.point_data.depletion_score is not None:
            base_depletion = cell.point_data.depletion_score

        # ── Get historical excerpts for this location ──────────────────────────
        historical_text = _find_relevant_excerpts(
            self._historical_docs,
            cell.centroid_lat,
            cell.centroid_lon,
        )

        # ── Call Grok if we have any context at all ────────────────────────────
        cell_summary = cell.to_llm_prompt()
        response = call_grok_history(cell_summary, historical_text, cell)

        if response:
            history = parse_history_response(response)

            # Blend with point_data depletion
            if history.depletion_score is not None:
                history.depletion_score = round(
                    max(history.depletion_score, base_depletion), 3
                )
            else:
                history.depletion_score = base_depletion

            cell.history = history

            # Final opportunity score
            if cell.probability_score is not None and history.depletion_score is not None:
                cell.opportunity_score = round(
                    cell.probability_score * (1 - history.depletion_score * 0.7), 4
                )

            print(f"\n{'='*60}")
            print(f"HISTORY ANALYSIS — {cell.tile_id}")
            print(f"{'='*60}")
            print(response)
            print(f"Final opportunity score: {cell.opportunity_score:.3f}")
            print(f"{'='*60}\n")

            self.log.info("History analysis complete",
                          tile_id=cell.tile_id,
                          depletion=history.depletion_score,
                          opportunity=cell.opportunity_score)
        else:
            # No LLM response — use point_data depletion only
            cell.history = HistorySummary(
                depletion_score  = base_depletion,
                depletion_reason = "estimated_from_point_data",
            )
            if cell.probability_score is not None:
                cell.opportunity_score = round(
                    cell.probability_score * (1 - base_depletion * 0.7), 4
                )

        return cell
