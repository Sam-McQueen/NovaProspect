"""
sierra_prospector/core/comms.py
=====================================
Two-way communication broker between the reasoning LLM and domain agents.

This is what enables the reasoning agent to say:
  "I need a detailed spectral analysis of quadrant Z8_R120_C340 RIGHT NOW"
  "Also compute terrain features for the 4 cells surrounding it"
  "And flag Z7_R60_C170 as EXCLUDED — I've determined it's all hydraulic tailings"

The broker:
  1. Receives requests from the reasoning LLM (or any orchestrator)
  2. Routes them to the appropriate agent
  3. Runs agents synchronously (blocking) or asynchronously (fire-and-forget)
  4. Returns results to the caller in LLM-consumable format

Design:
  - Synchronous path: reasoning LLM calls request_cell() and blocks until done
  - Async path: reasoning LLM calls enqueue_request() and polls later
  - Both paths write results back to the DB
"""

import json
import threading
import time
from typing import Optional, Dict, Any, List, Callable

from config.settings import ON_DEMAND_TIMEOUT, BROKER_POLL_INTERVAL
from core.database import db
from core.ontology import GridCell, CellStatus, GeologyNote
from core.grid import grid
from core.logger import get_logger
from datetime import datetime, timezone

log = get_logger("comms_broker")


class RequestBroker:
    """
    The central nervous system connecting the reasoning LLM to the agent layer.

    Usage from reasoning LLM / orchestrator:

        broker = RequestBroker()
        broker.register_agent("spectral_agent",  spectral_agent_instance)
        broker.register_agent("terrain_agent",   terrain_agent_instance)

        # Synchronous — blocks until result ready
        cell = broker.request_cell("Z8_R120_C340", agents=["spectral_agent"])

        # Get LLM-consumable summary
        print(cell.to_llm_prompt())

        # Write LLM reasoning back to the cell
        broker.write_llm_note(
            tile_id    = "Z8_R120_C340",
            note       = "Strong argillic alteration coincides with mapped fault intersection. High priority.",
            confidence = 0.82,
            model      = "claude-opus-4-6"
        )

        # Exclude a cell
        broker.exclude_cell("Z7_R60_C170", reason="Confirmed hydraulic tailings — no lode potential")
    """

    def __init__(self):
        self._agents: Dict[str, Any] = {}   # agent_name → BaseAgent instance
        self._lock = threading.Lock()
        log.info("RequestBroker initialised")

    # ── Agent registry ────────────────────────────────────────────────────────

    def register_agent(self, name: str, agent):
        """Register a domain agent so the broker can route requests to it."""
        with self._lock:
            self._agents[name] = agent
        log.info("Agent registered", agent=name)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    # ── Core request methods ──────────────────────────────────────────────────

    def request_cell(
        self,
        tile_id: str,
        agents: Optional[List[str]] = None,
        infer_missing: bool = True,
        timeout_s: float = ON_DEMAND_TIMEOUT,
    ) -> Optional[GridCell]:
        """
        Synchronously enrich and return a GridCell.

        Parameters:
            tile_id:        The cell to fetch/enrich
            agents:         Which agents to run. None = all registered agents
            infer_missing:  If True, also generates inferred attributes from available data
                            (e.g. estimates geochemistry from spectral proxies when USGS
                            data is absent). This is the 'infer any additional useful data'
                            the user requested.
            timeout_s:      Max seconds to wait for all agents

        Returns the enriched GridCell, or None if tile_id invalid.
        """
        log.info("Cell requested",
                 tile_id=tile_id,
                 agents=agents,
                 infer_missing=infer_missing)

        # ── Ensure cell exists in DB ──────────────────────────────────────────
        cell = db.get_cell(tile_id)
        if cell is None:
            log.info("Cell not in DB yet — building from grid", tile_id=tile_id)
            try:
                level, row, col = grid.parse_tile_id(tile_id)
                cell = grid.build_cell(level, row, col)
                db.upsert_cell(cell)
            except Exception as e:
                log.error("Failed to build cell from tile_id", tile_id=tile_id, error=str(e))
                return None

        # ── Determine which agents to run ─────────────────────────────────────
        target_agents = agents or list(self._agents.keys())
        missing_agents = [a for a in target_agents if a not in self._agents]
        if missing_agents:
            log.warning("Requested agents not registered",
                        missing=missing_agents,
                        available=list(self._agents.keys()))
            target_agents = [a for a in target_agents if a in self._agents]

        # ── Run each agent in sequence (could parallelise later) ──────────────
        t0 = time.time()
        for agent_name in target_agents:
            if time.time() - t0 > timeout_s:
                log.warning("Timeout reached during on-demand enrichment",
                            tile_id=tile_id,
                            completed_agents=target_agents[:target_agents.index(agent_name)])
                break

            agent = self._agents[agent_name]
            log.info("Running agent on-demand",
                     tile_id=tile_id,
                     agent=agent_name)
            try:
                cell = agent._safe_process(cell)
                db.upsert_cell(cell)
            except Exception as e:
                log.exception("On-demand agent failed",
                              exc=e,
                              tile_id=tile_id,
                              agent=agent_name)

        # ── Optional inference pass ───────────────────────────────────────────
        if infer_missing:
            cell = self._infer_missing_attributes(cell)
            db.upsert_cell(cell)

        log.info("On-demand enrichment complete",
                 tile_id=tile_id,
                 duration_s=round(time.time() - t0, 2),
                 status=cell.status)
        return cell

    def request_neighborhood(
        self,
        tile_id: str,
        radius: int = 1,
        agents: Optional[List[str]] = None,
    ) -> List[GridCell]:
        """
        Request enrichment for a cell AND its neighbors within `radius` cells.
        Useful when the reasoning LLM spots a pattern and wants surrounding context.

        radius=1 → 3×3 grid (8 neighbors + center = 9 cells)
        radius=2 → 5×5 grid (24 neighbors + center = 25 cells)
        """
        level, row, col = grid.parse_tile_id(tile_id)
        cells = []

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                neighbor_id = grid.make_tile_id(level, row + dr, col + dc)
                result = self.request_cell(neighbor_id, agents=agents, infer_missing=False)
                if result:
                    cells.append(result)

        log.info("Neighborhood enrichment complete",
                 center=tile_id,
                 radius=radius,
                 cells_processed=len(cells))
        return cells

    def drill_down(
        self,
        tile_id: str,
        agents: Optional[List[str]] = None,
    ) -> List[GridCell]:
        """
        Drill down from `tile_id` to all child cells at the next finer level.
        Runs agents on each child cell and returns them.

        This is how the reasoning LLM zooms in on a promising area.
        """
        children = grid.get_children(tile_id)
        if not children:
            log.warning("No child level available", tile_id=tile_id)
            return []

        log.info("Drilling down",
                 parent=tile_id,
                 n_children=len(children))

        results = []
        for child_id in children:
            cell = self.request_cell(child_id, agents=agents, infer_missing=True)
            if cell:
                results.append(cell)

        return results

    # ── LLM write-back ────────────────────────────────────────────────────────

    def write_llm_note(
        self,
        tile_id:    str,
        note:       str,
        confidence: float = 0.5,
        model:      str   = "unknown",
        llm_probability: Optional[float] = None,
        llm_reasoning:   Optional[str]   = None,
    ):
        """
        Write the reasoning LLM's analysis back into the cell's ontology.
        This is the 'memory' that accumulates over time.
        """
        note_obj = GeologyNote(
            note       = note,
            confidence = confidence,
            timestamp  = datetime.now(timezone.utc).isoformat(),
            model      = model,
        )
        db.append_llm_note(tile_id, {
            "note":       note,
            "confidence": confidence,
            "timestamp":  note_obj.timestamp,
            "model":      model,
        })

        # Update LLM-specific fields if provided
        if llm_probability is not None or llm_reasoning is not None:
            cell = db.get_cell(tile_id)
            if cell:
                if llm_probability is not None:
                    cell.llm_probability = llm_probability
                    # Blend with agent probability
                    if cell.probability_score is not None:
                        cell.probability_score = (cell.probability_score * 0.6 +
                                                   llm_probability * 0.4)
                if llm_reasoning is not None:
                    cell.llm_reasoning = llm_reasoning
                db.upsert_cell(cell)

        log.info("LLM note written",
                 tile_id=tile_id,
                 confidence=confidence,
                 model=model)

    def exclude_cell(self, tile_id: str, reason: str, model: str = "reasoning_llm"):
        """
        Mark a cell EXCLUDED based on LLM reasoning.
        The cell is NEVER deleted — it stays in DB to inform spatial inference.
        """
        cell = db.get_cell(tile_id)
        if not cell:
            log.warning("Cannot exclude — cell not found", tile_id=tile_id)
            return

        cell.status = CellStatus.EXCLUDED
        note = {
            "note":       f"EXCLUDED: {reason}",
            "confidence": 1.0,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "model":      model,
        }
        cell.llm_notes.append(note)
        cell.llm_reasoning = reason
        db.upsert_cell(cell)

        log.info("Cell excluded",
                 tile_id=tile_id,
                 reason=reason)

    # ── Retrieval for LLM context window ─────────────────────────────────────

    def get_context_for_llm(
        self,
        level: int = 5,
        top_n: int = 15,
        format: str = "prompt",    # "prompt" | "json"
    ) -> str:
        """
        Return a formatted string of the top N cells at `level` for injection
        into the reasoning LLM's context window.

        format="prompt" → natural language summaries stacked
        format="json"   → JSON array (useful for structured LLM outputs)
        """
        cells = db.get_top_cells(level=level, n=top_n)
        log.info("Building LLM context",
                 level=level,
                 n_cells=len(cells),
                 format=format)

        if not cells:
            return f"No complete cells found at level {level}. Ingest may still be running."

        if format == "json":
            return json.dumps([c.to_dict() for c in cells], indent=2)

        # Natural language format
        parts = [
            f"=== TOP {len(cells)} CELLS AT LEVEL {level} "
            f"(cell size: {cells[0].cell_size_m/1000:.1f}km) ===\n"
        ]
        for i, cell in enumerate(cells, 1):
            parts.append(f"\n--- Rank {i} ---")
            parts.append(cell.to_llm_prompt())

        return "\n".join(parts)

    def get_cell_for_llm(self, tile_id: str, enrich_if_needed: bool = True) -> str:
        """
        Return a single cell's LLM summary string.
        If enrich_if_needed=True and cell is PENDING, runs all agents first.
        """
        cell = db.get_cell(tile_id)

        if cell is None or (cell.status == CellStatus.PENDING and enrich_if_needed):
            cell = self.request_cell(tile_id)

        if cell is None:
            return f"ERROR: Cell {tile_id} not found and could not be built."

        return cell.to_llm_prompt()

    # ── Inference engine ──────────────────────────────────────────────────────

    def _infer_missing_attributes(self, cell: GridCell) -> GridCell:
        """
        Generate inferred/estimated attributes from available data when
        primary data sources are absent.

        Examples:
        - No USGS geochemistry nearby → estimate Au anomaly proxy from spectral gossan score
        - No terrain agent run → estimate slope class from elevation range in spectral metadata
        - No history data → assume unworked if no claims in DB

        This is the 'infer any additional useful data on the fly' feature.
        All inferred values are clearly flagged as estimates in the notes.
        """
        inferred = []

        # Infer geochemistry from spectral if absent
        if cell.geochemistry is None and cell.spectral is not None:
            from core.ontology import GeochemistrySummary
            s = cell.spectral

            # Gossan / iron oxide as Au anomaly proxy (weak correlation but non-zero signal)
            proxy_score = None
            if s.iron_oxide_ratio is not None and s.gossan_ratio is not None:
                proxy_score = min(
                    (s.iron_oxide_ratio / 2.0) * 0.5 + (s.gossan_ratio / 1.8) * 0.5,
                    1.0
                )
                inferred.append(f"Au anomaly estimated from spectral gossan proxy (no USGS data)")

            cell.geochemistry = GeochemistrySummary(
                au_anomaly_score    = round(proxy_score, 3) if proxy_score else None,
                nearest_au_sample_m = None,   # Unknown — no sample
            )

        if inferred:
            for note in inferred:
                cell.llm_notes.append({
                    "note":      f"[INFERRED] {note}",
                    "confidence": 0.3,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model":     "inference_engine",
                })

        return cell


# ── Module-level singleton ────────────────────────────────────────────────────
broker = RequestBroker()
