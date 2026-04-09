"""
sierra_prospector/agents/base_agent.py
=====================================
Abstract base class. All agents inherit from this.
Provides: logging, error handling, batch runner, task queue listener.
"""

import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List, Optional, Iterator, Dict

from config.settings import MAX_WORKERS, ON_DEMAND_TIMEOUT
from core.ontology import GridCell, CellStatus
from core.database import db
from core.logger import get_logger


class BaseAgent(ABC):
    agent_name:  str = "base_agent"
    description: str = "Abstract base agent"

    def __init__(self):
        self.log = get_logger(self.agent_name)
        self.log.info("Agent initialised", agent=self.agent_name)

    @abstractmethod
    def process_cell(self, cell: GridCell, **kwargs) -> GridCell:
        """Process one cell. Return modified cell. Never write to DB directly."""
        ...

    def run_on_cells(
        self,
        cells: Iterator[GridCell],
        workers: int = MAX_WORKERS,
        dry_run: bool = False,
    ) -> Dict:
        self.log.info("Starting batch run", agent=self.agent_name,
                      workers=workers, dry_run=dry_run)
        t0 = time.time()
        stats = {"total": 0, "success": 0, "failed": 0, "excluded": 0}
        batch_buffer: List[GridCell] = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._safe_process, cell): cell for cell in cells}
            for future in as_completed(futures):
                stats["total"] += 1
                try:
                    result = future.result()
                    if result.status == CellStatus.ERROR:
                        stats["failed"] += 1
                    elif result.status == CellStatus.EXCLUDED:
                        stats["excluded"] += 1
                    else:
                        stats["success"] += 1
                        from core.alerts import alerts
                        alerts.cell_ok(result.tile_id)
                    batch_buffer.append(result)
                    if not dry_run and len(batch_buffer) >= 100:
                        db.upsert_cells_batch(batch_buffer)
                        batch_buffer = []
                except Exception as e:
                    stats["failed"] += 1
                    self.log.exception("Future raised unexpected exception",
                                       exc=e, agent=self.agent_name)

        if batch_buffer and not dry_run:
            db.upsert_cells_batch(batch_buffer)

        stats["duration_s"] = round(time.time() - t0, 2)
        self.log.info("Batch run complete", **stats)
        return stats

    def run_on_tile(self, tile_id: str, **kwargs) -> Optional[GridCell]:
        cell = db.get_cell(tile_id)
        if cell is None:
            self.log.error("Tile not found in DB", tile_id=tile_id)
            return None
        result = self._safe_process(cell, **kwargs)
        db.upsert_cell(result)
        return result

    def _safe_process(self, cell: GridCell, **kwargs) -> GridCell:
        try:
            db.update_cell_status(cell.tile_id, CellStatus.PROCESSING)
            result = self.process_cell(cell, **kwargs)
            if result.status not in (CellStatus.EXCLUDED, CellStatus.ERROR):
                result.status = CellStatus.COMPLETE
            result.updated_at = datetime.now(timezone.utc).isoformat()
            return result
        except Exception as e:
            tb = traceback.format_exc()
            cell.error_log.append({
                "agent":     self.agent_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error":     str(e),
                "traceback": tb,
            })
            cell.status = CellStatus.ERROR
            cell.updated_at = datetime.now(timezone.utc).isoformat()
            self.log.exception("Cell processing failed",
                               exc=e, tile_id=cell.tile_id, agent=self.agent_name)
            return cell

    def listen_for_tasks(self, poll_interval: float = 1.0,
                         max_iterations: Optional[int] = None):
        self.log.info("Agent listening for tasks", agent=self.agent_name)
        iterations = 0
        while True:
            tasks = db.poll_tasks(self.agent_name)
            for task in tasks:
                task_id = task["task_id"]
                tile_id = task["tile_id"]
                try:
                    import json
                    payload = json.loads(task.get("request_payload") or "{}")
                    result_cell = self.run_on_tile(tile_id, **payload)
                    result = {"tile_id": tile_id, "status": "ok"}
                    if result_cell:
                        result["summary"] = result_cell.to_llm_prompt()
                    db.complete_task(task_id, result)
                except Exception as e:
                    self.log.exception("Task processing failed",
                                       exc=e, task_id=task_id, tile_id=tile_id)
                    db.fail_task(task_id, str(e))
            time.sleep(poll_interval)
            iterations += 1
            if max_iterations and iterations >= max_iterations:
                break
