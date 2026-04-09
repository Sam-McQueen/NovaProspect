"""
sierra_prospector/core/alerts.py
=====================================
Quality control and escalation system.

Every agent can raise a flag. Flags bubble up the chain until they
reach Berg if serious enough.

Severity levels:
    INFO     — logged only, never stops pipeline
    WARNING  — logged + counted. 3 consecutive cells or 10 per run → STOP
    CRITICAL — stop immediately, log, tell Berg

Usage from any agent:
    from core.alerts import alerts
    alerts.info("hyperspectral_agent",  cell.tile_id, "Only 3 EMIT files")
    alerts.warning("textual_agent",     cell.tile_id, "Gravity -216 mGal implausible")
    alerts.critical("structural_agent", cell.tile_id, "All datasets contradictory")
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum

LOG_PATH = Path("/home/placer/sierra_prospector/logs/alerts.log")

CONSECUTIVE_WARNING_THRESHOLD = 3
WARNINGS_PER_RUN_THRESHOLD    = 10


class Severity(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    severity:  str
    agent:     str
    tile_id:   Optional[str]
    message:   str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context:   Optional[Dict] = None

    def to_log_line(self) -> str:
        ctx  = f" | context={json.dumps(self.context)}" if self.context else ""
        tile = f" | cell={self.tile_id}" if self.tile_id else ""
        return f"[{self.timestamp}] [{self.severity}] [{self.agent}]{tile} {self.message}{ctx}"


class PipelineHaltException(Exception):
    """Raised when the alert system halts the pipeline."""
    pass


class AlertSystem:
    """
    Central alert manager. Thread-safe singleton.
    Import: from core.alerts import alerts
    """

    def __init__(self):
        self._lock              = threading.Lock()
        self._warning_count     = 0
        self._consecutive_warns = 0
        self._last_warn_tile    = None
        self._all_alerts: List[Alert] = []
        self._stopped           = False
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def info(self, agent: str, tile_id: Optional[str],
             message: str, context: dict = None):
        """Logged only. Never stops pipeline."""
        self._handle(Alert(Severity.INFO, agent, tile_id, message, context=context))

    def warning(self, agent: str, tile_id: Optional[str],
                message: str, context: dict = None):
        """Logged + counted. Escalates if thresholds exceeded."""
        alert = Alert(Severity.WARNING, agent, tile_id, message, context=context)
        self._handle(alert)

        with self._lock:
            self._warning_count += 1
            if tile_id != self._last_warn_tile:
                self._consecutive_warns += 1
                self._last_warn_tile = tile_id
            consec = self._consecutive_warns
            total  = self._warning_count

        if consec >= CONSECUTIVE_WARNING_THRESHOLD:
            self._escalate(
                f"WARNING THRESHOLD: {consec} consecutive cells with warnings. "
                f"Last: [{agent}] {message}",
                agent, tile_id
            )
        elif total >= WARNINGS_PER_RUN_THRESHOLD:
            self._escalate(
                f"WARNING THRESHOLD: {total} warnings this run. "
                f"Last: [{agent}] {message}",
                agent, tile_id
            )

    def critical(self, agent: str, tile_id: Optional[str],
                 message: str, context: dict = None):
        """Always stops immediately."""
        self._handle(Alert(Severity.CRITICAL, agent, tile_id, message, context=context))
        self._escalate(message, agent, tile_id, force_critical=True)

    def cell_ok(self, tile_id: str):
        """Call when a cell completes cleanly. Resets consecutive counter."""
        with self._lock:
            if self._last_warn_tile != tile_id:
                self._consecutive_warns = 0

    def reset_run_counters(self):
        """Call at the start of each pipeline run."""
        with self._lock:
            self._warning_count     = 0
            self._consecutive_warns = 0
            self._last_warn_tile    = None
            self._stopped           = False

    def is_stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def summary(self) -> str:
        with self._lock:
            total     = len(self._all_alerts)
            warnings  = sum(1 for a in self._all_alerts if a.severity == Severity.WARNING)
            criticals = sum(1 for a in self._all_alerts if a.severity == Severity.CRITICAL)
            infos     = sum(1 for a in self._all_alerts if a.severity == Severity.INFO)
        return (f"Alerts this run: {total} total "
                f"({infos} info, {warnings} warning, {criticals} critical)")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _handle(self, alert: Alert):
        with self._lock:
            self._all_alerts.append(alert)
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(alert.to_log_line() + "\n")
        except Exception:
            pass

    def _escalate(self, message: str, agent: str,
                  tile_id: Optional[str], force_critical: bool = False):
        with self._lock:
            self._stopped = True

        tile  = f" — cell {tile_id}" if tile_id else ""
        level = "CRITICAL" if force_critical else "WARNING THRESHOLD REACHED"

        print(f"\n{'='*70}")
        print(f"  {level} — PIPELINE HALTED{tile}")
        print(f"  Agent:   {agent}")
        print(f"  Message: {message}")
        print(f"")
        print(f"  Review alert log: {LOG_PATH}")
        print(f"  Fix the issue and re-run. Completed cells will not be reprocessed.")
        print(f"{'='*70}\n")

        raise PipelineHaltException(f"{level}: {message}")


# ── Singleton ─────────────────────────────────────────────────────────────────
alerts = AlertSystem()


# ── Convenience validators ────────────────────────────────────────────────────

def check_coverage(agent: str, tile_id: str, n_found: int,
                   min_expected: int = 1, data_type: str = "data files"):
    if n_found == 0:
        alerts.warning(agent, tile_id,
                       f"No {data_type} found for this cell",
                       {"found": 0})
    elif n_found < min_expected:
        alerts.info(agent, tile_id,
                    f"Sparse {data_type}: {n_found} found (expected ≥{min_expected})",
                    {"found": n_found, "min_expected": min_expected})


def check_value_range(agent: str, tile_id: str, field_name: str,
                      value: float, min_val: float, max_val: float):
    if not (min_val <= value <= max_val):
        alerts.warning(agent, tile_id,
                       f"Implausible value: {field_name}={value:.3f} "
                       f"(expected {min_val}–{max_val})",
                       {"field": field_name, "value": value,
                        "min": min_val, "max": max_val})


def check_vegetation_dominance(agent: str, tile_id: str,
                                veg_pct: float, threshold: float = 0.8):
    if veg_pct > threshold:
        alerts.warning(agent, tile_id,
                       f"{veg_pct:.0%} of hyperspectral pixels are vegetation — "
                       f"mineral detection unreliable (cloud cover or dense forest?)",
                       {"vegetation_pct": round(veg_pct, 3)})


def check_identical_outputs(agent: str, tile_id: str,
                             current: str, previous: str):
    if current and current == previous:
        alerts.critical(agent, tile_id,
                        "Agent producing identical output to previous cell — "
                        "possible data reading error")
