"""
sierra_prospector/core/database.py
=====================================
DuckDB-backed persistence layer.

Why DuckDB over SQLite:
  - Columnar storage → fast analytical queries ("give me all cells with iron_oxide > 2.0")
  - Native GeoJSON/JSON support
  - 5-10x compression on float columns
  - Can query Parquet files directly if we ever export

All DB operations go through this module — no raw SQL elsewhere.
The schema mirrors GridCell exactly so serialization is trivial.
"""

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Iterator

import duckdb

from config.settings import MAIN_DB_PATH, COMMS_DB_PATH, DB_BATCH_SIZE
from core.ontology import GridCell, CellStatus
from core.logger import get_logger

log = get_logger("database")


# ── Schema DDL ────────────────────────────────────────────────────────────────

GRID_CELLS_DDL = """
CREATE TABLE IF NOT EXISTS grid_cells (
    -- Identity
    tile_id          VARCHAR PRIMARY KEY,
    level            INTEGER NOT NULL,
    row              INTEGER NOT NULL,
    col              INTEGER NOT NULL,
    parent_tile_id   VARCHAR,
    cell_size_m      DOUBLE NOT NULL,

    -- Spatial (WGS84)
    min_lon          DOUBLE,
    min_lat          DOUBLE,
    max_lon          DOUBLE,
    max_lat          DOUBLE,
    centroid_lon     DOUBLE,
    centroid_lat     DOUBLE,

    -- Status & scoring
    status           VARCHAR DEFAULT 'PENDING',
    probability_score DOUBLE,
    opportunity_score DOUBLE,
    confidence       DOUBLE,

    -- Agent outputs (stored as JSON strings)
    spectral         VARCHAR,
    terrain          VARCHAR,
    hyperspectral    VARCHAR,
    geochemistry     VARCHAR,
    point_data       VARCHAR,
    structural       VARCHAR,
    history          VARCHAR,

    -- LLM write-back
    llm_notes        VARCHAR DEFAULT '[]',
    llm_probability  DOUBLE,
    llm_reasoning    VARCHAR,

    -- Timestamps
    created_at       VARCHAR,
    updated_at       VARCHAR,

    -- Error tracking
    error_log        VARCHAR DEFAULT '[]'
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_level          ON grid_cells(level);
CREATE INDEX IF NOT EXISTS idx_status         ON grid_cells(status);
CREATE INDEX IF NOT EXISTS idx_prob           ON grid_cells(probability_score DESC);
CREATE INDEX IF NOT EXISTS idx_opportunity    ON grid_cells(opportunity_score DESC);
CREATE INDEX IF NOT EXISTS idx_parent         ON grid_cells(parent_tile_id);
CREATE INDEX IF NOT EXISTS idx_level_status   ON grid_cells(level, status);
CREATE INDEX IF NOT EXISTS idx_spatial        ON grid_cells(centroid_lat, centroid_lon);
"""

# Message queue for two-way agent/LLM communication
COMMS_DDL = """
CREATE TABLE IF NOT EXISTS task_queue (
    task_id          VARCHAR PRIMARY KEY,
    created_at       VARCHAR NOT NULL,
    updated_at       VARCHAR NOT NULL,
    status           VARCHAR DEFAULT 'PENDING',   -- PENDING|RUNNING|COMPLETE|FAILED
    priority         INTEGER DEFAULT 5,           -- 1=urgent, 10=low
    requester        VARCHAR,                     -- 'reasoning_llm' | agent name
    agent_target     VARCHAR,                     -- Which agent should handle this
    tile_id          VARCHAR,                     -- Target tile
    level            INTEGER,
    task_type        VARCHAR,                     -- 'enrich'|'summarize'|'drill_down'|'custom'
    request_payload  VARCHAR DEFAULT '{}',        -- JSON: task parameters
    result_payload   VARCHAR,                     -- JSON: result when complete
    error_message    VARCHAR,
    retry_count      INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_queue_status   ON task_queue(status, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_queue_tile     ON task_queue(tile_id);
CREATE INDEX IF NOT EXISTS idx_queue_agent    ON task_queue(agent_target, status);
"""


# ── Connection management ─────────────────────────────────────────────────────

class ProspectorDB:
    """
    Thread-safe DuckDB wrapper.

    DuckDB has one write connection; reads can use separate read-only connections.
    A threading.Lock guards all writes.
    """

    def __init__(self, db_path=MAIN_DB_PATH, comms_path=COMMS_DB_PATH):
        self._db_path    = db_path
        self._comms_path = comms_path
        self._lock       = threading.Lock()
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._comms_conn: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self):
        """Open connections and ensure schema exists."""
        db_path_str    = str(self._db_path)
        comms_path_str = str(self._comms_path)

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._comms_path.parent.mkdir(parents=True, exist_ok=True)

        log.info("Connecting to DuckDB", main_db=db_path_str, comms_db=comms_path_str)
        self._conn       = duckdb.connect(db_path_str)
        self._comms_conn = duckdb.connect(comms_path_str)

        # Apply schema (idempotent — IF NOT EXISTS)
        for stmt in GRID_CELLS_DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)

        for stmt in COMMS_DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._comms_conn.execute(stmt)

        log.info("Database schema initialised")

    def disconnect(self):
        if self._conn:
            self._conn.close()
        if self._comms_conn:
            self._comms_conn.close()
        log.info("Database connections closed")

    @contextmanager
    def write_lock(self):
        with self._lock:
            yield self._conn

    # ── Grid cell operations ──────────────────────────────────────────────────

    def upsert_cell(self, cell: GridCell):
        """Insert or update a single GridCell. Thread-safe."""
        d = cell.to_dict()
        d["updated_at"] = datetime.now(timezone.utc).isoformat()

        cols   = list(d.keys())
        values = [d[c] for c in cols]
        # DuckDB INSERT OR REPLACE
        placeholders = ", ".join(["?"] * len(cols))
        sql = (
            f"INSERT OR REPLACE INTO grid_cells ({', '.join(cols)}) "
            f"VALUES ({placeholders})"
        )
        with self.write_lock() as conn:
            conn.execute(sql, values)

        log.debug("Cell upserted", tile_id=cell.tile_id, status=cell.status)

    def upsert_cells_batch(self, cells: List[GridCell]):
        """
        Batch upsert — significantly faster than one-by-one for large grids.
        Commits every DB_BATCH_SIZE rows.
        """
        if not cells:
            return

        log.info("Batch upserting cells", count=len(cells))
        batch = []
        total = 0

        for cell in cells:
            d = cell.to_dict()
            d["updated_at"] = datetime.now(timezone.utc).isoformat()
            batch.append(d)

            if len(batch) >= DB_BATCH_SIZE:
                self._flush_batch(batch)
                total += len(batch)
                batch = []
                log.debug("Batch flushed", total_so_far=total)

        if batch:
            self._flush_batch(batch)
            total += len(batch)

        log.info("Batch upsert complete", total=total)

    def _flush_batch(self, batch: List[Dict]):
        """Write a batch to DuckDB using executemany."""
        if not batch:
            return
        cols   = list(batch[0].keys())
        values = [[row[c] for c in cols] for row in batch]
        sql = (
            f"INSERT OR REPLACE INTO grid_cells ({', '.join(cols)}) "
            f"VALUES ({', '.join(['?']*len(cols))})"
        )
        with self.write_lock() as conn:
            conn.executemany(sql, values)

    def get_cell(self, tile_id: str) -> Optional[GridCell]:
        """Fetch a single cell by tile_id."""
        row = self._conn.execute(
            "SELECT * FROM grid_cells WHERE tile_id = ?", [tile_id]
        ).fetchdf()

        if row.empty:
            log.debug("Cell not found in DB", tile_id=tile_id)
            return None

        record = row.iloc[0].to_dict()
        return GridCell.from_dict(record)

    def get_cells_at_level(self, level: int, status: Optional[str] = None) -> List[GridCell]:
        """Fetch all cells at a given resolution level, optionally filtered by status."""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM grid_cells WHERE level = ? AND status = ?",
                [level, status]
            ).fetchdf()
        else:
            rows = self._conn.execute(
                "SELECT * FROM grid_cells WHERE level = ?", [level]
            ).fetchdf()

        return [GridCell.from_dict(r.to_dict()) for _, r in rows.iterrows()]

    def get_top_cells(
        self,
        level: int,
        n: int = 20,
        exclude_depleted: bool = True,
        min_confidence: float = 0.3
    ) -> List[GridCell]:
        """
        Return the top N cells by opportunity_score at a given level.
        Used by the reasoning agent to decide where to drill down.
        """
        sql = """
            SELECT * FROM grid_cells
            WHERE level = ?
              AND status = 'COMPLETE'
              AND opportunity_score IS NOT NULL
              AND confidence >= ?
        """
        params = [level, min_confidence]

        if exclude_depleted:
            sql += " AND status != 'EXCLUDED'"

        sql += " ORDER BY opportunity_score DESC LIMIT ?"
        params.append(n)

        rows = self._conn.execute(sql, params).fetchdf()
        return [GridCell.from_dict(r.to_dict()) for _, r in rows.iterrows()]

    def update_cell_status(self, tile_id: str, status: str, error_msg: Optional[str] = None):
        """Quick status update without fetching the full cell."""
        now = datetime.now(timezone.utc).isoformat()
        if error_msg:
            # Append to error_log JSON array
            self._conn.execute("""
                UPDATE grid_cells
                SET status = ?, updated_at = ?,
                    error_log = json_insert(COALESCE(error_log, '[]'), '$[#]', ?)
                WHERE tile_id = ?
            """, [status, now, error_msg, tile_id])
        else:
            with self.write_lock() as conn:
                conn.execute(
                    "UPDATE grid_cells SET status = ?, updated_at = ? WHERE tile_id = ?",
                    [status, now, tile_id]
                )

    def append_llm_note(self, tile_id: str, note_dict: Dict):
        """Write an LLM reasoning note back to a cell's llm_notes array."""
        now = datetime.now(timezone.utc).isoformat()
        note_json = json.dumps(note_dict)
        with self.write_lock() as conn:
            conn.execute("""
                UPDATE grid_cells
                SET llm_notes  = json_insert(COALESCE(llm_notes, '[]'), '$[#]', json(?)),
                    updated_at = ?
                WHERE tile_id = ?
            """, [note_json, now, tile_id])
        log.debug("LLM note written back", tile_id=tile_id)

    # ── Spatial queries ───────────────────────────────────────────────────────

    def get_cells_in_bbox(
        self,
        min_lon: float, min_lat: float,
        max_lon: float, max_lat: float,
        level: int
    ) -> List[GridCell]:
        """Return all cells at `level` whose centroid falls within the bounding box."""
        rows = self._conn.execute("""
            SELECT * FROM grid_cells
            WHERE level = ?
              AND centroid_lon BETWEEN ? AND ?
              AND centroid_lat BETWEEN ? AND ?
        """, [level, min_lon, max_lon, min_lat, max_lat]).fetchdf()
        return [GridCell.from_dict(r.to_dict()) for _, r in rows.iterrows()]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def level_summary(self) -> List[Dict]:
        """Quick stats per level — useful for monitoring ingest progress."""
        return self._conn.execute("""
            SELECT
                level,
                cell_size_m,
                COUNT(*)                                           AS total_cells,
                SUM(CASE WHEN status='COMPLETE'   THEN 1 ELSE 0 END) AS complete,
                SUM(CASE WHEN status='PENDING'    THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN status='EXCLUDED'   THEN 1 ELSE 0 END) AS excluded,
                SUM(CASE WHEN status='ERROR'      THEN 1 ELSE 0 END) AS errors,
                ROUND(AVG(probability_score), 4)                   AS avg_probability,
                ROUND(MAX(opportunity_score), 4)                   AS max_opportunity
            FROM grid_cells
            GROUP BY level, cell_size_m
            ORDER BY level
        """).fetchdf().to_dict("records")

    # ── Comms / message queue ─────────────────────────────────────────────────

    def enqueue_task(self, task: Dict) -> str:
        """
        Add a task to the queue.
        task must have: agent_target, tile_id, task_type
        Returns the new task_id.
        """
        import uuid
        task_id = str(uuid.uuid4())
        now     = datetime.now(timezone.utc).isoformat()

        self._comms_conn.execute("""
            INSERT INTO task_queue
              (task_id, created_at, updated_at, status, priority,
               requester, agent_target, tile_id, level, task_type, request_payload)
            VALUES (?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?)
        """, [
            task_id, now, now,
            task.get("priority", 5),
            task.get("requester", "unknown"),
            task["agent_target"],
            task["tile_id"],
            task.get("level"),
            task["task_type"],
            json.dumps(task.get("payload", {})),
        ])

        log.info("Task enqueued",
                 task_id=task_id,
                 tile_id=task["tile_id"],
                 task_type=task["task_type"],
                 agent=task["agent_target"])
        return task_id

    def poll_tasks(self, agent_name: str, limit: int = 5) -> List[Dict]:
        """Claim and return pending tasks for an agent. Updates status to RUNNING."""
        rows = self._comms_conn.execute("""
            SELECT * FROM task_queue
            WHERE agent_target = ? AND status = 'PENDING'
            ORDER BY priority ASC, created_at ASC
            LIMIT ?
        """, [agent_name, limit]).fetchdf()

        if rows.empty:
            return []

        task_ids = rows["task_id"].tolist()
        now      = datetime.now(timezone.utc).isoformat()

        for tid in task_ids:
            self._comms_conn.execute(
                "UPDATE task_queue SET status='RUNNING', updated_at=? WHERE task_id=?",
                [now, tid]
            )

        log.debug("Tasks claimed", agent=agent_name, count=len(task_ids))
        return rows.to_dict("records")

    def complete_task(self, task_id: str, result: Dict):
        """Mark a task complete and store the result payload."""
        now = datetime.now(timezone.utc).isoformat()
        self._comms_conn.execute("""
            UPDATE task_queue
            SET status='COMPLETE', updated_at=?, result_payload=?
            WHERE task_id=?
        """, [now, json.dumps(result), task_id])
        log.debug("Task completed", task_id=task_id)

    def fail_task(self, task_id: str, error: str, retry: bool = True):
        """Mark a task failed. If retry=True and under max retries, requeue it."""
        from config.settings import MAX_TASK_RETRIES
        now = datetime.now(timezone.utc).isoformat()

        row = self._comms_conn.execute(
            "SELECT retry_count FROM task_queue WHERE task_id=?", [task_id]
        ).fetchone()
        retry_count = row[0] if row else 0

        if retry and retry_count < MAX_TASK_RETRIES:
            self._comms_conn.execute("""
                UPDATE task_queue
                SET status='PENDING', updated_at=?, error_message=?, retry_count=retry_count+1
                WHERE task_id=?
            """, [now, error, task_id])
            log.warning("Task requeued for retry",
                        task_id=task_id, retry_count=retry_count+1, error=error)
        else:
            self._comms_conn.execute("""
                UPDATE task_queue
                SET status='FAILED', updated_at=?, error_message=?
                WHERE task_id=?
            """, [now, error, task_id])
            log.error("Task permanently failed",
                      task_id=task_id, retry_count=retry_count, error=error)

    def get_task_result(self, task_id: str, timeout_s: float = 30.0) -> Optional[Dict]:
        """
        Poll until a task is COMPLETE or FAILED, then return the result.
        Used by the reasoning agent to wait for on-demand enrichment results.
        """
        import time
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            row = self._comms_conn.execute(
                "SELECT status, result_payload, error_message FROM task_queue WHERE task_id=?",
                [task_id]
            ).fetchone()
            if not row:
                return None
            status, result_payload, error = row
            if status == "COMPLETE":
                return json.loads(result_payload) if result_payload else {}
            if status == "FAILED":
                log.error("Task failed while waiting", task_id=task_id, error=error)
                return None
            time.sleep(0.5)
        log.warning("Timeout waiting for task", task_id=task_id)
        return None


# ── Module-level singleton ────────────────────────────────────────────────────
db = ProspectorDB()
