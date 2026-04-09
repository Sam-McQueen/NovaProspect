"""
sierra_prospector/main.py
=====================================
Main entry point.

AGENT ROUNDS:
  Round 1 — Raw data readers (free, local, run in any order)
    terrain_agent      DEM → elevation, slope, aspect
    spectral_agent     Landsat → band ratios, alteration
    geochemistry_agent USGS stream sediment → Au anomaly scores
    point_data_agent   Gravity, magnetics, faults, mines, boreholes

  Round 2 — Context readers (run after Round 1)
    structural_agent   Synthesises Round 1 → structural gold trap assessment
    vision_agent       Renders DEM images → Grok visual interpretation

  Round 3 — Final (runs last, reads everything)
    history_agent      Historical docs + all prior data → depletion score

COMMANDS:
  python main.py                              Full ingest levels 0-5 (all agents)
  python main.py --agent terrain              Run ONLY terrain agent
  python main.py --agent spectral             Run ONLY spectral agent
  python main.py --agent geochemistry         Run ONLY geochemistry agent
  python main.py --agent point_data           Run ONLY point data agent
  python main.py --agent structural           Run ONLY structural agent (Round 2)
  python main.py --agent vision               Run ONLY vision agent (Round 2)
  python main.py --agent history              Run ONLY history agent (Round 3)
  python main.py --max-level 0                Run through level 0 only
  python main.py --dry-run                    No DB writes
  python main.py --check                      Verify setup
  python main.py --status                     Current DB progress
  python main.py --export-geojson 0           Export level 0 to QGIS
  python main.py --read-cell Z00_R008_C004    Human summary for one cell
  python main.py --test-vision Z00_R008_C004  Run vision on one cell
  python main.py --storage-estimate           Disk space estimate
"""

import argparse
import sys
import json
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    RESOLUTION_LEVELS, DRILL_THRESHOLD, OUTPUTS_DIR,
    ACTIVE_CONFIG, AGENT_PHASE
)
from core.logger import get_logger
from core.database import db
from core.grid import grid
from core.comms import broker
from core.ontology import CellStatus

log = get_logger("main")


# ── Agent registry ────────────────────────────────────────────────────────────
# Add new agents here. Order matters within rounds.

ROUND_1_AGENTS = [
    "terrain_agent",
    "spectral_agent",
    "geochemistry_agent",
    "point_data_agent",
    "lidar_agent",
    "hyperspectral_agent",
    "textual_agent",
    "vector_agent",
]

ROUND_2_AGENTS = [
    "structural_agent",
    "vision_agent",
]

ROUND_3_AGENTS = [
    "history_agent",
]

ALL_AGENTS = ROUND_1_AGENTS + ROUND_2_AGENTS + ROUND_3_AGENTS


def build_agents():
    """Instantiate all agents and register with broker."""
    agents = {}

    from agents.terrain_agent        import TerrainAgent
    from agents.spectral_agent       import SpectralAgent
    from agents.geochemistry_agent   import GeochemistryAgent
    from agents.point_data_agent     import PointDataAgent
    from agents.lidar_agent          import LidarAgent
    from agents.hyperspectral_agent  import HyperspectralAgent
    from agents.textual_agent        import TextualAgent
    from agents.vector_agent         import VectorAgent
    from agents.structural_agent     import StructuralAgent
    from agents.vision_agent         import VisionAgent
    from agents.history_agent        import HistoryAgent

    agents["terrain_agent"]       = TerrainAgent()
    agents["spectral_agent"]      = SpectralAgent()
    agents["geochemistry_agent"]  = GeochemistryAgent()
    agents["point_data_agent"]    = PointDataAgent()
    agents["lidar_agent"]         = LidarAgent()
    agents["hyperspectral_agent"] = HyperspectralAgent()
    agents["textual_agent"]       = TextualAgent()
    agents["vector_agent"]        = VectorAgent()
    agents["structural_agent"]    = StructuralAgent()
    agents["vision_agent"]        = VisionAgent()
    agents["history_agent"]       = HistoryAgent()

    for name, agent in agents.items():
        broker.register_agent(name, agent)

    log.info("All agents ready", registered=list(agents.keys()))
    return agents


# ── Ingest ────────────────────────────────────────────────────────────────────

def run_ingest(
    min_level:    int  = 0,
    max_level:    int  = 5,
    dry_run:      bool = False,
    workers:      int  = 4,
    agent_filter: str  = None,   # None = all agents, else agent name
):
    """
    Main ingest loop. Processes all cells from min_level to max_level.

    agent_filter: run only this one agent (for manual dataset construction)
    """
    log.info("Starting ingest",
             min_level=min_level, max_level=max_level,
             dry_run=dry_run, agent_filter=agent_filter or "all")

    agents = build_agents()

    from core.alerts import alerts, PipelineHaltException
    alerts.reset_run_counters()

    db.connect()

    try:
        for level in range(min_level, max_level + 1):
            cell_size_m = RESOLUTION_LEVELS.get(level)
            if cell_size_m is None:
                continue
        log.info("=" * 60)
        log.info(f"LEVEL {level} — cell size {cell_size_m:,}m ({cell_size_m/1000:.1f}km)")
        log.info("=" * 60)

        # Get cells to process
        if level == 0:
            cells_iter = list(grid.iter_cells_at_level(level))
        else:
            qualifying_parents = db.get_top_cells(
                level=level - 1, n=999_999, min_confidence=0.2
            )
            log.info("Qualifying parent cells",
                     parent_level=level-1,
                     count=len(qualifying_parents))

            def child_cells(parents):
                for parent in parents:
                    for child_id in grid.get_children(parent.tile_id):
                        cell = grid.build_cell(*grid.parse_tile_id(child_id))
                        yield cell

            cells_iter = list(child_cells(qualifying_parents))

        # Register all cells in DB and load existing data
        batch = []
        cells_for_analysis = []
        for cell in cells_iter:
            existing = db.get_cell(cell.tile_id)
            if existing:
                if existing.status == CellStatus.EXCLUDED:
                    continue   # Never re-process excluded cells
                if existing.status == CellStatus.COMPLETE and agent_filter is None:
                    continue   # Skip complete cells only in full pipeline mode
                # Always use existing cell — preserves all prior agent data
                cells_for_analysis.append(existing)
            else:
                # New cell — register it
                batch.append(cell)
                cells_for_analysis.append(cell)
            if len(batch) >= 500:
                if not dry_run:
                    db.upsert_cells_batch(batch)
                batch = []

        if batch and not dry_run:
            db.upsert_cells_batch(batch)

        log.info("Cells registered", level=level, to_process=len(cells_for_analysis))

        if not cells_for_analysis:
            log.info("No cells to process", level=level)
            continue

        # ── Determine which agents to run ─────────────────────────────────────
        if agent_filter:
            # Single agent mode
            if agent_filter not in agents:
                print(f"Unknown agent: {agent_filter}")
                print(f"Available: {', '.join(ALL_AGENTS)}")
                return
            run_order = [agent_filter]
        else:
            # Full pipeline
            run_order = ALL_AGENTS

        # Reset alert counters for this level
        from core.alerts import alerts
        alerts.reset_run_counters()

        # ── Round 1: local agents ─────────────────────────────────────────────
        for agent_name in run_order:
            if agent_name not in ROUND_1_AGENTS:
                continue
            agent = agents[agent_name]

            # Vector agent uses bulk loader — different interface
            if agent_name == "vector_agent":
                print(f"\n--- Running {agent_name} on {len(cells_for_analysis)} cells ---")
                stats = agent.run_all_cells(cells_for_analysis, dry_run=dry_run)
                log.info("Agent complete", agent=agent_name, **stats)
                print(f"    {agent_name}: {stats['cells_updated']} cells updated")
                continue

            # Textual agent uses bulk loader — different interface
            if agent_name == "textual_agent":
                print(f"\n--- Running {agent_name} on {len(cells_for_analysis)} cells ---")
                stats = agent.run_all_cells(cells_for_analysis, dry_run=dry_run)
                log.info("Agent complete", agent=agent_name, **stats)
                print(f"    {agent_name}: {stats['cells_updated']} cells updated, "
                      f"{stats['total_points']} points distributed")
                continue
            print(f"\n--- Running {agent_name} on {len(cells_for_analysis)} cells ---")
            stats = agent.run_on_cells(
                iter(cells_for_analysis),
                workers=workers,
                dry_run=dry_run,
            )
            log.info("Agent complete", agent=agent_name, **stats)
            print(f"    {agent_name}: {stats['success']} ok, {stats['failed']} failed, {stats['duration_s']}s")

            # Check for critical alerts — halt if needed
            if alerts.should_halt():
                print(f"\nPIPELINE HALTED by alert system.")
                print(f"Reason: {alerts.halt_reason()}")
                db.disconnect()
                return

        # ── Round 2: context agents (API calls — only qualifying cells) ────────
        for agent_name in run_order:
            if agent_name not in ROUND_2_AGENTS:
                continue
            agent = agents[agent_name]
            cutoff = ACTIVE_CONFIG.get("confidence_cutoff", 0.3)

            # Refresh cells from DB to get Round 1 scores
            if not dry_run:
                qualifying = [
                    c for c in [db.get_cell(cell.tile_id) for cell in cells_for_analysis]
                    if c and (c.probability_score or 0) >= cutoff
                ]
            else:
                qualifying = cells_for_analysis

            print(f"\n--- Running {agent_name} on {len(qualifying)} qualifying cells ---")
            if not qualifying:
                print(f"    No cells above cutoff ({cutoff}) — skipping")
                continue

            stats = agent.run_on_cells(
                iter(qualifying),
                workers=1,   # API agents single-threaded
                dry_run=dry_run,
            )
            log.info("Agent complete", agent=agent_name, **stats)
            print(f"    {agent_name}: {stats['success']} ok, {stats['failed']} failed")

        # ── Round 3: history (runs last on all cells) ──────────────────────────
        for agent_name in run_order:
            if agent_name not in ROUND_3_AGENTS:
                continue
            agent = agents[agent_name]

            if not dry_run:
                all_cells = [
                    c for c in [db.get_cell(cell.tile_id) for cell in cells_for_analysis]
                    if c is not None
                ]
            else:
                all_cells = cells_for_analysis

            print(f"\n--- Running {agent_name} on {len(all_cells)} cells ---")
            stats = agent.run_on_cells(
                iter(all_cells),
                workers=1,
                dry_run=dry_run,
            )
            log.info("Agent complete", agent=agent_name, **stats)

        # ── Export GeoJSON ─────────────────────────────────────────────────────
        if not dry_run:
            export_geojson(level)

        # Print alert summary for this level
        print(alerts.summary())

    except PipelineHaltException:
        pass   # Banner already printed by alert system

    log.info("Ingest complete")
    print_status()


# ── Utilities ─────────────────────────────────────────────────────────────────

def export_geojson(level: int):
    cells = db.get_cells_at_level(level)
    if not cells:
        return
    features = []
    for cell in cells:
        spectral = json.loads(cell.spectral) if isinstance(cell.spectral, str) and cell.spectral else {}
        history  = json.loads(cell.history)  if isinstance(cell.history, str) and cell.history else {}
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [cell.min_lon, cell.min_lat],
                    [cell.max_lon, cell.min_lat],
                    [cell.max_lon, cell.max_lat],
                    [cell.min_lon, cell.max_lat],
                    [cell.min_lon, cell.min_lat],
                ]]
            },
            "properties": {
                "tile_id":           cell.tile_id,
                "level":             cell.level,
                "cell_size_m":       cell.cell_size_m,
                "status":            cell.status,
                "probability_score": cell.probability_score,
                "opportunity_score": cell.opportunity_score,
                "confidence":        cell.confidence,
                "alteration_type":   spectral.get("alteration_type"),
                "depletion_score":   history.get("depletion_score"),
                "centroid_lat":      cell.centroid_lat,
                "centroid_lon":      cell.centroid_lon,
            }
        })
    out = OUTPUTS_DIR / "geojson" / f"level_{level:02d}.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    log.info("GeoJSON exported", level=level, path=str(out), features=len(features))
    print(f"  → QGIS file: {out}")


def print_status():
    summary = db.level_summary()
    if not summary:
        print("No data in DB yet.")
        return
    print("\n" + "=" * 70)
    print(f"{'LEVEL':>5}  {'SIZE':>8}  {'TOTAL':>8}  {'DONE':>8}  "
          f"{'EXCL':>8}  {'ERR':>5}  {'AVG PROB':>9}  {'MAX OPP':>8}")
    print("-" * 70)
    for row in summary:
        print(
            f"{row['level']:>5}  "
            f"{row['cell_size_m']/1000:>6.1f}km  "
            f"{row['total_cells']:>8,}  "
            f"{row['complete']:>8,.0f}  "
            f"{row['excluded']:>8,.0f}  "
            f"{row['errors']:>5,.0f}  "
            f"{(row['avg_probability'] or 0):>9.4f}  "
            f"{(row['max_opportunity'] or 0):>8.4f}"
        )
    print("=" * 70)


def print_storage_estimate():
    estimates = grid.storage_estimate()
    print("\n" + "=" * 80)
    print(f"{'LEVEL':>5}  {'SIZE':>8}  {'EST CELLS':>12}  {'FILL':>6}  "
          f"{'RAW GB':>8}  {'COMP GB':>8}  {'CUMUL GB':>9}")
    print("-" * 80)
    for level, e in estimates.items():
        print(
            f"{level:>5}  {e['cell_size_m']/1000:>6.1f}km  "
            f"{e['est_cells']:>12,}  {e['fill_factor']:>5.0%}  "
            f"{e['raw_gb']:>8.3f}  {e['compressed_gb']:>8.3f}  "
            f"{e['cumulative_gb']:>9.3f}"
        )
    total = list(estimates.values())[-1]["cumulative_gb"]
    print("=" * 80)
    print(f"  Estimated total compressed: {total:.1f} GB")


def run_check():
    from config.settings import SIERRA_BOUNDARY_GEOJSON, RAW_DIR, DB_DIR, LOGS_DIR, OUTPUTS_DIR
    print("\n=== Sierra Prospector — Setup Check ===\n")
    all_ok = True

    def check(label, condition, fix=""):
        nonlocal all_ok
        status = "  OK " if condition else "MISSING"
        if not condition:
            all_ok = False
        print(f"  [{status}]  {label}")
        if not condition and fix:
            print(f"           → {fix}")

    print("Python packages:")
    for pkg in ["rasterio","numpy","shapely","pyproj","duckdb","geopandas","scipy","PIL","requests"]:
        imp = "PIL" if pkg == "PIL" else pkg
        try:
            __import__(imp)
            check(pkg, True)
        except ImportError:
            check(pkg, False, f"pip install {'Pillow' if pkg == 'PIL' else pkg}")

    print("\nData files:")
    check("Sierra Nevada boundary GeoJSON", SIERRA_BOUNDARY_GEOJSON.exists(),
          f"Place at: {SIERRA_BOUNDARY_GEOJSON}")
    tifs = list(RAW_DIR.glob("**/*.tif")) + list(RAW_DIR.glob("**/*.TIF")) if RAW_DIR.exists() else []
    check(f"GeoTIFF files ({len(tifs)} found)", len(tifs) > 0,
          f"Place in: {RAW_DIR}")

    print("\nAPI:")
    from config.settings import GROK_API_KEY
    check("GROK_API_KEY set", bool(GROK_API_KEY),
          "Run: export GROK_API_KEY='xai-...'")

    print("\nDirectories:")
    for label, path in [("db/", DB_DIR), ("logs/", LOGS_DIR), ("outputs/", OUTPUTS_DIR)]:
        path.mkdir(parents=True, exist_ok=True)
        check(label, True)

    print("\nDatabase:")
    try:
        db.connect()
        check("DuckDB", True)
        db.disconnect()
    except Exception as e:
        check("DuckDB", False, str(e))

    print()
    if all_ok:
        print("  All good. Run: python main.py --dry-run --max-level 0")
    else:
        print("  Fix MISSING items then run --check again.")
    print()


def read_cell(tile_id: str):
    db.connect()
    cell = db.get_cell(tile_id)
    db.disconnect()
    if cell is None:
        print(f"\nCell '{tile_id}' not found. Run an ingest first.\n")
        return
    print()
    print(cell.to_llm_prompt())
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sierra Nevada Gold Prospecting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Individual agent runs (for building datasets with full oversight):
  python main.py --agent terrain              Terrain only — all levels
  python main.py --agent terrain --max-level 0  Terrain on level 0 only
  python main.py --agent spectral             Spectral only
  python main.py --agent geochemistry         Geochemistry only
  python main.py --agent point_data           Gravity/magnetics/faults/mines
  python main.py --agent structural           Structural synthesis (needs Round 1)
  python main.py --agent vision               Vision/Grok analysis
  python main.py --agent history              History + depletion (runs last)

Full pipeline:
  python main.py --max-level 0                All agents, level 0 only
  python main.py                              All agents, all levels

Utilities:
  python main.py --check                      Verify setup
  python main.py --status                     DB progress
  python main.py --export-geojson 0           Export to QGIS
  python main.py --read-cell Z00_R000008_C000004
  python main.py --test-vision Z00_R000008_C000004
  python main.py --storage-estimate
        """
    )
    parser.add_argument("--agent",            type=str,  default=None, help=f"Run one agent only: {', '.join(ALL_AGENTS)}")
    parser.add_argument("--min-level",        type=int,  default=0)
    parser.add_argument("--max-level",        type=int,  default=5)
    parser.add_argument("--workers",          type=int,  default=4)
    parser.add_argument("--dry-run",          action="store_true")
    parser.add_argument("--check",            action="store_true")
    parser.add_argument("--storage-estimate", action="store_true")
    parser.add_argument("--status",           action="store_true")
    parser.add_argument("--export-geojson",   type=int,  default=None)
    parser.add_argument("--read-cell",        type=str,  default=None)
    parser.add_argument("--test-vision",      type=str,  default=None)
    parser.add_argument("--test-lidar",       type=str,  default=None)
    parser.add_argument("--lidar-tiles",      action="store_true", help="Run LiDAR agent on all tiles at native resolution")

    args = parser.parse_args()

    if args.check:
        run_check()
        return

    if args.storage_estimate:
        db.connect(); print_storage_estimate(); db.disconnect()
        return

    if args.status:
        db.connect(); print_status(); db.disconnect()
        return

    if args.export_geojson is not None:
        db.connect(); export_geojson(args.export_geojson); db.disconnect()
        return

    if args.read_cell:
        read_cell(args.read_cell)
        return

    if args.test_vision:
        from agents.vision_agent import VisionAgent
        print(f"\nVision test — phase: {AGENT_PHASE}")
        print(f"Model: {ACTIVE_CONFIG['vision_model']}")
        print(f"Max tokens: {ACTIVE_CONFIG['max_tokens']}\n")
        db.connect()
        cell = db.get_cell(args.test_vision)
        if cell is None:
            print(f"Cell {args.test_vision} not found. Run an ingest first.")
            db.disconnect()
            return
        agent = VisionAgent()
        result = agent.process_cell(cell)
        db.upsert_cell(result)
        print("\nUpdated cell summary:")
        print(result.to_llm_prompt())
        db.disconnect()
        return

    if args.test_lidar:
        from agents.lidar_agent import LidarAgent
        print(f"\nLiDAR test — processing all tiles near a known location")
        db.connect()
        agent = LidarAgent()
        # Find a zip that overlaps the requested cell
        cell = db.get_cell(args.test_lidar)
        if cell is None:
            print(f"Cell {args.test_lidar} not found.")
            db.disconnect()
            return
        from agents.lidar_agent import find_overlapping_tiles
        bounds  = (cell.min_lon, cell.min_lat, cell.max_lon, cell.max_lat)
        tiles   = find_overlapping_tiles(bounds, agent._tile_index, agent._lidar_dir)
        if not tiles:
            print(f"No LiDAR tiles overlap cell {args.test_lidar}")
            print("Try a cell in the -121 to -118W, 35-40N range")
            db.disconnect()
            return
        print(f"Found {len(tiles)} tiles — processing first one")
        note = agent._process_one_zip(tiles[0])
        if note:
            print(f"\nConfidence: {note['confidence']}")
            print(f"Depletion:  {note['depletion']}")
        db.disconnect()
        return

    if args.lidar_tiles:
        from agents.lidar_agent import LidarAgent
        print(f"\nRunning LiDAR agent on all {args.lidar_tiles} tiles")
        db.connect()
        agent = LidarAgent()
        stats = agent.process_all_tiles(dry_run=args.dry_run)
        print(f"\nDone: {stats}")
        db.disconnect()
        return
    run_ingest(
        min_level    = args.min_level,
        max_level    = args.max_level,
        dry_run      = args.dry_run,
        workers      = args.workers,
        agent_filter = args.agent,
    )


if __name__ == "__main__":
    main()
