"""
test_structural.py
==================
Tests the structural agent on one cell.

Steps:
  1. Copies the real DB to a test DB (no risk to production data)
  2. Picks the best-populated cell from the test DB
  3. Runs the structural agent on it
  4. Prints a clean summary of both input data and Grok's output
  5. Does NOT write anything back to the real DB

Usage:
    python test_structural.py                    # uses best available cell
    python test_structural.py Z00_R000009_C000002  # test specific cell
"""

import sys
import json
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
REAL_DB    = Path("/home/placer/sierra_prospector/db/prospector.duckdb")
TEST_DB    = Path("/home/placer/sierra_prospector/db/test_structural.duckdb")
PROJECT    = Path("/home/placer/sierra_prospector")

sys.path.insert(0, str(PROJECT))


def main():

    # ── Step 1: Copy real DB to test DB ──────────────────────────────────────
    print("=" * 60)
    print("STRUCTURAL AGENT TEST")
    print("=" * 60)
    print(f"\n[1] Copying real DB to test DB...")
    shutil.copy2(REAL_DB, TEST_DB)
    print(f"    {REAL_DB.name} → {TEST_DB.name}")
    print(f"    Size: {TEST_DB.stat().st_size / 1024 / 1024:.1f} MB")

    # ── Step 2: Connect to test DB ────────────────────────────────────────────
    print(f"\n[2] Connecting to test DB...")
    from config.settings import Settings
    import config.settings as settings_module

    # Override DB path to use test DB
    original_db_path = settings_module.DB_PATH if hasattr(settings_module, 'DB_PATH') else None

    from core.database import DatabaseManager
    test_db = DatabaseManager(
        main_db_path=str(TEST_DB),
        comms_db_path=str(TEST_DB.parent / "test_comms.duckdb")
    )
    test_db.connect()
    print("    Connected.")

    # ── Step 3: Pick cell to test ─────────────────────────────────────────────
    tile_id = sys.argv[1] if len(sys.argv) > 1 else None

    if tile_id:
        cell = test_db.get_cell(tile_id)
        if cell is None:
            print(f"\nERROR: Cell {tile_id} not found in DB.")
            test_db.disconnect()
            return
    else:
        # Pick cell with most data populated
        print(f"\n[3] Finding best-populated cell...")
        rows = test_db._conn.execute("""
            SELECT tile_id,
                   (CASE WHEN terrain IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN hyperspectral IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN point_data IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN structural IS NULL THEN 1 ELSE 0 END) as score
            FROM grid_cells
            WHERE level = 0
            ORDER BY score DESC, probability_score DESC NULLS LAST
            LIMIT 5
        """).fetchall()

        if not rows:
            print("    No level 0 cells found.")
            test_db.disconnect()
            return

        print("    Top 5 candidates:")
        for tid, score in rows:
            print(f"      {tid} (data fields: {score-1})")

        tile_id = rows[0][0]
        print(f"    Selected: {tile_id}")

    cell = test_db.get_cell(tile_id)

    # ── Step 4: Print input data summary ─────────────────────────────────────
    print(f"\n[4] Input data for {tile_id}:")
    print("-" * 60)

    from agents.structural_agent import build_cell_summary
    summary_text = build_cell_summary(cell)
    print(summary_text)

    # Data completeness report
    print("\n    Data fields populated:")
    fields = {
        "terrain":       cell.terrain is not None,
        "spectral":      cell.spectral is not None,
        "hyperspectral": cell.hyperspectral is not None,
        "point_data":    cell.point_data is not None,
        "geochemistry":  cell.geochemistry is not None,
        "structural":    cell.structural is not None,
        "history":       cell.history is not None,
    }
    for name, populated in fields.items():
        status = "✓" if populated else "✗"
        print(f"      {status} {name}")

    # ── Step 5: Run structural agent on test DB ───────────────────────────────
    print(f"\n[5] Running structural agent...")
    print("-" * 60)

    from agents.structural_agent import StructuralAgent
    agent = StructuralAgent()

    result = agent.process_cell(cell)

    # ── Step 6: Print results ─────────────────────────────────────────────────
    print(f"\n[6] Results:")
    print("=" * 60)

    if result.structural is None:
        print("    FAILED — no structural summary produced.")
        print("    Check GROK_API_KEY and API connectivity.")
    else:
        s = result.structural
        print(f"  STRUCTURAL_SETTING:  {s.structural_setting}")
        print(f"  DEFORMATION_STYLE:   {s.deformation_style}")
        print(f"  KEY_OBSERVATIONS:    {s.key_observations}")
        print(f"  NOTABLE_FEATURES:    {s.notable_features}")
        print(f"  STRUCTURAL_SCORE:    {s.structural_score}")
        print(f"\n  Probability score:   {result.probability_score}")
        print(f"  Opportunity score:   {result.opportunity_score}")

    # ── Step 7: Write to test DB only ─────────────────────────────────────────
    print(f"\n[7] Writing result to TEST DB (not real DB)...")
    test_db.upsert_cell(result)
    print(f"    Written to {TEST_DB.name}")

    # Verify it was written
    check = test_db.get_cell(tile_id)
    if check and check.structural:
        print(f"    Verified: structural summary present in test DB ✓")
    else:
        print(f"    WARNING: Could not verify write.")

    test_db.disconnect()

    print(f"\n{'='*60}")
    print(f"TEST COMPLETE")
    print(f"Real DB untouched: {REAL_DB.name}")
    print(f"Test DB:           {TEST_DB.name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
