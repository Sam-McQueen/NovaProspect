"""
sierra_prospector/agents/textual_agent.py
=====================================
Textual agent — reads all CSV/JSON point datasets and distributes
their data into the appropriate grid cells.

No LLM involved. Pure spatial point-in-cell assignment.
Job: for every data point, find which cell it belongs to, attach a
structured summary of what's there.

Datasets handled:
    sierra_nevada_gravity.csv         — Bouguer gravity anomaly measurements
    sierra_nevada_physical_properties.csv — Rock density and magnetic susceptibility
    ca_dwr_well_completion_reports.csv    — Water well completion reports
    mindat_mines*.csv (from zip)          — Historical mine locations
    emag2_v3_magnetic_anomaly.zip         — Global magnetic anomaly grid (4.3GB, streamed)

All files read directly from Windows path — no migration needed.
"""

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np

from config.settings import EXCLUSION_THRESHOLD
from core.ontology import GridCell, CellStatus, PointDataSummary
from core.logger import get_logger
from agents.base_agent import BaseAgent

log = get_logger("textual_agent")

TEXTUAL_DIR = Path("/mnt/c/Geodata/Textual")


# ── Per-cell accumulator ──────────────────────────────────────────────────────

@dataclass
class CellAccumulator:
    """Accumulates point data for one cell before writing summary."""
    # Gravity
    gravity_cba:       List[float] = field(default_factory=list)
    gravity_faa:       List[float] = field(default_factory=list)

    # Physical properties
    rock_density:      List[float] = field(default_factory=list)
    mag_susceptibility:List[float] = field(default_factory=list)
    rock_types:        List[str]   = field(default_factory=list)
    unit_names:        List[str]   = field(default_factory=list)

    # Wells
    well_depths:       List[float] = field(default_factory=list)
    well_yields:       List[float] = field(default_factory=list)
    well_water_levels: List[float] = field(default_factory=list)

    # Mines
    mine_names:        List[str]   = field(default_factory=list)
    mine_types:        List[str]   = field(default_factory=list)
    mine_minerals:     List[str]   = field(default_factory=list)
    mine_formations:   List[str]   = field(default_factory=list)
    mine_notes:        List[str]   = field(default_factory=list)

    # Magnetic anomaly
    mag_anomaly_nt:    List[float] = field(default_factory=list)

    def to_point_data_summary(self) -> PointDataSummary:
        s = PointDataSummary()

        # Gravity
        if self.gravity_cba:
            s.gravity_bouguer_mgal = round(float(np.mean(self.gravity_cba)), 4)
            s.gravity_sample_count = len(self.gravity_cba)

        # Magnetics (from physical properties)
        if self.mag_susceptibility:
            s.magnetic_intensity_nt = round(float(np.mean(self.mag_susceptibility)), 4)
            s.magnetic_sample_count = len(self.mag_susceptibility)

        # Magnetic anomaly (EMAG2)
        if self.mag_anomaly_nt:
            existing = s.magnetic_intensity_nt or 0.0
            emag_mean = float(np.mean(self.mag_anomaly_nt))
            if s.magnetic_sample_count == 0:
                s.magnetic_intensity_nt = round(emag_mean, 2)
                s.magnetic_sample_count = len(self.mag_anomaly_nt)
            else:
                # Blend both sources
                s.magnetic_intensity_nt = round(
                    (existing + emag_mean) / 2, 4
                )

        # Wells
        if self.well_depths:
            s.borehole_count = len(self.well_depths)
            s.max_borehole_depth_m = round(float(max(self.well_depths)), 1)

        # Mines
        if self.mine_names:
            s.historic_mine_count = len(self.mine_names)
            s.nearest_mine_name      = self.mine_names[0]
            s.nearest_mine_commodity = self.mine_minerals[0] if self.mine_minerals else None

            # Depletion signal from mine density
            depletion = min(len(self.mine_names) / 5.0, 1.0) * 0.6
            s.depletion_score = round(depletion, 3)

        return s

    def has_data(self) -> bool:
        return any([
            self.gravity_cba, self.rock_density, self.well_depths,
            self.mine_names, self.mag_anomaly_nt,
        ])


# ── Spatial helpers ───────────────────────────────────────────────────────────

def point_in_cell(lat: float, lon: float, cell: GridCell) -> bool:
    return (cell.min_lat <= lat <= cell.max_lat and
            cell.min_lon <= lon <= cell.max_lon)


def build_cell_lookup(cells: List[GridCell]) -> Dict[str, GridCell]:
    """Build a dict keyed by tile_id for fast access."""
    return {c.tile_id: c for c in cells}


def find_cell_for_point(
    lat: float, lon: float, cells: List[GridCell]
) -> Optional[GridCell]:
    """Find which cell contains this lat/lon point."""
    for cell in cells:
        if point_in_cell(lat, lon, cell):
            return cell
    return None


def safe_float(val: Any, default: float = None) -> Optional[float]:
    """Convert to float safely, return default on failure."""
    try:
        f = float(val)
        return f if not (f != f) else default   # NaN check
    except (TypeError, ValueError):
        return default


# ── Dataset readers ───────────────────────────────────────────────────────────

def read_gravity(path: Path, accumulators: Dict[str, CellAccumulator],
                 cells: List[GridCell]):
    """
    Sierra Nevada gravity CSV.
    Columns: id, latitude, longitude, elevation, obs_grav, faa, sba, itc, ttc, code, cba, iso, Source
    Key values: cba (complete Bouguer anomaly), faa (free-air anomaly)
    """
    if not path.exists():
        log.warning("Gravity file not found", path=str(path))
        return 0

    count = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = safe_float(row.get("latitude"))
            lon = safe_float(row.get("longitude"))
            cba = safe_float(row.get("cba"))
            faa = safe_float(row.get("faa"))

            if lat is None or lon is None:
                continue

            cell = find_cell_for_point(lat, lon, cells)
            if cell is None:
                continue

            acc = accumulators.setdefault(cell.tile_id, CellAccumulator())
            if cba is not None:
                acc.gravity_cba.append(cba)
            if faa is not None:
                acc.gravity_faa.append(faa)
            count += 1

    log.info("Gravity loaded", points=count)
    return count


def read_physical_properties(path: Path, accumulators: Dict[str, CellAccumulator],
                              cells: List[GridCell]):
    """
    Physical properties CSV.
    Columns: ID, longitude, latitude, grain_density, sbdensity, dbdensity,
             k_SI_10-3 (magnetic susceptibility), NRM, rock_type, unit_name_or_age
    """
    if not path.exists():
        log.warning("Physical properties file not found", path=str(path))
        return 0

    count = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = safe_float(row.get("latitude"))
            lon = safe_float(row.get("longitude"))

            if lat is None or lon is None:
                continue

            cell = find_cell_for_point(lat, lon, cells)
            if cell is None:
                continue

            acc = accumulators.setdefault(cell.tile_id, CellAccumulator())

            density = safe_float(row.get("dbdensity") or row.get("sbdensity"))
            if density:
                acc.rock_density.append(density)

            # Magnetic susceptibility — strip column name spaces
            k_col = next((k for k in row.keys() if "k_SI" in k or "k _cgs" in k), None)
            if k_col:
                k = safe_float(row.get(k_col))
                if k is not None:
                    acc.mag_susceptibility.append(k)

            rock = row.get("rock_type", "").strip()
            if rock:
                acc.rock_types.append(rock)

            unit = row.get("unit_name_or_age", "").strip()
            if unit:
                acc.unit_names.append(unit)

            count += 1

    log.info("Physical properties loaded", points=count)
    return count


def read_wells(path: Path, accumulators: Dict[str, CellAccumulator],
               cells: List[GridCell]):
    """
    CA DWR well completion reports.
    Key columns: DECIMALLATITUDE, DECIMALLONGITUDE, TOTALDRILLDEPTH,
                 STATICWATERLEVEL, WELLYIELD, PLANNEDUSEFORMERUSE
    Filter: only keep wells in Sierra Nevada counties and reasonable depths.
    """
    if not path.exists():
        log.warning("Wells file not found", path=str(path))
        return 0

    sierra_counties = {
        "Alpine", "Amador", "Calaveras", "El Dorado", "Fresno",
        "Inyo", "Kern", "Madera", "Mariposa", "Mono", "Nevada",
        "Placer", "Plumas", "Sierra", "Tehama", "Tulare", "Tuolumne",
        "Yuba", "Butte", "Shasta",
    }

    count = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            county = row.get("COUNTYNAME", "").replace(" County", "").strip()
            if county not in sierra_counties:
                continue

            lat = safe_float(row.get("DECIMALLATITUDE"))
            lon = safe_float(row.get("DECIMALLONGITUDE"))

            if lat is None or lon is None:
                continue

            cell = find_cell_for_point(lat, lon, cells)
            if cell is None:
                continue

            acc = accumulators.setdefault(cell.tile_id, CellAccumulator())

            depth = safe_float(row.get("TOTALDRILLDEPTH"))
            if depth and 0 < depth < 5000:
                acc.well_depths.append(depth)

            swl = safe_float(row.get("STATICWATERLEVEL"))
            if swl is not None:
                acc.well_water_levels.append(swl)

            yield_val = safe_float(row.get("WELLYIELD"))
            if yield_val and yield_val > 0:
                acc.well_yields.append(yield_val)

            count += 1

    log.info("Wells loaded", points=count)
    return count


def read_mindat_mines(zip_path: Path, accumulators: Dict[str, CellAccumulator],
                      cells: List[GridCell]):
    """
    Mindat mines CSV from zip.
    Columns: mine_name, lat, lon, county, state, mineral_types, mine_type,
             formation_type, active_dates, production_notes, historical_notes,
             geological_notes, confidence_score
    """
    if not zip_path.exists():
        log.warning("Mindat zip not found", path=str(zip_path))
        return 0

    count = 0
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                log.warning("No CSV in mindat zip")
                return 0

            with z.open(csv_files[0]) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8'))
                for row in reader:
                    lat = safe_float(row.get("lat"))
                    lon = safe_float(row.get("lon"))

                    if lat is None or lon is None:
                        continue

                    cell = find_cell_for_point(lat, lon, cells)
                    if cell is None:
                        continue

                    acc = accumulators.setdefault(cell.tile_id, CellAccumulator())

                    name = row.get("mine_name", "").strip()
                    if name:
                        acc.mine_names.append(name)

                    mine_type = row.get("mine_type", "").strip()
                    if mine_type:
                        acc.mine_types.append(mine_type)

                    # Parse mineral_types JSON array
                    minerals_raw = row.get("mineral_types", "")
                    try:
                        minerals = json.loads(minerals_raw)
                        if isinstance(minerals, list) and minerals:
                            acc.mine_minerals.append(", ".join(minerals))
                    except (json.JSONDecodeError, TypeError):
                        if minerals_raw:
                            acc.mine_minerals.append(minerals_raw)

                    formation = row.get("formation_type", "").strip()
                    if formation:
                        acc.mine_formations.append(formation)

                    # Combine notes fields
                    notes = " | ".join(filter(None, [
                        row.get("geological_notes", "").strip(),
                        row.get("historical_notes", "").strip(),
                    ]))
                    if notes:
                        acc.mine_notes.append(notes[:200])

                    count += 1

    except Exception as e:
        log.exception("Failed to read mindat zip", exc=e)

    log.info("Mindat mines loaded", points=count)
    return count


def read_magnetic_anomaly(zip_path: Path, accumulators: Dict[str, CellAccumulator],
                           cells: List[GridCell]):
    """
    EMAG2 magnetic anomaly — 4.3GB CSV, streamed from zip.
    Format: lon, lat, height, mag_anomaly (nT)
    Only reads rows within Sierra Nevada bounding box to avoid loading 4GB.
    """
    if not zip_path.exists():
        log.warning("Magnetic anomaly zip not found", path=str(zip_path))
        return 0

    # Sierra Nevada bounds for pre-filter
    LAT_MIN, LAT_MAX = 35.5, 42.0
    LON_MIN, LON_MAX = -122.5, -117.0

    count = 0
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                return 0

            log.info("Streaming EMAG2 magnetic anomaly",
                     file=csv_files[0],
                     note="4.3GB file — streaming Sierra Nevada rows only")

            with z.open(csv_files[0]) as f:
                for line_bytes in f:
                    try:
                        line = line_bytes.decode('utf-8', errors='replace').strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split(',')
                        if len(parts) < 4:
                            continue

                        lon = safe_float(parts[0])
                        lat = safe_float(parts[1])
                        mag = safe_float(parts[3])

                        if lat is None or lon is None or mag is None:
                            continue

                        # Pre-filter before expensive cell search
                        if not (LAT_MIN <= lat <= LAT_MAX and
                                LON_MIN <= lon <= LON_MAX):
                            continue

                        cell = find_cell_for_point(lat, lon, cells)
                        if cell is None:
                            continue

                        acc = accumulators.setdefault(cell.tile_id, CellAccumulator())
                        acc.mag_anomaly_nt.append(mag)
                        count += 1

                    except Exception:
                        continue

    except Exception as e:
        log.exception("Failed to read magnetic anomaly", exc=e)

    log.info("Magnetic anomaly loaded", sierra_points=count)
    return count


# ── Main agent ────────────────────────────────────────────────────────────────

class TextualAgent(BaseAgent):
    """
    Reads all CSV/JSON point datasets and distributes data into grid cells.

    No LLM. Pure spatial assignment.
    Each dataset is read once, each point assigned to its cell,
    summary statistics written to PointDataSummary.

    Runs independently of all other agents — order does not matter.
    """

    agent_name  = "textual_agent"
    description = "Distributes CSV/JSON point data (gravity, magnetics, wells, mines) into grid cells"

    def __init__(self, data_dir: Path = TEXTUAL_DIR):
        super().__init__()
        self._data_dir = data_dir
        self.log.info("TextualAgent ready", data_dir=str(data_dir))

    def process_cell(self, cell: GridCell, **kwargs) -> GridCell:
        """
        Standard BaseAgent interface — called when iterating cells.
        For the textual agent this is inefficient (would re-read files per cell).
        Use run_all_cells() instead for batch processing.
        """
        return cell

    def run_all_cells(
        self,
        cells: List[GridCell],
        dry_run: bool = False,
    ) -> Dict:
        """
        Primary entry point. Reads all datasets once, assigns to cells,
        writes summaries to DB.

        Much more efficient than process_cell() for large CSV files.
        """
        from core.database import db

        self.log.info("Starting textual agent batch",
                      cells=len(cells), dry_run=dry_run)

        accumulators: Dict[str, CellAccumulator] = {}
        total_points = 0

        # ── Read all datasets ─────────────────────────────────────────────────

        # Gravity
        gravity_path = self._data_dir / "sierra_nevada_gravity.csv"
        if not gravity_path.exists():
            gravity_path = self._data_dir / "GravityDensity USGS" / "sierra_nevada_gravity.csv"
        total_points += read_gravity(gravity_path, accumulators, cells)

        # Physical properties
        phys_path = self._data_dir / "sierra_nevada_physical_properties.csv"
        if not phys_path.exists():
            phys_path = self._data_dir / "GravityDensity USGS" / "sierra_nevada_physical_properties.csv"
        total_points += read_physical_properties(phys_path, accumulators, cells)

        # Wells
        wells_path = self._data_dir / "ca_dwr_well_completion_reports.csv"
        total_points += read_wells(wells_path, accumulators, cells)

        # Mindat mines
        mindat_zips = list(self._data_dir.glob("mindat*.zip"))
        if mindat_zips:
            total_points += read_mindat_mines(mindat_zips[0], accumulators, cells)

        # Magnetic anomaly (large — do last)
        emag_zip = self._data_dir / "emag2_v3_magnetic_anomaly.zip"
        if emag_zip.exists():
            total_points += read_magnetic_anomaly(emag_zip, accumulators, cells)

        self.log.info("All datasets loaded", total_points=total_points,
                      cells_with_data=len(accumulators))

        # ── Write summaries to cells ──────────────────────────────────────────
        updated = 0
        batch   = []

        for cell in cells:
            acc = accumulators.get(cell.tile_id)
            if acc is None or not acc.has_data():
                continue

            summary = acc.to_point_data_summary()
            cell.point_data = summary

            # Update opportunity score with depletion
            if cell.probability_score is not None and summary.depletion_score:
                cell.opportunity_score = round(
                    cell.probability_score * (1 - summary.depletion_score * 0.5), 4
                )

            batch.append(cell)
            updated += 1

            # Print summary for cells with mine data
            if summary.historic_mine_count > 0:
                print(f"\n{cell.tile_id} | {cell.centroid_lat:.2f}N "
                      f"{cell.centroid_lon:.2f}W")
                print(f"  Mines:   {summary.historic_mine_count} — "
                      f"{summary.nearest_mine_name}")
                print(f"  Gravity: {summary.gravity_bouguer_mgal} mGal")
                print(f"  Magnetics: {summary.magnetic_intensity_nt}")
                print(f"  Wells:   {summary.borehole_count}")

        if not dry_run and batch:
            db.upsert_cells_batch(batch)

        stats = {
            "total_points": total_points,
            "cells_with_data": len(accumulators),
            "cells_updated": updated,
            "dry_run": dry_run,
        }
        self.log.info("Textual agent complete", **stats)
        return stats
