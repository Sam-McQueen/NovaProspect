"""
sierra_prospector/agents/point_data_agent.py
=====================================
Point data agent — one agent that ingests ALL simple coordinate-based datasets.

Handles:
    - Gravity anomaly CSV          → Bouguer anomaly per cell
    - Magnetic anomaly grid/CSV    → magnetic intensity per cell
    - Qfault shapefile             → fault proximity, type, age, density
    - Hydrology features           → stream count, order, proximity
    - Historic mine coordinates    → mine count, proximity, commodity
    - Borehole collar locations    → count, depth, proximity

Why one agent for all of these:
    They are all the same operation — "which points fall in my cell bounds?"
    No vision, no LLM, no complex math. Just spatial queries.
    Combining them into one agent means one pass over the cell, one DB write.

Can run at any time — does not depend on other agents.

Data files auto-discovered from RAW_DIR. Place files in:
    /mnt/c/Geodata/remote_sensing/gravity/    ← gravity CSV
    /mnt/c/Geodata/remote_sensing/magnetics/  ← magnetics CSV or grid
    /mnt/c/Geodata/remote_sensing/faults/     ← Qfault shapefile
    /mnt/c/Geodata/remote_sensing/hydrology/  ← hydrology shapefile
    /mnt/c/Geodata/remote_sensing/mines/      ← historic mine points
    /mnt/c/Geodata/remote_sensing/boreholes/  ← borehole collar CSV
    OR directly in /mnt/c/Geodata/remote_sensing/ — agent searches recursively.
"""

import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from config.settings import RAW_DIR
from core.ontology import GridCell, CellStatus, PointDataSummary
from core.logger import get_logger
from agents.base_agent import BaseAgent

log = get_logger("point_data_agent")


# ── Data loading helpers ──────────────────────────────────────────────────────

def _load_csv_points(paths: List[Path], lat_hints: List[str],
                     lon_hints: List[str]) -> Optional[Any]:
    """Generic CSV loader — auto-detects lat/lon columns."""
    try:
        import pandas as pd
        frames = []
        for p in paths:
            try:
                df = pd.read_csv(p, low_memory=False)
                lat = next((c for c in df.columns if c.lower() in lat_hints), None)
                lon = next((c for c in df.columns if c.lower() in lon_hints), None)
                if lat and lon:
                    df = df.dropna(subset=[lat, lon])
                    df["_lat"] = pd.to_numeric(df[lat], errors="coerce")
                    df["_lon"] = pd.to_numeric(df[lon], errors="coerce")
                    df = df.dropna(subset=["_lat", "_lon"])
                    frames.append(df)
                    log.info("Loaded CSV", file=p.name, rows=len(df))
            except Exception as e:
                log.warning("Failed to load CSV", file=str(p), error=str(e))
        if not frames:
            return None
        import pandas as pd
        return pd.concat(frames, ignore_index=True)
    except ImportError:
        log.error("pandas not installed")
        return None


def _load_shapefile(paths: List[Path]) -> Optional[Any]:
    """Load shapefile(s) into GeoDataFrame."""
    try:
        import geopandas as gpd
        frames = []
        for p in paths:
            try:
                gdf = gpd.read_file(p)
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs("EPSG:4326")
                frames.append(gdf)
                log.info("Loaded shapefile", file=p.name, rows=len(gdf))
            except Exception as e:
                log.warning("Failed to load shapefile", file=str(p), error=str(e))
        if not frames:
            return None
        import pandas as pd
        return pd.concat(frames, ignore_index=True)
    except ImportError:
        log.error("geopandas not installed")
        return None


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance in metres between two lat/lon points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _in_bounds(df, min_lon, min_lat, max_lon, max_lat,
               lat_col="_lat", lon_col="_lon"):
    """Filter a DataFrame to rows within bounding box."""
    return df[
        (df[lat_col] >= min_lat) & (df[lat_col] <= max_lat) &
        (df[lon_col] >= min_lon) & (df[lon_col] <= max_lon)
    ]


# ── Main agent ────────────────────────────────────────────────────────────────

class PointDataAgent(BaseAgent):
    """
    Ingests all simple coordinate-based datasets in one pass per cell.
    Add new data sources by adding a load block in __init__ and a
    process block in process_cell.
    """

    agent_name  = "point_data_agent"
    description = "Spatial point query agent — gravity, magnetics, faults, hydrology, mines, boreholes"

    def __init__(self, data_dir: Path = None):
        super().__init__()
        search = data_dir or RAW_DIR

        LAT = ["lat","latitude","y","lat_dd","latitude_dd"]
        LON = ["lon","long","longitude","x","lon_dd","longitude_dd"]

        # ── Gravity ───────────────────────────────────────────────────────────
        gravity_files = (list(search.glob("**/gravity*.csv")) +
                         list(search.glob("**/grav*.csv")) +
                         list(search.glob("gravity/**/*.csv")))
        self._gravity = _load_csv_points(gravity_files, LAT, LON)
        self._gravity_col = None
        if self._gravity is not None:
            for col in self._gravity.columns:
                if any(x in col.lower() for x in ("bouguer","grav","mgal","anomaly")):
                    self._gravity_col = col
                    break
            self.log.info("Gravity data loaded",
                          points=len(self._gravity), value_col=self._gravity_col)

        # ── Magnetics ─────────────────────────────────────────────────────────
        mag_files = (list(search.glob("**/magnet*.csv")) +
                     list(search.glob("**/mag*.csv")) +
                     list(search.glob("magnetics/**/*.csv")))
        self._magnetics = _load_csv_points(mag_files, LAT, LON)
        self._mag_col = None
        if self._magnetics is not None:
            for col in self._magnetics.columns:
                if any(x in col.lower() for x in ("mag","nt","nanotesla","intensity","field")):
                    self._mag_col = col
                    break
            self.log.info("Magnetics data loaded",
                          points=len(self._magnetics), value_col=self._mag_col)

        # ── Qfaults ───────────────────────────────────────────────────────────
        fault_files = (list(search.glob("**/qfault*.shp")) +
                       list(search.glob("**/fault*.shp")) +
                       list(search.glob("**/Qfault*.shp")) +
                       list(search.glob("faults/**/*.shp")))
        self._faults = _load_shapefile(fault_files)
        if self._faults is not None:
            self.log.info("Fault data loaded", features=len(self._faults))

        # ── Hydrology ─────────────────────────────────────────────────────────
        hydro_files = (list(search.glob("**/hydro*.shp")) +
                       list(search.glob("**/stream*.shp")) +
                       list(search.glob("**/river*.shp")) +
                       list(search.glob("hydrology/**/*.shp")))
        self._hydrology = _load_shapefile(hydro_files)
        if self._hydrology is not None:
            self.log.info("Hydrology data loaded", features=len(self._hydrology))

        # ── Historic mines ────────────────────────────────────────────────────
        mine_files = (list(search.glob("**/mine*.csv")) +
                      list(search.glob("**/mrds*.csv")) +
                      list(search.glob("**/deposit*.csv")) +
                      list(search.glob("**/mine*.shp")) +
                      list(search.glob("mines/**/*")))
        # Try CSV first, then shapefile
        csv_mines = [f for f in mine_files if f.suffix.lower() == ".csv"]
        shp_mines = [f for f in mine_files if f.suffix.lower() == ".shp"]
        self._mines = _load_csv_points(csv_mines, LAT, LON) if csv_mines else _load_shapefile(shp_mines)
        if self._mines is not None:
            self.log.info("Historic mines data loaded", points=len(self._mines))

        # ── Boreholes ─────────────────────────────────────────────────────────
        borehole_files = (list(search.glob("**/borehole*.csv")) +
                          list(search.glob("**/collar*.csv")) +
                          list(search.glob("**/drill*.csv")) +
                          list(search.glob("boreholes/**/*.csv")))
        self._boreholes = _load_csv_points(borehole_files, LAT, LON)
        if self._boreholes is not None:
            self.log.info("Borehole data loaded", points=len(self._boreholes))

        self.log.info("PointDataAgent ready",
                      has_gravity=self._gravity is not None,
                      has_magnetics=self._magnetics is not None,
                      has_faults=self._faults is not None,
                      has_hydrology=self._hydrology is not None,
                      has_mines=self._mines is not None,
                      has_boreholes=self._boreholes is not None)

    def process_cell(self, cell: GridCell, **kwargs) -> GridCell:
        b = (cell.min_lon, cell.min_lat, cell.max_lon, cell.max_lat)
        cx, cy = cell.centroid_lon, cell.centroid_lat
        summary = PointDataSummary()

        # ── Gravity ───────────────────────────────────────────────────────────
        if self._gravity is not None:
            sub = _in_bounds(self._gravity, *b)
            summary.gravity_sample_count = len(sub)
            if len(sub) > 0 and self._gravity_col:
                vals = sub[self._gravity_col].dropna()
                if len(vals) > 0:
                    summary.gravity_bouguer_mgal = round(float(vals.mean()), 4)
            if len(sub) > 0:
                dists = [_haversine_m(cy, cx, r._lat, r._lon)
                         for _, r in sub.iterrows()]
                summary.gravity_nearest_m = round(min(dists), 1)

        # ── Magnetics ─────────────────────────────────────────────────────────
        if self._magnetics is not None:
            sub = _in_bounds(self._magnetics, *b)
            summary.magnetic_sample_count = len(sub)
            if len(sub) > 0 and self._mag_col:
                vals = sub[self._mag_col].dropna()
                if len(vals) > 0:
                    summary.magnetic_intensity_nt = round(float(vals.mean()), 2)
                    if len(vals) > 1:
                        summary.magnetic_gradient = round(float(vals.std()), 2)

        # ── Faults ────────────────────────────────────────────────────────────
        if self._faults is not None:
            try:
                from shapely.geometry import box as sbox
                cell_box = sbox(cell.min_lon, cell.min_lat,
                                cell.max_lon, cell.max_lat)

                intersecting = self._faults[
                    self._faults.geometry.intersects(cell_box)
                ] if "geometry" in self._faults.columns else self._faults.iloc[0:0]

                summary.fault_count = len(intersecting)

                if len(intersecting) > 0:
                    # Find nearest fault
                    min_dist = float("inf")
                    nearest = None
                    for _, row in intersecting.iterrows():
                        try:
                            from shapely.geometry import Point
                            d = row.geometry.distance(Point(cx, cy)) * 111000
                            if d < min_dist:
                                min_dist = d
                                nearest = row
                        except Exception:
                            pass

                    if nearest is not None:
                        summary.nearest_fault_m = round(min_dist, 1)
                        # Try common Qfault column names
                        for col in ("fault_name","name","FAULT_NAME","NAME","flt_name"):
                            if col in nearest.index and nearest[col]:
                                summary.nearest_fault_name = str(nearest[col])
                                break
                        for col in ("slip_sense","type","SLIP_SENSE","fault_type","FLT_TYPE"):
                            if col in nearest.index and nearest[col]:
                                summary.nearest_fault_type = str(nearest[col])
                                break
                        for col in ("age","AGEGROUP","age_group","agegrp"):
                            if col in nearest.index and nearest[col]:
                                summary.nearest_fault_age = str(nearest[col])
                                break

                    # Fault density: total length / cell area
                    try:
                        total_len_deg = sum(
                            row.geometry.length
                            for _, row in intersecting.iterrows()
                            if row.geometry is not None
                        )
                        total_len_km = total_len_deg * 111
                        cell_area_km2 = (cell.cell_size_m / 1000) ** 2
                        summary.fault_density_km_per_km2 = round(
                            total_len_km / max(cell_area_km2, 1), 4
                        )
                    except Exception:
                        pass

            except Exception as e:
                self.log.warning("Fault processing failed",
                                 tile_id=cell.tile_id, error=str(e))

        # ── Hydrology ─────────────────────────────────────────────────────────
        if self._hydrology is not None:
            try:
                from shapely.geometry import box as sbox
                cell_box = sbox(cell.min_lon, cell.min_lat,
                                cell.max_lon, cell.max_lat)
                intersecting = self._hydrology[
                    self._hydrology.geometry.intersects(cell_box)
                ] if "geometry" in self._hydrology.columns else self._hydrology.iloc[0:0]

                summary.stream_count = len(intersecting)
                if len(intersecting) > 0:
                    for col in ("stream_ord","strahler","order","STREAMORDE"):
                        if col in intersecting.columns:
                            orders = intersecting[col].dropna()
                            if len(orders) > 0:
                                summary.stream_order_max = int(orders.max())
                            break
            except Exception as e:
                self.log.warning("Hydrology processing failed",
                                 tile_id=cell.tile_id, error=str(e))

        # ── Historic mines ────────────────────────────────────────────────────
        if self._mines is not None:
            sub = _in_bounds(self._mines, *b)
            summary.historic_mine_count = len(sub)
            if len(sub) > 0:
                dists = [_haversine_m(cy, cx, r._lat, r._lon)
                         for _, r in sub.iterrows()]
                min_idx = int(np.argmin(dists))
                summary.nearest_mine_m = round(dists[min_idx], 1)
                nearest_row = sub.iloc[min_idx]
                for col in ("name","mine_name","NAME","deposit_name"):
                    if col in nearest_row.index and nearest_row[col]:
                        summary.nearest_mine_name = str(nearest_row[col])
                        break
                for col in ("commodity","commod1","COMMOD1","primary_commodity"):
                    if col in nearest_row.index and nearest_row[col]:
                        summary.nearest_mine_commodity = str(nearest_row[col])
                        break

        # ── Boreholes ─────────────────────────────────────────────────────────
        if self._boreholes is not None:
            sub = _in_bounds(self._boreholes, *b)
            summary.borehole_count = len(sub)
            if len(sub) > 0:
                dists = [_haversine_m(cy, cx, r._lat, r._lon)
                         for _, r in sub.iterrows()]
                summary.nearest_borehole_m = round(min(dists), 1)
                for col in ("depth","total_depth","depth_m","max_depth"):
                    if col in sub.columns:
                        depths = sub[col].dropna()
                        if len(depths) > 0:
                            summary.max_borehole_depth_m = float(depths.max())
                        break

        # ── Depletion signal from point data ──────────────────────────────────
        depletion = 0.0
        if summary.historic_mine_count > 0:
            depletion += min(summary.historic_mine_count / 5, 1.0) * 0.6
        if summary.nearest_mine_m is not None and summary.nearest_mine_m < 1000:
            depletion += 0.3
        summary.depletion_score = round(min(depletion, 1.0), 3)

        cell.point_data = summary

        # Update opportunity score with depletion
        if cell.probability_score is not None:
            cell.opportunity_score = round(
                cell.probability_score * (1 - summary.depletion_score * 0.5), 4
            )

        self.log.info("Cell complete",
                      tile_id=cell.tile_id,
                      faults=summary.fault_count,
                      mines=summary.historic_mine_count,
                      boreholes=summary.borehole_count,
                      depletion=summary.depletion_score)
        return cell
