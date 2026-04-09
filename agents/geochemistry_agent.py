"""
sierra_prospector/agents/geochemistry_agent.py
=====================================
Geochemistry agent — processes USGS stream sediment and soil geochemistry
data into per-cell anomaly scores.

Reads:
    - USGS National Geochemical Survey CSV/shapefile
    - Any point-based geochemistry file with lat/lon + element values

Produces per cell:
    - Au anomaly score (0-1 normalised)
    - Au ppb raw values
    - Pathfinder element scores: As, Sb, Hg, Ag
    - Sample count and proximity

Why geochemistry matters:
    Gold disperses downstream from its source. A USGS stream sediment sample
    showing elevated Au ppb 2km downstream from your cell is a strong signal
    that the source is somewhere upslope in your cell.
    Pathfinder elements (As, Sb, Hg) disperse further than gold itself —
    they show up in samples even when Au is below detection.

Data expected at:
    /mnt/c/Geodata/remote_sensing/usgs/ or any subfolder
    Files: *.csv, *.shp with columns including lat, lon, Au_ppm or Au_ppb
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np

from config.settings import RAW_DIR, EXCLUSION_THRESHOLD
from core.ontology import GridCell, CellStatus, GeochemistrySummary
from core.logger import get_logger
from agents.base_agent import BaseAgent

log = get_logger("geochemistry_agent")


def find_geochemistry_files(search_dir: Path = None) -> List[Path]:
    """Find geochemistry CSV and shapefile data."""
    search_dir = search_dir or RAW_DIR
    if not search_dir.exists():
        return []

    files = []
    for pattern in ["**/*geochem*.csv", "**/*geochemistry*.csv",
                    "**/*sediment*.csv", "**/*usgs*.csv",
                    "**/*geochem*.shp", "**/*sediment*.shp",
                    "usgs/**/*.csv", "usgs/**/*.shp"]:
        files.extend(search_dir.glob(pattern))

    log.info("Discovered geochemistry files",
             count=len(files), path=str(search_dir))
    return list(set(files))


def load_geochemistry_points(files: List[Path]) -> Optional[object]:
    """
    Load all geochemistry point data into a GeoDataFrame.
    Tries to auto-detect column names for lat/lon and element values.
    Returns None if no files or import fails.
    """
    try:
        import pandas as pd
        import geopandas as gpd
        from shapely.geometry import Point

        all_frames = []

        for f in files:
            try:
                if f.suffix.lower() == ".csv":
                    df = pd.read_csv(f, low_memory=False)

                    # Auto-detect lat/lon columns
                    lat_col = next((c for c in df.columns
                                   if c.lower() in ("lat","latitude","y","lat_dd")), None)
                    lon_col = next((c for c in df.columns
                                   if c.lower() in ("lon","long","longitude","x","lon_dd")), None)

                    if not lat_col or not lon_col:
                        log.warning("Could not find lat/lon columns",
                                    file=f.name, columns=list(df.columns[:10]))
                        continue

                    df = df.dropna(subset=[lat_col, lon_col])
                    gdf = gpd.GeoDataFrame(
                        df,
                        geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
                        crs="EPSG:4326"
                    )
                    all_frames.append(gdf)
                    log.info("Loaded geochemistry file",
                             file=f.name, rows=len(gdf))

                elif f.suffix.lower() == ".shp":
                    gdf = gpd.read_file(f)
                    if gdf.crs and gdf.crs.to_epsg() != 4326:
                        gdf = gdf.to_crs("EPSG:4326")
                    all_frames.append(gdf)
                    log.info("Loaded geochemistry shapefile",
                             file=f.name, rows=len(gdf))

            except Exception as e:
                log.warning("Failed to load geochemistry file",
                            file=str(f), error=str(e))

        if not all_frames:
            return None

        import pandas as pd
        combined = pd.concat(all_frames, ignore_index=True)
        log.info("Geochemistry data combined", total_points=len(combined))
        return combined

    except ImportError as e:
        log.error("Missing dependency for geochemistry", error=str(e))
        return None


def find_au_column(df) -> Optional[str]:
    """Auto-detect gold column name."""
    for col in df.columns:
        if col.lower() in ("au","au_ppm","au_ppb","gold","au_ppm_icp",
                           "au_ppb_fire","au_fa_ppb"):
            return col
    return None


def find_element_column(df, element: str) -> Optional[str]:
    """Auto-detect a pathfinder element column."""
    for col in df.columns:
        if col.lower().startswith(element.lower()):
            return col
    return None


def normalise_to_score(values: np.ndarray, high_percentile: float = 95.0) -> np.ndarray:
    """
    Normalise raw geochemical values to 0-1 anomaly scores.
    Uses percentile-based normalisation so regional background is suppressed.
    """
    if len(values) == 0:
        return np.array([])
    p95 = np.percentile(values, high_percentile)
    if p95 <= 0:
        return np.zeros_like(values, dtype=float)
    return np.clip(values / p95, 0, 1)


class GeochemistryAgent(BaseAgent):
    """
    Processes USGS geochemistry point data for each grid cell.
    Finds all sample points within cell bounds, computes anomaly scores.
    """

    agent_name  = "geochemistry_agent"
    description = "Computes Au and pathfinder element anomaly scores from USGS geochemistry"

    def __init__(self, data_dir: Path = None):
        super().__init__()
        files = find_geochemistry_files(data_dir)
        self._data = load_geochemistry_points(files) if files else None

        if self._data is None:
            self.log.warning(
                "No geochemistry data loaded. Place USGS geochemistry CSV/shapefiles "
                "in data directory. Agent will return empty summaries."
            )
        else:
            self._au_col = find_au_column(self._data)
            self.log.info("GeochemistryAgent ready",
                          total_points=len(self._data),
                          au_column=self._au_col)

    def process_cell(self, cell: GridCell, **kwargs) -> GridCell:
        if self._data is None:
            cell.geochemistry = GeochemistrySummary()
            return cell

        # Clip to cell bounds
        in_cell = self._data[
            (self._data.geometry.x >= cell.min_lon) &
            (self._data.geometry.x <= cell.max_lon) &
            (self._data.geometry.y >= cell.min_lat) &
            (self._data.geometry.y <= cell.max_lat)
        ]

        if len(in_cell) == 0:
            self.log.debug("No geochemistry samples in cell",
                           tile_id=cell.tile_id)
            cell.geochemistry = GeochemistrySummary(au_sample_count=0)
            return cell

        summary = GeochemistrySummary(au_sample_count=len(in_cell))

        # Au values
        if self._au_col and self._au_col in in_cell.columns:
            au_vals = in_cell[self._au_col].dropna().values
            au_vals = au_vals[au_vals > 0]
            if len(au_vals) > 0:
                # Convert ppm to ppb if needed
                if au_vals.max() < 10:   # Likely ppm not ppb
                    au_vals = au_vals * 1000
                summary.au_ppb_max  = round(float(au_vals.max()), 2)
                summary.au_ppb_mean = round(float(au_vals.mean()), 2)
                # Score relative to 95th percentile of all data
                all_au = self._data[self._au_col].dropna().values
                all_au = all_au[all_au > 0]
                if len(all_au) > 0:
                    p95 = float(np.percentile(all_au, 95))
                    summary.au_anomaly_score = round(
                        min(float(au_vals.max()) / max(p95, 1e-9), 1.0), 4
                    )

        # Pathfinder elements
        for elem, attr in [("As","as_anomaly_score"),("Sb","sb_anomaly_score"),
                            ("Ag","ag_anomaly_score"),("Hg","hg_anomaly_score")]:
            col = find_element_column(in_cell, elem)
            if col:
                vals = in_cell[col].dropna().values
                vals = vals[vals > 0]
                if len(vals) > 0:
                    all_vals = self._data[col].dropna().values
                    all_vals = all_vals[all_vals > 0]
                    p95 = float(np.percentile(all_vals, 95)) if len(all_vals) > 0 else 1.0
                    setattr(summary, attr, round(min(float(vals.max()) / max(p95, 1e-9), 1.0), 4))

        cell.geochemistry = summary

        # Update probability
        if summary.au_anomaly_score is not None:
            prior = cell.probability_score or 0.0
            cell.probability_score = round(prior * 0.7 + summary.au_anomaly_score * 0.3, 4)
            cell.opportunity_score = cell.probability_score

        self.log.info("Cell complete",
                      tile_id=cell.tile_id,
                      samples=summary.au_sample_count,
                      au_ppb_max=summary.au_ppb_max,
                      au_score=summary.au_anomaly_score)
        return cell
