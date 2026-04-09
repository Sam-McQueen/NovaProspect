"""
sierra_prospector/agents/terrain_agent.py
=====================================
Terrain agent — processes DEM (Digital Elevation Model) files into
per-cell terrain summaries.

Reads:
    - USGS 3DEP 1/3 arc-second DEMs (usgs_3dep_13_*.tif)
    - OpenTopography COP30 / SRTM DEMs
    - Any standard single-band elevation GeoTIFF

Produces per cell:
    - Mean / min / max elevation (metres)
    - Mean slope (degrees)
    - Dominant aspect (N/NE/E/SE/S/SW/W/NW)
    - Terrain roughness index
    - Drainage proximity estimate
    - Topographic wetness index (TWI) — proxy for where water pools

Why terrain matters for gold prospecting:
    - Gold concentrates in specific topographic settings:
      * Benches above modern stream level (ancient placer deposits)
      * Fault-controlled ridges (lode gold pathways)
      * SW-facing slopes in Sierra Nevada (more exposed, less vegetated = better signal)
    - Slope and drainage density help distinguish lode vs placer targets
    - Elevation range within a cell indicates structural complexity

DEM files expected at:
    /mnt/c/Geodata/remote_sensing/usgs_3dep_13_*.tif
    /mnt/c/Geodata/remote_sensing/opentopo_*.tif
    Or wherever RAW_DIR points in config/settings.py
"""

import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.crs import CRS
from shapely.geometry import box, mapping
from pyproj import Transformer

from config.settings import RAW_DIR
from core.ontology import GridCell, CellStatus, TerrainSummary
from core.logger import get_logger
from agents.base_agent import BaseAgent

log = get_logger("terrain_agent")


# ── DEM file discovery ────────────────────────────────────────────────────────

def find_dem_files(dem_dir: Path = None) -> List[Path]:
    """
    Discover all DEM GeoTIFF files in the data directory.
    Looks for USGS 3DEP, OpenTopography COP30, SRTM, and generic DEM files.
    """
    search_dir = dem_dir or RAW_DIR

    if not search_dir.exists():
        log.warning("DEM search directory does not exist", path=str(search_dir))
        return []

    patterns = [
        "**/usgs_3dep*.tif",
        "**/opentopo*.tif",
        "**/srtm*.tif",
        "**/dem*.tif",
        "**/elevation*.tif",
        "**/DEM*.tif",
        "**/dem/**/*.tif",
    ]

    found = set()
    for pattern in patterns:
        for f in search_dir.glob(pattern):
            # Skip aux.xml sidecar files
            if ".aux" not in f.name:
                found.add(f)

    files = sorted(found)
    log.info("Discovered DEM files", count=len(files), path=str(search_dir))
    for f in files:
        log.debug("DEM file found", file=f.name)

    return files


# ── DEM reading ───────────────────────────────────────────────────────────────

def load_dem_for_bounds(
    dem_files: List[Path],
    bounds_wgs84: Tuple[float, float, float, float]
) -> Optional[np.ndarray]:
    """
    Load and merge DEM data for a bounding box from multiple tiles.

    USGS 3DEP files are named by lat/lon tile (e.g. usgs_3dep_13_n37w120.tif)
    so a single cell may span multiple tiles. This function handles that by
    trying all files and merging overlapping data.

    Returns a 2D float32 array of elevation values in metres, or None.
    Nodata pixels are set to NaN.
    """
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    query_box = box(min_lon, min_lat, max_lon, max_lat)
    geom = [mapping(query_box)]

    collected = []

    for dem_path in dem_files:
        try:
            with rasterio.open(dem_path) as src:
                scene_crs  = src.crs
                nodata_val = src.nodata

                # Reproject query bounds to DEM CRS if needed
                if scene_crs and scene_crs.to_epsg() != 4326:
                    t = Transformer.from_crs(
                        CRS.from_epsg(4326), scene_crs, always_xy=True
                    )
                    min_x, min_y = t.transform(min_lon, min_lat)
                    max_x, max_y = t.transform(max_lon, max_lat)
                    scene_query_box = box(min_x, min_y, max_x, max_y)
                else:
                    scene_query_box = query_box

                scene_box = box(*src.bounds)
                if not scene_query_box.intersects(scene_box):
                    continue

                clipped, _ = rio_mask(
                    src, geom,
                    crop=True,
                    indexes=[1],
                    nodata=nodata_val,
                    filled=True,
                )
                elev = clipped[0].astype(np.float32)

                # Mask nodata
                if nodata_val is not None:
                    elev[elev == nodata_val] = np.nan
                elev[elev < -500] = np.nan    # Ocean/void fill values
                elev[elev > 9000] = np.nan    # Above Everest = bad data

                valid_pct = np.sum(np.isfinite(elev)) / elev.size
                if valid_pct < 0.05:
                    continue

                collected.append(elev)
                log.debug("DEM tile loaded",
                          file=dem_path.name,
                          shape=elev.shape,
                          valid_pct=round(float(valid_pct), 3))

        except rasterio.errors.RasterioIOError as e:
            log.error("Failed to open DEM file",
                      file=str(dem_path),
                      error=str(e))
        except Exception as e:
            log.exception("Unexpected error reading DEM",
                          exc=e,
                          file=str(dem_path))

    if not collected:
        return None

    # If multiple tiles, take the one with most valid pixels
    # (merging properly requires rasterio.merge which adds complexity)
    best = max(collected, key=lambda a: np.sum(np.isfinite(a)))
    return best


# ── Terrain calculations ──────────────────────────────────────────────────────

def compute_slope(elev: np.ndarray, cell_size_m: float) -> Optional[np.ndarray]:
    """
    Compute slope in degrees from a DEM array.
    Uses central difference gradient — standard GIS method.
    Returns array of same shape with NaN where elevation is NaN.
    """
    if elev.shape[0] < 3 or elev.shape[1] < 3:
        return None

    # Pixel size in metres — approximate from cell size and array dimensions
    pixel_size_x = cell_size_m / elev.shape[1]
    pixel_size_y = cell_size_m / elev.shape[0]

    # numpy gradient returns [dy, dx]
    dy, dx = np.gradient(elev, pixel_size_y, pixel_size_x)

    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # Mask where elevation was NaN
    slope_deg[~np.isfinite(elev)] = np.nan
    return slope_deg


def compute_aspect(elev: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute aspect (direction slope faces) in degrees from North.
    0/360 = North, 90 = East, 180 = South, 270 = West.
    """
    if elev.shape[0] < 3 or elev.shape[1] < 3:
        return None

    dy, dx = np.gradient(elev)
    # Aspect in degrees, 0 = North
    aspect = np.degrees(np.arctan2(-dy, dx)) % 360
    aspect[~np.isfinite(elev)] = np.nan
    return aspect


def aspect_to_cardinal(mean_aspect_deg: float) -> str:
    """Convert mean aspect degrees to 8-point cardinal direction."""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((mean_aspect_deg + 22.5) / 45) % 8
    return directions[idx]


def compute_roughness(elev: np.ndarray) -> Optional[float]:
    """
    Terrain Roughness Index — std dev of elevation within the cell.
    Higher roughness = more structural complexity = often more interesting geology.
    """
    valid = elev[np.isfinite(elev)]
    if len(valid) < 10:
        return None
    return float(np.std(valid))


def compute_twi(elev: np.ndarray, cell_size_m: float) -> Optional[float]:
    """
    Simplified Topographic Wetness Index (TWI).
    TWI = ln(upslope_area / tan(slope))
    High TWI = water accumulates here = potential placer deposit zone.

    This is a cell-level approximation — proper TWI needs flow accumulation
    which requires the full watershed. This gives a useful proxy.
    """
    slope_arr = compute_slope(elev, cell_size_m)
    if slope_arr is None:
        return None

    valid_slope = slope_arr[np.isfinite(slope_arr) & (slope_arr > 0.1)]
    if len(valid_slope) < 10:
        return None

    # Simplified: use cell area as proxy for upslope area
    cell_area = cell_size_m ** 2
    mean_slope_rad = np.radians(float(np.nanmean(valid_slope)))

    if math.tan(mean_slope_rad) < 0.001:
        return None

    twi = math.log(cell_area / math.tan(mean_slope_rad))
    return round(twi, 3)


def estimate_drainage_proximity(elev: np.ndarray, cell_size_m: float) -> Optional[float]:
    """
    Estimate proximity to drainage (stream channels) in metres.
    Drainage channels occur at local elevation minima.
    Uses a simple local minimum detection as a proxy.

    Returns estimated distance to nearest drainage feature in metres.
    """
    if elev.shape[0] < 5 or elev.shape[1] < 5:
        return None

    valid = np.isfinite(elev)
    if valid.sum() < 20:
        return None

    # Find local minima using a simple neighborhood comparison
    from scipy.ndimage import minimum_filter
    local_min = minimum_filter(
        np.where(valid, elev, np.nanmax(elev[valid])),
        size=5
    )
    drainage_mask = (elev == local_min) & valid

    if drainage_mask.sum() == 0:
        return None

    # Distance from center pixel to nearest drainage pixel
    center_r = elev.shape[0] // 2
    center_c = elev.shape[1] // 2

    drain_rows, drain_cols = np.where(drainage_mask)
    if len(drain_rows) == 0:
        return None

    pixel_size = cell_size_m / max(elev.shape)
    distances = np.sqrt(
        ((drain_rows - center_r) * pixel_size) ** 2 +
        ((drain_cols - center_c) * pixel_size) ** 2
    )
    return round(float(np.min(distances)), 1)


def compute_probability_adjustment(summary: TerrainSummary) -> float:
    """
    Compute a terrain-based probability adjustment factor (0 to 1).
    This MODIFIES the spectral probability score — it doesn't replace it.

    High-value terrain indicators for Sierra Nevada gold:
    - Elevation 800-2500m (optimal gold district range)
    - Moderate slope 5-25° (not too flat, not too steep for placer)
    - SW or W aspect (more erosion = more exposure)
    - High roughness (structural complexity)
    - Moderate TWI (some drainage but not a swamp)
    """
    score = 0.5    # Neutral baseline
    factors = 0

    if summary.mean_elevation_m is not None:
        elev = summary.mean_elevation_m
        if 800 <= elev <= 2500:
            score += 0.15     # Sweet spot for Sierra gold districts
        elif 500 <= elev <= 3500:
            score += 0.05
        else:
            score -= 0.1      # Too low (valley) or too high (above treeline)
        factors += 1

    if summary.mean_slope_deg is not None:
        slope = summary.mean_slope_deg
        if 5 <= slope <= 25:
            score += 0.10     # Good placer slope range
        elif slope < 2:
            score -= 0.05     # Flat valley floor — likely placer exhausted
        elif slope > 40:
            score -= 0.05     # Too steep for significant accumulation
        factors += 1

    if summary.dominant_aspect is not None:
        if summary.dominant_aspect in ("SW", "W", "NW"):
            score += 0.05     # Classic Sierra Nevada productive aspect
        factors += 1

    if summary.terrain_roughness is not None:
        if summary.terrain_roughness > 100:
            score += 0.10     # High structural complexity
        elif summary.terrain_roughness > 50:
            score += 0.05
        factors += 1

    return round(min(max(score, 0.0), 1.0), 3)


# ── Main agent class ──────────────────────────────────────────────────────────

class TerrainAgent(BaseAgent):
    """
    Processes DEM elevation files for each grid cell.

    Discovers all DEM files at startup.
    For each cell, clips DEM to cell bounds, computes terrain metrics,
    writes TerrainSummary to GridCell.
    """

    agent_name  = "terrain_agent"
    description = "Computes terrain features from DEM elevation data"

    def __init__(self, dem_dir: Path = None):
        super().__init__()
        self._dem_files = find_dem_files(dem_dir)
        self.log.info("TerrainAgent ready",
                      dem_count=len(self._dem_files))

        if not self._dem_files:
            self.log.warning(
                "No DEM files found. Place GeoTIFF elevation files in the data directory. "
                "Agent will return empty summaries until DEMs are available."
            )

    def process_cell(self, cell: GridCell, **kwargs) -> GridCell:
        """
        For one GridCell:
          1. Load DEM data clipped to cell bounds
          2. Compute elevation statistics
          3. Compute slope and aspect
          4. Compute roughness and TWI
          5. Attach TerrainSummary to cell
          6. Adjust probability score
        """
        bounds = (cell.min_lon, cell.min_lat, cell.max_lon, cell.max_lat)

        self.log.debug("Processing cell",
                       tile_id=cell.tile_id,
                       level=cell.level,
                       cell_size_m=cell.cell_size_m,
                       bounds=bounds)

        # ── Load elevation data ───────────────────────────────────────────────
        elev = load_dem_for_bounds(self._dem_files, bounds)

        if elev is None:
            self.log.warning("No DEM data for cell",
                             tile_id=cell.tile_id,
                             dems_tried=len(self._dem_files))
            cell.terrain = TerrainSummary()
            cell.status  = CellStatus.COMPLETE
            return cell

        valid = elev[np.isfinite(elev)]
        if len(valid) < 10:
            self.log.warning("Insufficient valid DEM pixels",
                             tile_id=cell.tile_id,
                             valid_pixels=len(valid))
            cell.terrain = TerrainSummary()
            cell.status  = CellStatus.COMPLETE
            return cell

        # ── Elevation statistics ──────────────────────────────────────────────
        mean_elev = float(np.nanmean(valid))
        min_elev  = float(np.nanmin(valid))
        max_elev  = float(np.nanmax(valid))

        self.log.debug("Elevation stats",
                       tile_id=cell.tile_id,
                       mean=round(mean_elev, 1),
                       min=round(min_elev, 1),
                       max=round(max_elev, 1))

        # ── Slope ─────────────────────────────────────────────────────────────
        slope_arr = compute_slope(elev, cell.cell_size_m)
        mean_slope = None
        if slope_arr is not None:
            valid_slope = slope_arr[np.isfinite(slope_arr)]
            if len(valid_slope) > 0:
                mean_slope = round(float(np.nanmean(valid_slope)), 2)

        # ── Aspect ────────────────────────────────────────────────────────────
        aspect_arr = compute_aspect(elev)
        dominant_aspect = None
        if aspect_arr is not None:
            valid_aspect = aspect_arr[np.isfinite(aspect_arr)]
            if len(valid_aspect) > 0:
                mean_aspect = float(np.nanmean(valid_aspect))
                dominant_aspect = aspect_to_cardinal(mean_aspect)

        # ── Roughness ─────────────────────────────────────────────────────────
        roughness = compute_roughness(elev)

        # ── TWI ───────────────────────────────────────────────────────────────
        twi = compute_twi(elev, cell.cell_size_m)

        # ── Drainage proximity ────────────────────────────────────────────────
        try:
            drainage_proximity = estimate_drainage_proximity(elev, cell.cell_size_m)
        except Exception as e:
            self.log.warning("Drainage proximity failed",
                             tile_id=cell.tile_id,
                             error=str(e))
            drainage_proximity = None

        # ── Assemble summary ──────────────────────────────────────────────────
        summary = TerrainSummary(
            mean_elevation_m  = round(mean_elev, 1),
            max_elevation_m   = round(max_elev, 1),
            min_elevation_m   = round(min_elev, 1),
            mean_slope_deg    = mean_slope,
            dominant_aspect   = dominant_aspect,
            drainage_density  = None,        # Needs full watershed analysis
            nearest_ridge_m   = None,        # Needs ridge detection
            nearest_drainage_m = drainage_proximity,
            topographic_wetness = twi,
        )

        # Store roughness — add it to TerrainSummary if not already there
        # We attach it as a dynamic attribute for now
        summary_dict = summary.__dict__
        summary.terrain_roughness = roughness

        cell.terrain = summary

        # ── Update probability score ──────────────────────────────────────────
        terrain_factor = compute_probability_adjustment(summary)

        if cell.probability_score is not None:
            # Blend existing score with terrain factor
            cell.probability_score = round(
                cell.probability_score * 0.6 + terrain_factor * 0.4, 4
            )
        else:
            # Terrain is the only signal so far
            cell.probability_score = terrain_factor

        cell.opportunity_score = cell.probability_score
        cell.status = CellStatus.COMPLETE

        self.log.info("Cell complete",
                      tile_id=cell.tile_id,
                      mean_elev_m=round(mean_elev, 0),
                      slope_deg=mean_slope,
                      aspect=dominant_aspect,
                      roughness=round(roughness, 1) if roughness else None,
                      terrain_factor=terrain_factor,
                      probability=cell.probability_score)

        return cell
