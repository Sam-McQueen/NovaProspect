"""
sierra_prospector/agents/spectral_agent.py
=====================================
Spectral agent — processes Landsat 8/9 imagery into per-cell mineral alteration summaries.

This is the first and most critical agent in the pipeline.
It reads raw GeoTIFF raster data and produces structured SpectralSummary objects
that the reasoning LLM can consume without ever touching a pixel.

Key outputs per cell:
  - Iron oxide ratio         (gossan / oxidised sulphides indicator)
  - Hydroxyl / clay ratio    (hydrothermal alteration indicator)
  - Ferric iron ratio        (oxidised pyrite indicator)
  - NDVI                     (vegetation mask — high NDVI suppresses false positives)
  - Alteration type          (argillic | propylitic | silicic | unaltered)
  - Band statistics          (mean, std, valid_pixel_pct per band)

Multi-resolution strategy:
  - Coarse levels (0-5): median-pooled pixel values over the full cell
  - Fine levels (6+): per-pixel statistics with spatial variance included

Landsat scenes are expected in:
  data/raw/landsat/{scene_id}/{scene_id}_B{n}.TIF  (individual bands)
  OR
  data/raw/landsat/{scene_id}/{scene_id}_stack.TIF  (pre-stacked multiband GeoTIFF)
"""

import os
import glob
import json
import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform
import pyproj
from pyproj import Transformer

from config.settings import (
    RAW_DIR, LANDSAT_BANDS, SPECTRAL_INDICES,
    EXCLUSION_THRESHOLD, DRILL_THRESHOLD, CRS_WGS84
)
from core.ontology import GridCell, CellStatus, SpectralSummary
from core.logger import get_logger
from agents.base_agent import BaseAgent

log = get_logger("spectral_agent")


# ── Landsat scene discovery ───────────────────────────────────────────────────

def find_landsat_scenes(landsat_dir: Path = None) -> List[Path]:
    """
    Discover all Landsat scenes in the raw data directory.
    Returns a list of paths to stacked GeoTIFFs or band directories.
    """
    landsat_dir = landsat_dir or (RAW_DIR / "landsat")

    if not landsat_dir.exists():
        log.warning("Landsat directory does not exist", path=str(landsat_dir))
        return []

    # Look for pre-stacked files first (fastest)
    stacks = list(landsat_dir.glob("**/*_stack.TIF")) + \
             list(landsat_dir.glob("**/*_stack.tif")) + \
             list(landsat_dir.glob("**/*.tif"))

    log.info("Discovered Landsat scenes",
             count=len(stacks),
             path=str(landsat_dir))
    return stacks


def load_band_data(
    scene_path: Path,
    band_indices: List[int],
    window_bounds_wgs84: Tuple[float, float, float, float]
) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Load specific bands from a GeoTIFF, clipped to the given WGS84 bounding box.

    Returns:
        (data_array, metadata) where data_array has shape (n_bands, height, width)
        Returns (None, None) if the scene doesn't overlap the bounds.

    Handles:
        - Different CRS than WGS84 (reprojects bounds)
        - Missing bands (returns NaN slice)
        - All-nodata windows (returns None)
        - Partial overlaps (returns clipped data with NaN padding)
    """
    min_lon, min_lat, max_lon, max_lat = window_bounds_wgs84

    try:
        with rasterio.open(scene_path) as src:
            scene_crs = src.crs

            # Reproject query bounds to the scene's CRS
            if scene_crs and scene_crs.to_epsg() != 4326:
                transformer = Transformer.from_crs(
                    CRS.from_epsg(4326), scene_crs, always_xy=True
                )
                min_x, min_y = transformer.transform(min_lon, min_lat)
                max_x, max_y = transformer.transform(max_lon, max_lat)
            else:
                min_x, min_y, max_x, max_y = min_lon, min_lat, max_lon, max_lat

            # Check overlap with scene extent
            scene_bounds = src.bounds
            query_box = box(min_x, min_y, max_x, max_y)
            scene_box = box(*scene_bounds)

            if not query_box.intersects(scene_box):
                log.debug("Cell does not intersect this scene",
                          scene=scene_path.name,
                          cell_bounds=window_bounds_wgs84)
                return None, None

            # Clip to cell bounds using rasterio mask
            geom = [mapping(query_box)]
            data_bands = []
            nodata_val = src.nodata

            for band_idx in band_indices:
                if band_idx > src.count:
                    log.warning("Band index exceeds scene band count",
                                band=band_idx,
                                scene_bands=src.count,
                                scene=scene_path.name)
                    data_bands.append(None)
                    continue

                try:
                    clipped, _ = rio_mask(
                        src, geom,
                        crop=True,
                        indexes=[band_idx],
                        nodata=nodata_val,
                        filled=True
                    )
                    band_data = clipped[0].astype(np.float32)

                    # Mask nodata pixels → NaN
                    if nodata_val is not None:
                        band_data[band_data == nodata_val] = np.nan
                    # Mask zero pixels (common in Landsat edge fill)
                    band_data[band_data <= 0] = np.nan

                    data_bands.append(band_data)

                except Exception as e:
                    log.warning("Failed to read band",
                                band=band_idx,
                                scene=scene_path.name,
                                error=str(e))
                    data_bands.append(None)

            # Check if we got anything useful
            valid_bands = [b for b in data_bands if b is not None]
            if not valid_bands:
                log.debug("No valid bands in clip window", scene=scene_path.name)
                return None, None

            # Fill missing bands with NaN arrays of same shape
            ref_shape = valid_bands[0].shape
            for i, band in enumerate(data_bands):
                if band is None:
                    data_bands[i] = np.full(ref_shape, np.nan, dtype=np.float32)

            stacked = np.stack(data_bands, axis=0)   # (n_bands, H, W)

            # Check overall valid pixel coverage
            valid_pct = np.sum(np.isfinite(stacked)) / stacked.size
            if valid_pct < 0.05:
                log.debug("Insufficient valid pixels in clip",
                          valid_pct=round(valid_pct, 3),
                          scene=scene_path.name)
                return None, None

            meta = {
                "scene_path":    str(scene_path),
                "scene_crs":     str(scene_crs),
                "valid_pct":     round(float(valid_pct), 4),
                "shape":         list(stacked.shape),
                "nodata":        nodata_val,
            }
            return stacked, meta

    except rasterio.errors.RasterioIOError as e:
        log.error("RasterioIOError opening scene",
                  scene=str(scene_path),
                  error=str(e))
        return None, None
    except Exception as e:
        log.exception("Unexpected error reading scene",
                      exc=e,
                      scene=str(scene_path))
        return None, None


# ── Spectral index computation ────────────────────────────────────────────────

def compute_band_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    clip_percentile: float = 98.0
) -> Optional[float]:
    """
    Compute the median band ratio (num/denom) over valid pixels.
    Clips extreme values at the given percentile to remove outliers.
    Returns None if insufficient valid pixels.
    """
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    if valid.sum() < 10:
        return None

    ratio = numerator[valid] / denominator[valid]

    # Remove outliers
    if len(ratio) > 0:
        clip_val = np.percentile(ratio, clip_percentile)
        ratio = ratio[ratio <= clip_val]

    return float(np.nanmedian(ratio)) if len(ratio) > 0 else None


def compute_band_stats(band: np.ndarray) -> Dict[str, float]:
    """Per-band statistics for one clipped array."""
    valid = band[np.isfinite(band)]
    if len(valid) == 0:
        return {"mean": None, "std": None, "median": None,
                "p10": None, "p90": None, "valid_pct": 0.0}
    return {
        "mean":      float(np.mean(valid)),
        "std":       float(np.std(valid)),
        "median":    float(np.median(valid)),
        "p10":       float(np.percentile(valid, 10)),
        "p90":       float(np.percentile(valid, 90)),
        "valid_pct": float(len(valid) / band.size),
    }


def classify_alteration(
    iron_oxide: Optional[float],
    hydroxyl:   Optional[float],
    clay_alt:   Optional[float],
    ndvi:       Optional[float]
) -> Tuple[Optional[str], Optional[float]]:
    """
    Rule-based alteration classification based on spectral ratios.
    Returns (alteration_type, confidence).

    Rules derived from:
    - Crosta & Moore (1989) principal component analysis of Landsat TM
    - Mars & Rowan (2006) Sierra Nevada hydrothermal alteration mapping

    This is the 'dumb' first-pass classifier. The reasoning LLM will refine these.
    """
    if all(v is None for v in [iron_oxide, hydroxyl, clay_alt]):
        return None, None

    # Suppress vegetation-covered areas — forest canopy swamps spectral signal
    if ndvi is not None and ndvi > 0.6:
        return "vegetation_masked", 0.9

    scores = {}

    # Argillic alteration: strong clay + moderate iron oxide
    argillic_score = 0.0
    if hydroxyl is not None and hydroxyl > 1.5:
        argillic_score += 0.5
    if clay_alt is not None and clay_alt > 1.2:
        argillic_score += 0.3
    if iron_oxide is not None and iron_oxide > 1.8:
        argillic_score += 0.2
    scores["argillic"] = argillic_score

    # Propylitic: weak clay, low iron oxide
    propylitic_score = 0.0
    if hydroxyl is not None and 1.1 < hydroxyl < 1.5:
        propylitic_score += 0.5
    if iron_oxide is not None and iron_oxide < 1.5:
        propylitic_score += 0.3
    scores["propylitic"] = propylitic_score

    # Silicic / gossan: very high iron oxide
    silicic_score = 0.0
    if iron_oxide is not None and iron_oxide > 2.5:
        silicic_score += 0.7
    if hydroxyl is not None and hydroxyl < 1.3:
        silicic_score += 0.2
    scores["silicic"] = silicic_score

    best_type  = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score < 0.25:
        return "unaltered", 1.0 - best_score

    return best_type, round(min(best_score, 1.0), 3)


def compute_probability_score(summary: SpectralSummary) -> Tuple[float, float]:
    """
    Compute a composite gold probability score from spectral indices.
    Returns (probability_score, confidence).

    This is a simple linear combination — the LLM does the nuanced synthesis.
    Higher scores = more likely to be worth investigating.
    """
    score    = 0.0
    weights  = 0.0
    n_valid  = 0

    def _add(value, threshold, weight, high_is_good=True):
        nonlocal score, weights, n_valid
        if value is None:
            return
        if high_is_good:
            contribution = min(value / threshold, 1.5) * weight
        else:
            contribution = max(0, 1 - value / threshold) * weight
        score   += contribution
        weights += weight
        n_valid += 1

    _add(summary.iron_oxide_ratio,  2.0, 0.30, high_is_good=True)
    _add(summary.hydroxyl_ratio,    1.5, 0.25, high_is_good=True)
    _add(summary.gossan_ratio,      1.8, 0.20, high_is_good=True)
    _add(summary.ferric_iron_ratio, 1.3, 0.15, high_is_good=True)
    _add(summary.ndvi,              0.4, 0.10, high_is_good=False)   # Vegetation = bad signal

    if weights == 0:
        return 0.0, 0.0

    raw_prob   = score / weights
    prob       = round(min(raw_prob / 1.5, 1.0), 4)   # Normalise to 0-1
    confidence = round(n_valid / 5, 3)                  # Fraction of indices computed

    return prob, confidence


# ── Main agent class ──────────────────────────────────────────────────────────

class SpectralAgent(BaseAgent):
    """
    Processes Landsat imagery for each grid cell.

    Discovers all scenes in data/raw/landsat/ at startup.
    For each cell, finds overlapping scenes, reads bands, computes indices.
    Writes SpectralSummary back to GridCell.
    """

    agent_name  = "spectral_agent"
    description = "Computes mineral alteration indices from Landsat imagery"

    def __init__(self, landsat_dir: Path = None):
        super().__init__()
        self._scenes = find_landsat_scenes(landsat_dir)
        self.log.info("SpectralAgent ready", scene_count=len(self._scenes))

        if not self._scenes:
            self.log.warning(
                "No Landsat scenes found. Place GeoTIFF stacks in data/raw/landsat/. "
                "Agent will return empty summaries until scenes are available."
            )

    def process_cell(self, cell: GridCell, **kwargs) -> GridCell:
        """
        Main entry point. For one GridCell:
          1. Find overlapping Landsat scenes
          2. Load and clip relevant bands
          3. Compute spectral indices
          4. Classify alteration type
          5. Score probability
          6. Attach SpectralSummary to cell
        """
        bounds = (cell.min_lon, cell.min_lat, cell.max_lon, cell.max_lat)

        self.log.debug("Processing cell",
                       tile_id=cell.tile_id,
                       level=cell.level,
                       cell_size_m=cell.cell_size_m,
                       bounds=bounds)

        # ── Band indices we need ──────────────────────────────────────────────
        # Using Landsat 8/9 band numbering
        needed_bands = [
            LANDSAT_BANDS["blue"],    # B2
            LANDSAT_BANDS["green"],   # B3
            LANDSAT_BANDS["red"],     # B4
            LANDSAT_BANDS["nir"],     # B5
            LANDSAT_BANDS["swir1"],   # B6
            LANDSAT_BANDS["swir2"],   # B7
        ]
        band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]

        # ── Try each scene until we get valid data ────────────────────────────
        best_data = None
        best_meta = None
        best_valid_pct = 0.0

        for scene_path in self._scenes:
            self.log.debug("Trying scene", scene=scene_path.name, tile_id=cell.tile_id)

            data, meta = load_band_data(scene_path, needed_bands, bounds)

            if data is None:
                continue

            valid_pct = meta.get("valid_pct", 0.0)
            if valid_pct > best_valid_pct:
                best_data      = data
                best_meta      = meta
                best_valid_pct = valid_pct
                self.log.debug("Better scene found",
                               scene=scene_path.name,
                               valid_pct=valid_pct,
                               tile_id=cell.tile_id)

            # If we have >85% valid pixels, this scene is good enough
            if valid_pct > 0.85:
                break

        # ── Handle no valid data ──────────────────────────────────────────────
        if best_data is None:
            self.log.warning("No valid Landsat data for cell",
                             tile_id=cell.tile_id,
                             scenes_tried=len(self._scenes))
            summary = SpectralSummary(
                valid_pixel_pct  = 0.0,
                alteration_type  = "no_data",
                alteration_confidence = 0.0,
            )
            cell.spectral = summary
            cell.probability_score = 0.0
            cell.confidence        = 0.0
            # Don't exclude — maybe data will be available later
            cell.status = CellStatus.COMPLETE
            return cell

        # ── Unpack bands ──────────────────────────────────────────────────────
        b_blue, b_green, b_red, b_nir, b_swir1, b_swir2 = [best_data[i] for i in range(6)]

        # ── Compute spectral ratios ───────────────────────────────────────────
        iron_oxide_ratio  = compute_band_ratio(b_red,   b_blue)   # B4/B2
        hydroxyl_ratio    = compute_band_ratio(b_swir1, b_swir2)  # B6/B7
        ferric_iron_ratio = compute_band_ratio(b_swir1, b_nir)    # B6/B5
        clay_alteration   = compute_band_ratio(b_swir1, b_nir)    # B6/B5 (same as ferric; used separately)
        gossan_ratio      = compute_band_ratio(
            b_red + b_swir1, b_green + b_nir               # (B4+B6)/(B3+B5)
        )

        # NDVI
        ndvi_num = b_nir - b_red
        ndvi_den = b_nir + b_red
        ndvi = compute_band_ratio(ndvi_num, ndvi_den) if True else None
        # Direct NDVI computation (median over valid pixels)
        valid_ndvi = np.isfinite(b_nir) & np.isfinite(b_red) & ((b_nir + b_red) > 0)
        if valid_ndvi.sum() > 10:
            ndvi = float(np.nanmedian(
                (b_nir[valid_ndvi] - b_red[valid_ndvi]) /
                (b_nir[valid_ndvi] + b_red[valid_ndvi])
            ))
        else:
            ndvi = None

        # ── Per-band stats ────────────────────────────────────────────────────
        band_stats = {}
        for name, arr in zip(band_names, [b_blue, b_green, b_red, b_nir, b_swir1, b_swir2]):
            band_stats[name] = compute_band_stats(arr)

        self.log.debug("Indices computed",
                       tile_id=cell.tile_id,
                       iron_oxide=iron_oxide_ratio,
                       hydroxyl=hydroxyl_ratio,
                       ndvi=ndvi,
                       valid_pct=best_valid_pct)

        # ── Classify alteration ───────────────────────────────────────────────
        alteration_type, alteration_confidence = classify_alteration(
            iron_oxide_ratio, hydroxyl_ratio, clay_alteration, ndvi
        )

        # ── Cloud cover estimate ──────────────────────────────────────────────
        # Rough estimate: pixels with all bands very high reflectance = likely cloud
        if b_blue is not None:
            cloud_mask = (b_blue > 2000) & (b_green > 2000) & np.isfinite(b_blue)
            cloud_cover_pct = float(cloud_mask.sum() / b_blue.size * 100)
        else:
            cloud_cover_pct = None

        # ── Assemble summary ──────────────────────────────────────────────────
        summary = SpectralSummary(
            iron_oxide_ratio      = iron_oxide_ratio,
            hydroxyl_ratio        = hydroxyl_ratio,
            ferric_iron_ratio     = ferric_iron_ratio,
            gossan_ratio          = gossan_ratio,
            ndvi                  = ndvi,
            clay_alteration       = clay_alteration,
            band_stats            = band_stats,
            alteration_type       = alteration_type,
            alteration_confidence = alteration_confidence,
            cloud_cover_pct       = cloud_cover_pct,
            valid_pixel_pct       = best_valid_pct,
            landsat_scene_id      = Path(best_meta["scene_path"]).name,
        )

        cell.spectral = summary

        # ── Score ─────────────────────────────────────────────────────────────
        prob, conf = compute_probability_score(summary)
        cell.probability_score = prob
        cell.confidence        = conf
        # opportunity_score gets updated after history agent runs (accounts for depletion)
        cell.opportunity_score = prob   # Preliminary — will be refined

        # ── Exclusion logic ───────────────────────────────────────────────────
        if prob < EXCLUSION_THRESHOLD and conf > 0.5:
            cell.status = CellStatus.EXCLUDED
            self.log.debug("Cell excluded — low spectral probability",
                           tile_id=cell.tile_id,
                           probability=prob,
                           confidence=conf)
        else:
            cell.status = CellStatus.COMPLETE

        self.log.info("Cell complete",
                      tile_id=cell.tile_id,
                      alteration=alteration_type,
                      probability=prob,
                      confidence=conf,
                      status=cell.status)

        return cell
