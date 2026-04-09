"""
sierra_prospector/agents/lidar_agent.py
=====================================
LiDAR agent — processes Sierra Nevada .las point cloud files at native resolution.

Architecture:
    Each LiDAR tile (~100m) is processed as its OWN grid cell at level 8 (100m).
    Results are written to the appropriate level 8 cell and linked up the pyramid.
    Parent cells (levels 0-7) inherit LiDAR summaries via child aggregation.

    This is correct because:
    - LiDAR tiles are ~100m — forcing them into 50km cells loses all value
    - Processing 50km of LiDAR at once would crash most machines
    - Native resolution preserves the 10-25cm vertical accuracy advantage

Renders per tile (7 images sent to Grok):
    1. Hillshade NW  (315°) — standard reference
    2. Hillshade NE  (45°)  — reveals NW-SE features
    3. Hillshade S   (180°) — perpendicular to Sierra structural grain
    4. 3D perspective       — oblique view like Google Maps 3D
    5. Slope map            — bench and terrace detection
    6. Local relief model   — depletion/anomaly detection (best for hydraulic scars)
    7. Intensity map        — rock vs vegetation vs water discrimination

LiDAR files: /mnt/c/Geodata/lidar/Sierra25LasFiles/Sierra25_NNN.zip
Each zip contains one .las file. Extracted on-demand, processed, deleted.
"""

import os
import re
import shutil
import zipfile
import time
import base64
import io
import math
import json
import requests
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from config.settings import GROK_API_KEY, GROK_API_BASE, ACTIVE_CONFIG
from core.ontology import GridCell, CellStatus
from core.logger import get_logger
from agents.base_agent import BaseAgent

log = get_logger("lidar_agent")

LIDAR_DIR       = Path("/mnt/c/Geodata/lidar/Sierra25LasFiles")
LIDAR_TEMP      = Path("/tmp/lidar_extract")
LIDAR_INDEX_PATH= Path("/home/placer/sierra_prospector/db/lidar_tile_index.json")

_api_call_count = 0


# ── Tile discovery ────────────────────────────────────────────────────────────

def find_lidar_zips(lidar_dir: Path = LIDAR_DIR) -> List[Path]:
    if not lidar_dir.exists():
        log.warning("LiDAR directory not found", path=str(lidar_dir))
        return []
    zips = sorted(lidar_dir.glob("Sierra25_*.zip"))
    log.info("Found LiDAR zip files", count=len(zips))
    return zips


def load_tile_index() -> Dict:
    if LIDAR_INDEX_PATH.exists():
        log.info("Loading LiDAR tile index", path=str(LIDAR_INDEX_PATH))
        with open(LIDAR_INDEX_PATH) as f:
            idx = json.load(f)
        log.info("Tile index loaded", tiles=len(idx))
        return idx
    log.warning("LiDAR tile index not found", path=str(LIDAR_INDEX_PATH))
    return {}


def find_overlapping_tiles(
    bounds_wgs84: Tuple[float, float, float, float],
    tile_index: Dict,
    lidar_dir: Path = LIDAR_DIR,
) -> List[Path]:
    """Find tiles overlapping bounds using WGS84 index."""
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    overlapping = []
    for zip_name, (tx1, ty1, tx2, ty2) in tile_index.items():
        if tx2 >= min_lon and tx1 <= max_lon and ty2 >= min_lat and ty1 <= max_lat:
            zip_path = lidar_dir / zip_name
            if zip_path.exists():
                overlapping.append(zip_path)
    log.info("Tiles overlap bounds",
             count=len(overlapping),
             tiles=[p.name for p in overlapping[:5]])
    return overlapping


# ── LAS loading ───────────────────────────────────────────────────────────────

def load_las_file(zip_path: Path, max_points: int = 2_000_000) -> Optional[Dict]:
    """
    Extract and load a LiDAR tile. Returns point arrays and metadata.
    Cleans up extracted files after loading.
    """
    tmp = LIDAR_TEMP / zip_path.stem
    tmp.mkdir(parents=True, exist_ok=True)

    try:
        import laspy
        from pyproj import Transformer, CRS

        with zipfile.ZipFile(zip_path, 'r') as z:
            las_files = [f for f in z.namelist() if f.endswith('.las')]
            if not las_files:
                return None
            z.extract(las_files[0], tmp)
            las_path = tmp / las_files[0]

        with laspy.open(str(las_path)) as f:
            header = f.header

            # Detect CRS
            epsg = 26910
            try:
                crs = header.parse_crs()
                if crs:
                    epsg = crs.to_epsg() or 26910
            except Exception:
                pass

            # Read all points
            all_x, all_y, all_z = [], [], []
            all_intensity, all_class = [], []
            all_return_num, all_num_returns = [], []
            total = 0

            for chunk in f.chunk_iterator(500_000):
                all_x.append(chunk.x.copy())
                all_y.append(chunk.y.copy())
                all_z.append(chunk.z.copy())
                all_intensity.append(chunk.intensity.copy())
                all_class.append(chunk.classification.copy())
                try:
                    all_return_num.append(chunk.return_number.copy())
                    all_num_returns.append(chunk.number_of_returns.copy())
                except AttributeError:
                    all_return_num.append(np.ones(len(chunk.x), dtype=np.uint8))
                    all_num_returns.append(np.ones(len(chunk.x), dtype=np.uint8))
                total += len(chunk.x)
                if total >= max_points:
                    break

        if not all_x:
            return None

        x          = np.concatenate(all_x)
        y          = np.concatenate(all_y)
        z          = np.concatenate(all_z)
        intensity  = np.concatenate(all_intensity)
        classif    = np.concatenate(all_class)
        return_num = np.concatenate(all_return_num)
        num_returns= np.concatenate(all_num_returns)

        # Convert to WGS84 for metadata
        t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        cx = float((x.min() + x.max()) / 2)
        cy = float((y.min() + y.max()) / 2)
        center_lon, center_lat = t.transform(cx, cy)

        log.info("LAS loaded",
                 zip=zip_path.name,
                 points=len(x),
                 epsg=epsg,
                 center=f"{center_lat:.4f}N {center_lon:.4f}W",
                 z_range=f"{z.min():.1f}-{z.max():.1f}m")

        return {
            "x": x, "y": y, "z": z,
            "intensity":    intensity,
            "classification": classif,
            "return_number":  return_num,
            "num_returns":    num_returns,
            "epsg":           epsg,
            "center_lat":     center_lat,
            "center_lon":     center_lon,
            "zip_name":       zip_path.name,
            "point_count":    len(x),
        }

    except Exception as e:
        log.exception("Failed to load LAS", exc=e, zip=str(zip_path))
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Point cloud → raster ──────────────────────────────────────────────────────

def points_to_grid(x, y, values, resolution=512, method="mean"):
    if len(x) == 0:
        return np.full((resolution, resolution), np.nan, dtype=np.float32)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    if x_max == x_min or y_max == y_min:
        return np.full((resolution, resolution), np.nan, dtype=np.float32)
    col = np.clip(((x-x_min)/(x_max-x_min)*(resolution-1)).astype(int), 0, resolution-1)
    row = np.clip(((y_max-y)/(y_max-y_min)*(resolution-1)).astype(int), 0, resolution-1)
    grid   = np.zeros((resolution, resolution), dtype=np.float32)
    counts = np.zeros((resolution, resolution), dtype=np.int32)
    np.add.at(grid,   (row, col), np.where(np.isfinite(values), values, 0))
    np.add.at(counts, (row, col), np.isfinite(values).astype(int))
    if method == "mean":
        valid = counts > 0
        grid[valid]  = grid[valid] / counts[valid]
        grid[~valid] = np.nan
    elif method == "count":
        grid = counts.astype(np.float32)
        grid[grid == 0] = np.nan
    return grid


# ── Renders ───────────────────────────────────────────────────────────────────

def render_hillshade(dem, azimuth_deg=315.0, altitude_deg=45.0):
    if dem.shape[0] < 3 or dem.shape[1] < 3:
        return np.zeros(dem.shape, dtype=np.uint8)
    filled = dem.copy()
    filled[~np.isfinite(filled)] = np.nanmean(filled[np.isfinite(filled)]) if np.any(np.isfinite(filled)) else 0
    dy, dx  = np.gradient(filled)
    az_rad  = np.radians(360 - azimuth_deg + 90)
    alt_rad = np.radians(altitude_deg)
    slope   = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect  = np.arctan2(-dy, dx)
    shaded  = (np.sin(alt_rad)*np.cos(slope) +
               np.cos(alt_rad)*np.sin(slope)*np.cos(az_rad-aspect))
    return (np.clip(shaded, 0, 1) * 255).astype(np.uint8)


def render_slope_map(dem):
    if dem.shape[0] < 3 or dem.shape[1] < 3:
        return np.zeros(dem.shape, dtype=np.uint8)
    filled = dem.copy()
    filled[~np.isfinite(filled)] = np.nanmean(filled[np.isfinite(filled)]) if np.any(np.isfinite(filled)) else 0
    dy, dx     = np.gradient(filled)
    slope_deg  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope_deg  = np.clip(slope_deg, 0, 60)
    norm       = np.where(np.isfinite(dem), slope_deg / 60.0, 0)
    return (norm * 255).astype(np.uint8)


def render_local_relief(dem, window=25):
    if dem.shape[0] < window or dem.shape[1] < window:
        return np.zeros(dem.shape, dtype=np.uint8)
    filled = dem.copy()
    mean_v = float(np.nanmean(filled[np.isfinite(filled)])) if np.any(np.isfinite(filled)) else 0
    filled[~np.isfinite(filled)] = mean_v
    smoothed = uniform_filter(filled, size=window)
    lrm      = filled - smoothed
    valid    = lrm[np.isfinite(dem)]
    if len(valid) == 0:
        return np.zeros(dem.shape, dtype=np.uint8)
    p5, p95  = np.percentile(valid, 5), np.percentile(valid, 95)
    norm     = np.clip((lrm - p5) / max(p95-p5, 1e-6), 0, 1)
    norm     = np.where(np.isfinite(dem), norm, 0.5)
    return (norm * 255).astype(np.uint8)


def render_intensity_map(intensity_grid):
    valid = intensity_grid[np.isfinite(intensity_grid)]
    if len(valid) == 0:
        return np.zeros(intensity_grid.shape, dtype=np.uint8)
    p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
    norm    = np.clip((intensity_grid-p2)/max(p98-p2,1), 0, 1)
    norm    = np.where(np.isfinite(intensity_grid), norm, 0)
    return (norm * 255).astype(np.uint8)


def render_3d_perspective(dem, size=512, elev_angle_deg=35, azimuth_deg=315,
                          vertical_exaggeration=3.0):
    if dem.shape[0] < 3 or dem.shape[1] < 3:
        return np.zeros((size, size, 3), dtype=np.uint8)
    valid = np.isfinite(dem)
    if valid.sum() < 10:
        return np.zeros((size, size, 3), dtype=np.uint8)
    z_min = float(np.nanpercentile(dem[valid], 2))
    z_max = float(np.nanpercentile(dem[valid], 98))
    dem_norm = np.where(valid, (dem-z_min)/max(z_max-z_min,1), 0)
    hs = render_hillshade(dem, azimuth_deg, altitude_deg=40).astype(np.float32)/255
    r = np.where(dem_norm < 0.5,
                 0.1 + dem_norm*0.8,
                 0.9 - (dem_norm-0.5)*0.2)
    g = np.where(dem_norm < 0.5,
                 0.5 + dem_norm*0.3,
                 0.65 - (dem_norm-0.5)*0.5)
    b = np.where(dem_norm < 0.7, 0.1, 0.1+(dem_norm-0.7)/0.3*0.8)
    r = np.clip(r*(0.5+0.5*hs), 0, 1)
    g = np.clip(g*(0.5+0.5*hs), 0, 1)
    b = np.clip(b*(0.5+0.5*hs), 0, 1)
    rows, cols = dem.shape
    xs = np.linspace(-1,1,cols)
    ys = np.linspace(-1,1,rows)
    xg, yg = np.meshgrid(xs, ys)
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elev_angle_deg)
    xr =  xg*np.cos(az_rad) + yg*np.sin(az_rad)
    yr = -xg*np.sin(az_rad) + yg*np.cos(az_rad)
    z_sc   = dem_norm * vertical_exaggeration * 0.4
    sc_x   = xr * np.cos(el_rad)
    sc_y   = yr - z_sc * np.sin(el_rad)
    px = ((sc_x+1.5)/3.0*(size-1)).astype(int)
    py = ((sc_y+1.5)/3.0*(size-1)).astype(int)
    order  = np.argsort((yr - z_sc).ravel())
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    fpx = px.ravel()[order]; fpy = py.ravel()[order]
    fr  = r.ravel()[order];  fg  = g.ravel()[order]; fb = b.ravel()[order]
    fv  = valid.ravel()[order]
    ib  = (fpx>=0)&(fpx<size)&(fpy>=0)&(fpy<size)&fv
    canvas[fpy[ib], fpx[ib], 0] = fr[ib]
    canvas[fpy[ib], fpx[ib], 1] = fg[ib]
    canvas[fpy[ib], fpx[ib], 2] = fb[ib]
    return (canvas*255).astype(np.uint8)


def render_canopy_height(dem_first, dem_ground):
    """
    Canopy Height Model = first return minus ground return.
    Bright = tall trees. Dark = bare ground / low vegetation.
    Critical for distinguishing forest morphology from mining disturbance.
    """
    chm = dem_first - dem_ground
    chm = np.where(np.isfinite(chm) & (chm >= 0), chm, np.nan)
    valid = chm[np.isfinite(chm)]
    if len(valid) == 0:
        return np.zeros(dem_ground.shape, dtype=np.uint8)
    p98  = float(np.percentile(valid, 98))
    norm = np.clip(chm / max(p98, 1.0), 0, 1)
    norm = np.where(np.isfinite(chm), norm, 0)
    return (norm * 255).astype(np.uint8)


def arr_to_b64(arr, size=512, mode="L"):
    img = Image.fromarray(arr.astype(np.uint8), mode=mode)
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ── Grok call ─────────────────────────────────────────────────────────────────

LIDAR_SYSTEM_PROMPT = """You are an expert geologist and LiDAR remote sensing specialist.

You are analyzing high-resolution LiDAR terrain data (~100m tile).

IMPORTANT: Check CANOPY_HEIGHT_MODEL first. Bright = tall trees.
In forested areas, rough terrain in other images is forest floor morphology.
Only describe surface features where canopy height is LOW (dark = bare ground).

You will receive 8 images:
1. HILLSHADE_NW_315 — NW sun. Standard terrain reference.
2. HILLSHADE_NE_45  — NE sun.
3. HILLSHADE_S_180  — S sun.
4. 3D_PERSPECTIVE   — Oblique view.
5. SLOPE_MAP        — Dark=flat, bright=steep.
6. LOCAL_RELIEF_MODEL — Local relief anomalies.
7. INTENSITY_MAP    — Bright=bare rock/soil, dark=vegetation/water.
8. CANOPY_HEIGHT_MODEL — Bright=tall trees, dark=bare ground. CHECK FIRST.

Describe what you observe:
VEGETATION_COVER: [% canopy estimate]
BARE_GROUND_ZONES: [where ground is exposed]
SURFACE_MORPHOLOGY: [terrain character at bare ground zones]
GEOLOGICAL_SETTING: [landforms, lithology, processes evident]
CONFIDENCE: [0.0-1.0]
CONFIDENCE: [0.0-1.0]"""


def call_grok_lidar(images, tile_name, point_stats, center_lat, center_lon):
    global _api_call_count
    if _api_call_count >= ACTIVE_CONFIG.get("max_api_calls", 200):
        log.warning("API budget exhausted")
        return None
    _api_call_count += 1

    if not GROK_API_KEY:
        log.error("GROK_API_KEY not set")
        return None

    cfg = ACTIVE_CONFIG

    # Calculate actual tile dimensions from point bounds
    z_min = point_stats.get("z_min", 0)
    z_max = point_stats.get("z_max", 0)
    elev_mean = (z_min + z_max) / 2

    content = []
    content.append({"type": "text", "text":
        f"TILE: {tile_name}\n"
        f"LOCATION: {center_lat:.4f}°N {center_lon:.4f}°W\n"
        f"ELEVATION: {z_min:.0f}–{z_max:.0f}m "
        f"(mean {elev_mean:.0f}m)\n"
        f"POINT DENSITY: {point_stats['density_per_m2']:.1f} pts/m²\n"
        f"GROUND RETURNS: {point_stats['ground_pct']:.0%} of {point_stats['total_points']:,} points\n"
        f"\nBegin with CANOPY_HEIGHT_MODEL before assessing other images."
    })

    for img in images:
        content.append({"type": "text", "text": f"\n[{img['label'].upper()}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img['data']}", "detail": "high"}
        })

    content.append({"type": "text", "text":
        "\nAssess this LiDAR tile. Check CANOPY_HEIGHT_MODEL first.\n"
        "VEGETATION_COVER: [% estimate]\n"
        "BARE_GROUND_ZONES: [where]\n"
        "MORPHOLOGICAL_ANOMALIES: [bare ground anomalies only]\n"
        "DEPLETION_EVIDENCE: [none/possible/likely]\n"
        "GEOLOGICAL_NOTES: [terrain interpretation]\n"
        "CONFIDENCE: [0.0-1.0]"
    })

    payload = {
        "model":      cfg["vision_model"],
        "max_tokens": cfg["max_tokens"],
        "messages": [
            {"role": "system", "content": LIDAR_SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
    }

    try:
        t0 = time.time()
        r  = requests.post(
            f"{GROK_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GROK_API_KEY}",
                     "Content-Type": "application/json"},
            json=payload, timeout=120
        )
        elapsed = round(time.time()-t0, 2)
        if r.status_code != 200:
            log.error("Grok error", status=r.status_code,
                      response=r.text[:200])
            return None
        text  = r.json()["choices"][0]["message"]["content"]
        usage = r.json().get("usage", {})
        log.info("Grok response", elapsed_s=elapsed,
                 tokens=usage.get("completion_tokens"))
        return text
    except Exception as e:
        log.exception("Grok call failed", exc=e)
        return None


# ── GridCell builder for LiDAR tiles ─────────────────────────────────────────

def find_or_build_cell_for_tile(data: Dict) -> Optional[GridCell]:
    """
    Find the level 8 (100m) grid cell that contains this LiDAR tile's center.
    Creates the cell in DB if it doesn't exist.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.grid import grid
        from core.database import db

        lat = data["center_lat"]
        lon = data["center_lon"]

        # Find level 8 cell containing this point
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
        utm_x, utm_y = t.transform(lon, lat)

        grid._ensure_boundary()
        cell_size = 100   # Level 8
        min_x, min_y, _, _ = grid._utm_bounds

        col = int((utm_x - min_x) / cell_size)
        row = int((utm_y - min_y) / cell_size)

        tile_id = grid.make_tile_id(8, row, col)
        cell    = db.get_cell(tile_id)

        if cell is None:
            cell = grid.build_cell(8, row, col)
            db.upsert_cell(cell)
            log.info("Created level 8 cell for LiDAR tile",
                     tile_id=tile_id, zip=data["zip_name"])

        return cell

    except Exception as e:
        log.exception("Failed to find/build cell", exc=e)
        return None


# ── Main agent ────────────────────────────────────────────────────────────────

class LidarAgent(BaseAgent):
    """
    Processes each LiDAR tile at native ~100m resolution.
    Maps each tile to its level 8 grid cell.
    Does NOT try to clip tiles to coarser cells.
    """

    agent_name  = "lidar_agent"
    description = "LiDAR point cloud agent — native 100m tile processing with 7-image Grok analysis"

    def __init__(self, lidar_dir: Path = LIDAR_DIR):
        super().__init__()
        self._lidar_dir  = lidar_dir
        self._zip_files  = find_lidar_zips(lidar_dir)
        self._tile_index = load_tile_index()
        LIDAR_TEMP.mkdir(parents=True, exist_ok=True)

        self.log.info("LidarAgent ready",
                      zips=len(self._zip_files),
                      indexed=len(self._tile_index))

        try:
            import laspy
            self.log.info("laspy available", version=laspy.__version__)
        except ImportError:
            self.log.error("laspy not installed — pip install laspy[lazrs]")

    def process_cell(self, cell: GridCell, **kwargs) -> GridCell:
        """
        Standard BaseAgent interface — called when running via --agent lidar.
        Finds LiDAR tiles overlapping this cell and processes each one.
        """
        bounds = (cell.min_lon, cell.min_lat, cell.max_lon, cell.max_lat)
        tiles  = find_overlapping_tiles(bounds, self._tile_index, self._lidar_dir)

        if not tiles:
            self.log.info("No LiDAR tiles for cell", tile_id=cell.tile_id)
            return cell

        self.log.info("Processing cell with LiDAR tiles",
                      tile_id=cell.tile_id, n_tiles=len(tiles))

        notes = []
        for zip_path in tiles[:5]:   # Cap at 5 tiles per cell call
            result = self._process_one_zip(zip_path)
            if result:
                notes.append(result)

        if notes:
            from datetime import datetime, timezone
            cell.llm_notes.extend(notes)
            # Average confidence from all tiles
            confs = [n.get("confidence", 0.5) for n in notes]
            avg_conf = sum(confs) / len(confs)
            cell.probability_score = round(
                (cell.probability_score or 0.3) * 0.5 + avg_conf * 0.5, 4
            )
            cell.opportunity_score = cell.probability_score

        return cell

    def process_all_tiles(self, dry_run: bool = False):
        """
        Process every LiDAR zip as its own level 8 cell.
        This is the primary LiDAR workflow — not cell-based but tile-based.
        """
        self.log.info("Processing all LiDAR tiles at native resolution",
                      total=len(self._zip_files))

        from core.database import db

        success = failed = skipped = 0

        for i, zip_path in enumerate(self._zip_files):
            self.log.info("Processing tile",
                          zip=zip_path.name,
                          progress=f"{i+1}/{len(self._zip_files)}")
            try:
                result = self._process_one_zip(zip_path)
                if result:
                    success += 1
                else:
                    skipped += 1
            except Exception as e:
                self.log.exception("Tile failed", exc=e, zip=zip_path.name)
                failed += 1

        self.log.info("All tiles processed",
                      success=success, failed=failed, skipped=skipped)
        return {"success": success, "failed": failed, "skipped": skipped}

    def _process_one_zip(self, zip_path: Path) -> Optional[Dict]:
        """
        Load one LiDAR zip, render 7 images, call Grok, return note dict.
        """
        cfg  = ACTIVE_CONFIG
        size = cfg.get("image_size_px", 512)

        # Load points
        data = load_las_file(zip_path)
        if data is None:
            return None

        x, y, z          = data["x"], data["y"], data["z"]
        intensity         = data["intensity"]
        classif           = data["classification"]
        return_num        = data["return_number"]
        total_points      = data["point_count"]

        # Separate ground from vegetation
        ground_mask = classif == 2
        if ground_mask.sum() < 100:
            ground_mask = np.ones(len(x), dtype=bool)

        first_mask = return_num == 1

        gx, gy, gz = x[ground_mask], y[ground_mask], z[ground_mask]

        # Build grids
        dem_ground     = points_to_grid(gx, gy, gz, resolution=size, method="mean")

        # First return grid — top of canopy
        fx = x[first_mask]
        fy = y[first_mask]
        fz = z[first_mask]
        dem_first      = points_to_grid(fx, fy, fz, resolution=size, method="mean")

        intensity_grid = points_to_grid(x, y, intensity.astype(float),
                                        resolution=size, method="mean")

        # Tile area estimate
        tile_w = float(x.max() - x.min())
        tile_h = float(y.max() - y.min())
        tile_area = max(tile_w * tile_h, 1.0)
        density   = total_points / tile_area

        point_stats = {
            "total_points":  total_points,
            "ground_points": int(ground_mask.sum()),
            "ground_pct":    float(ground_mask.sum()) / max(total_points, 1),
            "density_per_m2": round(density, 4),
            "z_min":         float(gz.min()) if len(gz) > 0 else 0,
            "z_max":         float(gz.max()) if len(gz) > 0 else 0,
        }

        # Render 7 images
        images = []

        for az, label in [(315, "hillshade_nw_315"),
                          (45,  "hillshade_ne_45"),
                          (180, "hillshade_s_180")]:
            try:
                hs = render_hillshade(dem_ground, azimuth_deg=az)
                images.append({"label": label, "data": arr_to_b64(hs, size, "L")})
            except Exception as e:
                self.log.warning(f"Hillshade {az} failed", error=str(e))

        try:
            persp = render_3d_perspective(dem_ground, size=size)
            images.append({"label": "3d_perspective",
                           "data": arr_to_b64(persp, size, "RGB")})
        except Exception as e:
            self.log.warning("3D perspective failed", error=str(e))

        try:
            sl = render_slope_map(dem_ground)
            images.append({"label": "slope_map", "data": arr_to_b64(sl, size, "L")})
        except Exception as e:
            self.log.warning("Slope map failed", error=str(e))

        try:
            lrm = render_local_relief(dem_ground)
            images.append({"label": "local_relief_model",
                           "data": arr_to_b64(lrm, size, "L")})
        except Exception as e:
            self.log.warning("LRM failed", error=str(e))

        try:
            im = render_intensity_map(intensity_grid)
            images.append({"label": "intensity_map",
                           "data": arr_to_b64(im, size, "L")})
        except Exception as e:
            self.log.warning("Intensity map failed", error=str(e))

        # 8. Canopy height model — critical for distinguishing trees from disturbance
        try:
            chm = render_canopy_height(dem_first, dem_ground)
            images.append({"label": "canopy_height_model",
                           "data": arr_to_b64(chm, size, "L")})
        except Exception as e:
            self.log.warning("CHM failed", error=str(e))

        if not images:
            return None

        # Call Grok
        analysis = call_grok_lidar(
            images, zip_path.name, point_stats,
            data["center_lat"], data["center_lon"]
        )

        if not analysis:
            return None

        # Parse confidence and depletion
        import re
        from datetime import datetime, timezone

        conf_m = re.search(r"CONFIDENCE:\s*([0-9.]+)", analysis, re.I)
        confidence = float(conf_m.group(1)) if conf_m else 0.5
        confidence = min(max(confidence, 0.0), 1.0)

        dep_m = re.search(r"DEPLETION_EVIDENCE:\s*(none|low|medium|high)", analysis, re.I)
        dep_map = {"none": 0.0, "low": 0.2, "medium": 0.5, "high": 0.8}
        depletion = dep_map.get(dep_m.group(1).lower() if dep_m else "none", 0.0)

        note = {
            "note":        analysis,
            "confidence":  confidence,
            "depletion":   depletion,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "model":       ACTIVE_CONFIG["vision_model"],
            "agent":       "lidar_agent",
            "zip_name":    zip_path.name,
            "center_lat":  data["center_lat"],
            "center_lon":  data["center_lon"],
            "point_stats": point_stats,
        }

        # Print to console
        print(f"\n{'='*60}")
        print(f"LIDAR — {zip_path.name}")
        print(f"Location: {data['center_lat']:.4f}°N {data['center_lon']:.4f}°W")
        print(f"Points: {total_points:,} | Density: {density:.1f}/m²")
        print(f"{'='*60}")
        print(analysis)
        print(f"{'='*60}\n")

        return note
