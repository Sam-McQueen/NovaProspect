"""
sierra_prospector/agents/hyperspectral_agent.py
=====================================
Hyperspectral agent — processes NASA EMIT L2B mineral identification data.

EMIT L2B contains pre-identified minerals per pixel at ~60m resolution.
This is the highest-value remote sensing dataset for gold prospecting because
it directly names minerals rather than requiring spectral ratio approximations.

Each NetCDF file contains:
    group_1_mineral_id   — primary mineral per pixel (int16, index into EMIT library)
    group_1_band_depth   — abundance/confidence of primary mineral (float32, 0-1)
    group_2_mineral_id   — secondary mineral per pixel
    group_2_band_depth   — abundance of secondary mineral

Gold pathfinder minerals in EMIT library:
    Goethite   — iron oxide, gossanous material, oxidised sulphides
    Jarosite   — iron sulfate, strongly indicates oxidised pyrite
    Kaolinite  — clay, argillic hydrothermal alteration
    Alunite    — advanced argillic, high-sulfidation epithermal gold systems
    Chlorite   — propylitic alteration
    Calcite    — carbonate, sometimes associated with gold veins

Renders sent to Grok (3 images):
    1. Group 1 false color — dominant mineral map
    2. Group 2 false color — secondary mineral map
    3. Pathfinder composite — highlights gold-indicative minerals only

Files: ~/sierra_prospector/data/hyperspectral/EMIT_L2B_MIN_001_*.nc
"""

import time
import base64
import io
import requests
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import netCDF4 as nc
from PIL import Image

from config.settings import GROK_API_KEY, GROK_API_BASE, ACTIVE_CONFIG
from core.ontology import GridCell, CellStatus, HyperspectralSummary
from core.logger import get_logger
from agents.base_agent import BaseAgent

log = get_logger("hyperspectral_agent")

HYPERSPECTRAL_DIR = Path("/home/placer/sierra_prospector/data/hyperspectral")

_api_call_count = 0

# ── EMIT mineral lookup ───────────────────────────────────────────────────────
# Mineral IDs are indexes into USGS Spectral Library 06.
# The authoritative ID→name mapping is in mineral_grouping_matrix_20230503.csv
# from the emit-sds/emit-sds-l2b GitHub repository.

EMIT_MINERAL_LOOKUP_CSV = Path("/mnt/c/Geodata/Textual/emit_mineral_lookup.csv")

_GLOBAL_MINERAL_NAMES: Optional[Dict[int, str]] = None


def load_mineral_lookup() -> Dict[int, str]:
    """
    Load the EMIT mineral ID → name lookup from the official NASA CSV.
    Cached globally — loaded once per session.
    Returns dict of {index: simplified_name}.
    """
    global _GLOBAL_MINERAL_NAMES
    if _GLOBAL_MINERAL_NAMES is not None:
        return _GLOBAL_MINERAL_NAMES

    names = {}
    if not EMIT_MINERAL_LOOKUP_CSV.exists():
        log.warning("EMIT mineral lookup CSV not found",
                    path=str(EMIT_MINERAL_LOOKUP_CSV),
                    hint="Download from: https://raw.githubusercontent.com/"
                         "emit-sds/emit-sds-l2b/develop/data/"
                         "mineral_grouping_matrix_20230503.csv")
        _GLOBAL_MINERAL_NAMES = names
        return names

    try:
        import csv as _csv
        with open(EMIT_MINERAL_LOOKUP_CSV, encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                idx  = int(row["Index"])
                name = row["Name"].strip()
                # Simplify name: take first word, lowercase, strip special chars
                # e.g. "Goethite WS222 <250um..." → "goethite"
                simple = name.split()[0].lower().rstrip('.,;')
                # Handle compound names like "Illite+Muscovite"
                simple = simple.replace("+", "_")
                names[idx] = simple

        log.info("EMIT mineral lookup loaded",
                 entries=len(names),
                 sample={k: v for k, v in list(names.items())[:5]})

    except Exception as e:
        log.exception("Failed to load mineral lookup", exc=e)

    _GLOBAL_MINERAL_NAMES = names
    return names


def get_mineral_name(mineral_id: int, name_lookup: dict = None) -> str:
    """Get mineral name from global CSV lookup."""
    lookup = load_mineral_lookup()
    return lookup.get(int(mineral_id), f"mineral_{mineral_id}")


# EMIT-10 core validated minerals
EMIT_CORE_MINERALS = {
    "calcite", "chlorite", "dolomite", "goethite", "gypsum",
    "hematite", "illite", "muscovite", "illite_muscovite",
    "kaolinite", "montmorillonite", "vermiculite",
}

# Alteration classification sets
IRON_OXIDE_MINERALS  = {"goethite", "hematite", "ferrihydrite", "lepidocrocite"}
CLAY_MINERALS        = {"kaolinite", "illite", "illite_muscovite", "muscovite",
                        "montmorillonite", "smectite", "alunite", "pyrophyllite"}
PROPYLITIC_MINERALS  = {"chlorite", "epidote", "calcite", "dolomite"}

# False color RGB per mineral for rendering
MINERAL_COLORS = {
    "goethite":         (210, 120,  30),
    "hematite":         (180,  40,  40),
    "kaolinite":        (200, 200, 120),
    "illite":           (180, 180, 100),
    "illite_muscovite": (185, 185, 110),
    "muscovite":        (190, 190, 140),
    "montmorillonite":  (160, 180, 100),
    "chlorite":         ( 80, 160,  80),
    "calcite":          (200, 200, 255),
    "dolomite":         (180, 180, 240),
    "gypsum":           (230, 230, 250),
    "vermiculite":      (140, 160,  80),
    "alunite":          (255, 160,   0),
    "jarosite":         (240, 200,  20),
}


def get_mineral_color(mineral_name: str) -> tuple:
    return MINERAL_COLORS.get(mineral_name, (100, 100, 100))



# ── File discovery and loading ────────────────────────────────────────────────

def find_emit_files(data_dir: Path = HYPERSPECTRAL_DIR) -> List[Path]:
    if not data_dir.exists():
        log.warning("Hyperspectral directory not found", path=str(data_dir))
        return []
    files = sorted(data_dir.glob("EMIT_L2B_MIN_001_*.nc"))
    log.info("Found EMIT files", count=len(files))
    return files


def get_file_bounds(nc_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """Return (min_lon, min_lat, max_lon, max_lat) for an EMIT file."""
    try:
        ds = nc.Dataset(str(nc_path))
        w = float(ds.westernmost_longitude)
        e = float(ds.easternmost_longitude)
        s = float(ds.southernmost_latitude)
        n = float(ds.northernmost_latitude)
        ds.close()
        return w, s, e, n
    except Exception as ex:
        log.warning("Could not read bounds", file=nc_path.name, error=str(ex))
        return None


def find_overlapping_files(
    bounds_wgs84: Tuple[float, float, float, float],
    all_files: List[Path],
) -> List[Path]:
    """Find EMIT files overlapping the given cell bounds."""
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    overlapping = []
    for f in all_files:
        b = get_file_bounds(f)
        if b is None:
            continue
        fw, fs, fe, fn = b
        if fe >= min_lon and fw <= max_lon and fn >= min_lat and fs <= max_lat:
            overlapping.append(f)
    return overlapping


def load_emit_for_bounds(
    nc_path: Path,
    bounds_wgs84: Tuple[float, float, float, float],
) -> Optional[Dict]:
    """
    Load EMIT mineral data clipped to cell bounds.
    Uses geotransform to convert pixel coordinates.
    Returns dict with mineral arrays and metadata.
    """
    try:
        ds = nc.Dataset(str(nc_path))

        gt = ds.geotransform   # [west, xres, 0, north, 0, -yres]
        west  = float(gt[0])
        xres  = float(gt[1])
        north = float(gt[3])
        yres  = abs(float(gt[5]))

        min_lon, min_lat, max_lon, max_lat = bounds_wgs84

        # Pixel indices for cell bounds
        col_min = max(0, int((min_lon - west) / xres))
        col_max = min(ds.variables['group_1_mineral_id'].shape[1],
                      int((max_lon - west) / xres) + 1)
        row_min = max(0, int((north - max_lat) / yres))
        row_max = min(ds.variables['group_1_mineral_id'].shape[0],
                      int((north - min_lat) / yres) + 1)

        if col_max <= col_min or row_max <= row_min:
            ds.close()
            return None

        g1_id = ds.variables['group_1_mineral_id'][row_min:row_max, col_min:col_max]
        g1_bd = ds.variables['group_1_band_depth'][row_min:row_max, col_min:col_max]
        g2_id = ds.variables['group_2_mineral_id'][row_min:row_max, col_min:col_max]
        g2_bd = ds.variables['group_2_band_depth'][row_min:row_max, col_min:col_max]
        ds.close()

        if g1_id.size == 0:
            return None

        valid = np.sum(g1_id > 0)
        if valid < 10:
            return None

        log.debug("EMIT data loaded",
                  file=nc_path.name,
                  shape=g1_id.shape,
                  valid_pixels=int(valid))

        return {
            "g1_id": np.array(g1_id),
            "g1_bd": np.array(g1_bd, dtype=np.float32),
            "g2_id": np.array(g2_id),
            "g2_bd": np.array(g2_bd, dtype=np.float32),
            "file":  nc_path.name,
        }

    except Exception as ex:
        log.exception("Failed to load EMIT file",
                      exc=ex, file=nc_path.name)
        return None


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_minerals(data_list: List[Dict]) -> HyperspectralSummary:
    """
    Compute mineral statistics across all overlapping EMIT files.
    Returns a HyperspectralSummary with pathfinder scores.
    """
    if not data_list:
        return HyperspectralSummary()

    mineral_totals: Dict[str, float] = {}
    mineral_counts: Dict[str, int]   = {}
    total_valid = 0

    for data in data_list:
        for ids, bds in [(data["g1_id"], data["g1_bd"]),
                         (data["g2_id"], data["g2_bd"])]:
            flat_ids = ids.ravel()
            flat_bds = bds.ravel()
            valid    = flat_ids > 0
            total_valid += int(valid.sum())

            for mid, bd in zip(flat_ids[valid], flat_bds[valid]):
                name = get_mineral_name(int(mid))
                if name.startswith("mineral_"):
                    continue   # Skip unmapped IDs
                mineral_totals[name] = mineral_totals.get(name, 0.0) + float(bd)
                mineral_counts[name] = mineral_counts.get(name, 0) + 1

    if not mineral_totals:
        return HyperspectralSummary(files_used=len(data_list))

    # Sort by total abundance
    ranked = sorted(mineral_totals.items(), key=lambda x: -x[1])

    summary = HyperspectralSummary(
        files_used      = len(data_list),
        valid_pixel_pct = min(1.0, total_valid / max(sum(
            d["g1_id"].size for d in data_list), 1)),
    )

    if ranked:
        summary.dominant_mineral_1 = ranked[0][0]
        summary.dominant_mineral_1_abundance = round(
            mineral_totals[ranked[0][0]] / max(mineral_counts[ranked[0][0]], 1), 4
        )
    if len(ranked) > 1:
        summary.dominant_mineral_2 = ranked[1][0]
        summary.dominant_mineral_2_abundance = round(
            mineral_totals[ranked[1][0]] / max(mineral_counts[ranked[1][0]], 1), 4
        )

    # Pathfinder scores — EMIT-10 validated minerals only
    for attr, mineral in [
        ("goethite_score",  "goethite"),
        ("jarosite_score",  "jarosite"),     # not in EMIT-10 but may appear
        ("kaolinite_score", "kaolinite"),
        ("alunite_score",   "alunite"),       # not in EMIT-10
        ("calcite_score",   "calcite"),
        ("chlorite_score",  "chlorite"),
    ]:
        if mineral in mineral_totals and mineral_counts.get(mineral, 0) > 0:
            score = mineral_totals[mineral] / mineral_counts[mineral]
            setattr(summary, attr, round(float(score), 4))

    # Alteration classification
    # Use mean band depth across all pixels — not total accumulation
    # Thresholds are mean band depths (0-0.5 range)
    # Only classify as advanced_argillic if signal is strong and consistent
    total_valid_pixels = max(sum(mineral_counts.values()), 1)

    def mean_score(minerals):
        total = sum(mineral_totals.get(m, 0) for m in minerals)
        count = sum(mineral_counts.get(m, 0) for m in minerals)
        return total / max(count, 1)

    iron_mean = mean_score(["goethite","hematite","jarosite","ferrihydrite"])
    clay_mean  = mean_score(["kaolinite","alunite","pyrophyllite","illite","sericite"])
    prop_mean  = mean_score(["chlorite","epidote","calcite"])

    alunite_mean  = mineral_totals.get("alunite", 0) / max(mineral_counts.get("alunite", 1), 1)
    jarosite_mean = mineral_totals.get("jarosite", 0) / max(mineral_counts.get("jarosite", 1), 1)

    # Require both strong signal AND meaningful pixel coverage
    alunite_coverage  = mineral_counts.get("alunite", 0) / total_valid_pixels
    jarosite_coverage = mineral_counts.get("jarosite", 0) / total_valid_pixels

    if (alunite_mean > 0.25 and alunite_coverage > 0.05) or \
       (jarosite_mean > 0.25 and jarosite_coverage > 0.05):
        summary.alteration_class = "advanced_argillic"
    elif clay_mean > 0.15 and clay_mean > iron_mean and clay_mean > prop_mean:
        summary.alteration_class = "argillic"
    elif iron_mean > 0.15 and iron_mean > clay_mean and iron_mean > prop_mean:
        summary.alteration_class = "gossan"
    elif prop_mean > 0.15 and prop_mean > clay_mean:
        summary.alteration_class = "propylitic"
    else:
        summary.alteration_class = "unaltered"

    return summary


def compute_hyperspectral_probability(summary: HyperspectralSummary) -> float:
    """Score 0-1 based on pathfinder mineral presence."""
    score = 0.0
    weights = 0.0
    for attr, mineral, weight in [
        ("jarosite_score",  "jarosite",  0.9),
        ("alunite_score",   "alunite",   0.9),
        ("goethite_score",  "goethite",  0.7),
        ("kaolinite_score", "kaolinite", 0.5),
        ("chlorite_score",  "chlorite",  0.3),
    ]:
        val = getattr(summary, attr, None)
        if val is not None and val > 0:
            score   += val * weight
            weights += weight

    if weights == 0:
        return 0.0
    return round(min(score / weights, 1.0), 4)


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_mineral_map(
    mineral_ids: np.ndarray,
    band_depths:  np.ndarray,
    size: int = 512,
) -> np.ndarray:
    """Render mineral IDs as false-color RGB image."""
    h, w = mineral_ids.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    unique_ids = np.unique(mineral_ids[mineral_ids > 0])
    for mid in unique_ids:
        mask  = mineral_ids == mid
        name  = get_mineral_name(int(mid))
        color = get_mineral_color(name)
        depth = band_depths[mask]
        for c, val in enumerate(color):
            rgb[mask, c] = (val / 255.0) * np.clip(depth * 3, 0.2, 1.0)
    img = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    img = img.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_pathfinder_composite(
    data_list: List[Dict],
    size: int = 512,
) -> str:
    """
    Render a composite image highlighting only gold pathfinder minerals.
    Red channel   = iron oxides (goethite, jarosite, hematite)
    Green channel = clay minerals (kaolinite, alunite, illite)
    Blue channel  = propylitic (chlorite, epidote, calcite)
    """
    if not data_list:
        return ""

    ref_shape = data_list[0]["g1_id"].shape
    r = np.zeros(ref_shape, dtype=np.float32)
    g = np.zeros(ref_shape, dtype=np.float32)
    b = np.zeros(ref_shape, dtype=np.float32)

    for data in data_list:
        for ids, bds in [(data["g1_id"], data["g1_bd"]),
                         (data["g2_id"], data["g2_bd"])]:
            unique_ids = np.unique(ids[ids > 0])
            for mid in unique_ids:
                name = get_mineral_name(int(mid))
                mask = ids == mid
                if name in IRON_OXIDE_MINERALS:
                    r[mask] = np.maximum(r[mask], bds[mask])
                elif name in CLAY_MINERALS:
                    g[mask] = np.maximum(g[mask], bds[mask])
                elif name in PROPYLITIC_MINERALS:
                    b[mask] = np.maximum(b[mask], bds[mask])

    rgb = np.stack([
        np.clip(r * 3, 0, 1),
        np.clip(g * 3, 0, 1),
        np.clip(b * 3, 0, 1),
    ], axis=-1)

    img = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    img = img.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Grok call ─────────────────────────────────────────────────────────────────

HYPERSPECTRAL_PROMPT = """You are an expert geologist and mineralogist.

You are reviewing EMIT satellite mineral identification data for one geographic cell.
You will see 3 images:
1. GROUP_1_MINERALS — dominant mineral per pixel (false color)
2. GROUP_2_MINERALS — secondary mineral per pixel (false color)  
3. PATHFINDER_COMPOSITE — Red=iron oxides, Green=clay minerals, Blue=carbonate/propylitic minerals

In 1-2 sentences describe the dominant mineralogical assemblage and what geological setting or process it suggests."""


def call_grok_hyperspectral(
    images: List[Dict],
    summary: HyperspectralSummary,
    cell: GridCell,
) -> Optional[str]:
    global _api_call_count
    if _api_call_count >= ACTIVE_CONFIG.get("max_api_calls", 200):
        return None
    _api_call_count += 1

    if not GROK_API_KEY:
        log.error("GROK_API_KEY not set")
        return None

    cfg     = ACTIVE_CONFIG
    content = []

    content.append({"type": "text", "text":
        f"CELL: {cell.tile_id}\n"
        f"Location: {cell.centroid_lat:.4f}°N {cell.centroid_lon:.4f}°W\n"
        f"Dominant mineral: {summary.dominant_mineral_1} "
        f"(abundance {summary.dominant_mineral_1_abundance:.3f})\n"
        f"Secondary mineral: {summary.dominant_mineral_2}\n"
        f"Alteration class: {summary.alteration_class}\n"
        f"Files: {summary.files_used} EMIT granules"
    })

    for img in images:
        content.append({"type": "text", "text": f"\n[{img['label'].upper()}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img['data']}", "detail": "high"}
        })

    content.append({"type": "text", "text":
        "\nIn 1-2 sentences: dominant alteration assemblage and gold potential?"
    })

    payload = {
        "model":      cfg["vision_model"],
        "max_tokens": 200,   # Short answer — 1-2 sentences only
        "messages": [
            {"role": "system", "content": HYPERSPECTRAL_PROMPT},
            {"role": "user",   "content": content},
        ],
    }

    try:
        t0 = time.time()
        r  = requests.post(
            f"{GROK_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GROK_API_KEY}",
                     "Content-Type": "application/json"},
            json=payload, timeout=60
        )
        elapsed = round(time.time()-t0, 2)
        if r.status_code != 200:
            log.error("Grok error", status=r.status_code, response=r.text[:200])
            return None
        text = r.json()["choices"][0]["message"]["content"]
        log.info("Grok hyperspectral response",
                 tile_id=cell.tile_id, elapsed_s=elapsed)
        return text
    except Exception as ex:
        log.exception("Grok call failed", exc=ex)
        return None


# ── Main agent ────────────────────────────────────────────────────────────────

class HyperspectralAgent(BaseAgent):
    """
    Processes NASA EMIT L2B mineral identification data per grid cell.
    Finds overlapping EMIT files, clips to cell bounds, extracts mineral stats,
    renders 3 false-color images, sends to Grok for 1-2 sentence assessment.
    """

    agent_name  = "hyperspectral_agent"
    description = "NASA EMIT L2B mineral identification — direct mineral mapping per cell"

    def __init__(self, data_dir: Path = HYPERSPECTRAL_DIR):
        super().__init__()
        self._files = find_emit_files(data_dir)
        self.log.info("HyperspectralAgent ready",
                      files=len(self._files),
                      model=ACTIVE_CONFIG["vision_model"])

    def process_cell(self, cell: GridCell, **kwargs) -> GridCell:
        bounds = (cell.min_lon, cell.min_lat, cell.max_lon, cell.max_lat)
        cfg    = ACTIVE_CONFIG
        size   = cfg.get("image_size_px", 512)

        # Find overlapping EMIT files
        overlapping = find_overlapping_files(bounds, self._files)
        if not overlapping:
            from core.alerts import alerts
            alerts.info(cell.tile_id, self.agent_name,
                        "No EMIT coverage for this cell")
            self.log.debug("No EMIT coverage for cell", tile_id=cell.tile_id)
            return cell

        self.log.info("Processing cell",
                      tile_id=cell.tile_id,
                      emit_files=len(overlapping))

        # Load data from all overlapping files
        data_list = []
        for f in overlapping:
            data = load_emit_for_bounds(f, bounds)
            if data:
                data_list.append(data)

        if not data_list:
            self.log.warning("No valid EMIT data in bounds", tile_id=cell.tile_id)
            return cell

        # Analyze minerals
        summary = analyze_minerals(data_list)
        summary.model = cfg["vision_model"]

        # Warn if no real minerals found — likely all vegetation
        if not summary.dominant_mineral_1:
            from core.alerts import alerts
            alerts.warning(cell.tile_id, self.agent_name,
                           "No mineralogical signal detected — "
                           "all pixels may be vegetation or cloud-covered")
        else:
            from core.alerts import alerts as _a
            _a.reset_consecutive(self.agent_name)

        # Compute probability contribution
        hyper_prob = compute_hyperspectral_probability(summary)

        # Render images
        images = []
        first_data = data_list[0]

        try:
            b64 = render_mineral_map(first_data["g1_id"], first_data["g1_bd"], size)
            images.append({"label": "group_1_minerals", "data": b64})
        except Exception as ex:
            self.log.warning("Group 1 render failed", error=str(ex))

        try:
            b64 = render_mineral_map(first_data["g2_id"], first_data["g2_bd"], size)
            images.append({"label": "group_2_minerals", "data": b64})
        except Exception as ex:
            self.log.warning("Group 2 render failed", error=str(ex))

        try:
            b64 = render_pathfinder_composite(data_list, size)
            images.append({"label": "pathfinder_composite", "data": b64})
        except Exception as ex:
            self.log.warning("Pathfinder composite failed", error=str(ex))

        # Call Grok
        if images:
            note = call_grok_hyperspectral(images, summary, cell)
            if note:
                summary.grok_note = note
                print(f"\n{'='*60}")
                print(f"HYPERSPECTRAL — {cell.tile_id}")
                print(f"Dominant: {summary.dominant_mineral_1} | "
                      f"Alteration: {summary.alteration_class}")
                print(f"Grok: {note}")
                print(f"{'='*60}\n")

        cell.hyperspectral = summary

        # Blend into probability
        if cell.probability_score is not None:
            cell.probability_score = round(
                cell.probability_score * 0.6 + hyper_prob * 0.4, 4
            )
        else:
            cell.probability_score = hyper_prob
        cell.opportunity_score = cell.probability_score

        self.log.info("Cell complete",
                      tile_id=cell.tile_id,
                      dominant=summary.dominant_mineral_1,
                      alteration=summary.alteration_class,
                      hyper_prob=hyper_prob,
                      grok=bool(summary.grok_note))
        return cell
