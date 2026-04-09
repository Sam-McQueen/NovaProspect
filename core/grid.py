"""
sierra_prospector/core/grid.py
=====================================
Multi-resolution grid system.

Generates a hierarchy of grid cells over the Sierra Nevada AOI (area of interest).
Each resolution level subdivides the previous into smaller cells.
Cell IDs are deterministic: "Z{level}_R{row}_C{col}" — no UUIDs needed.

The grid is projected to UTM Zone 10N (EPSG:32610) for metric cell arithmetic,
then bounds are stored back in WGS84 for compatibility with all other data.

Key operations:
  - build_grid_cells(level)       → list of GridCell for that level
  - get_children(tile_id)        → tile_ids of the 4 child cells one level down
  - get_parent(tile_id)          → tile_id of the parent cell one level up
  - tile_id_to_bounds(tile_id)   → (min_lon, min_lat, max_lon, max_lat) WGS84
"""

import json
import math
from typing import List, Tuple, Optional, Dict, Iterator
from pathlib import Path

import numpy as np
from shapely.geometry import shape, box, mapping
from shapely.ops import transform
import pyproj
from pyproj import Transformer

from config.settings import (
    SIERRA_BOUNDARY_GEOJSON, RESOLUTION_LEVELS,
    CRS_WGS84, CRS_ANALYSIS
)
from core.ontology import GridCell, CellStatus
from core.logger import get_logger

log = get_logger("grid")


# ── Projection helpers ────────────────────────────────────────────────────────

def _get_transformers():
    """Return (wgs84_to_utm, utm_to_wgs84) transformers."""
    wgs_to_utm = Transformer.from_crs(CRS_WGS84, CRS_ANALYSIS, always_xy=True)
    utm_to_wgs = Transformer.from_crs(CRS_ANALYSIS, CRS_WGS84, always_xy=True)
    return wgs_to_utm, utm_to_wgs


# ── Sierra Nevada boundary ────────────────────────────────────────────────────

def load_sierra_boundary():
    """
    Load the Sierra Nevada GeoJSON boundary.
    Returns a shapely geometry (WGS84).
    Raises FileNotFoundError with a clear message if the file is missing.
    """
    if not SIERRA_BOUNDARY_GEOJSON.exists():
        raise FileNotFoundError(
            f"Sierra Nevada boundary GeoJSON not found at: {SIERRA_BOUNDARY_GEOJSON}\n"
            f"Place your boundary file there or update SIERRA_BOUNDARY_GEOJSON in config/settings.py"
        )

    log.info("Loading Sierra Nevada boundary", path=str(SIERRA_BOUNDARY_GEOJSON))
    with open(SIERRA_BOUNDARY_GEOJSON) as f:
        gj = json.load(f)

    # Handle FeatureCollection, Feature, or bare Geometry
    if gj.get("type") == "FeatureCollection":
        geom = shape(gj["features"][0]["geometry"])
    elif gj.get("type") == "Feature":
        geom = shape(gj["geometry"])
    else:
        geom = shape(gj)

    log.info("Boundary loaded",
             bounds=geom.bounds,
             area_km2=round(geom.area * (111**2), 0))   # rough WGS84 area
    return geom


# ── Grid cell generation ──────────────────────────────────────────────────────

class GridBuilder:
    """
    Builds and navigates the multi-resolution grid.

    The grid origin is the UTM bounding box of the Sierra Nevada boundary.
    Cell (row=0, col=0) is the BOTTOM-LEFT (SW) corner — consistent with raster conventions.
    """

    def __init__(self):
        self._boundary_wgs84 = None
        self._boundary_utm   = None
        self._utm_bounds     = None          # (min_x, min_y, max_x, max_y) in UTM metres
        self._wgs_to_utm, self._utm_to_wgs = _get_transformers()
        self._cell_cache: Dict[str, GridCell] = {}

    def _ensure_boundary(self):
        if self._boundary_wgs84 is not None:
            return
        self._boundary_wgs84 = load_sierra_boundary()
        # Project to UTM for metric calculations
        self._boundary_utm = transform(
            self._wgs_to_utm.transform,
            self._boundary_wgs84
        )
        self._utm_bounds = self._boundary_utm.bounds   # (minx, miny, maxx, maxy) metres
        log.debug("UTM bounds computed", bounds=self._utm_bounds)

    # ── Tile ID encoding / decoding ───────────────────────────────────────────

    @staticmethod
    def make_tile_id(level: int, row: int, col: int) -> str:
        return f"Z{level:02d}_R{row:06d}_C{col:06d}"

    @staticmethod
    def parse_tile_id(tile_id: str) -> Tuple[int, int, int]:
        """Returns (level, row, col)."""
        try:
            parts = tile_id.split("_")
            level = int(parts[0][1:])
            row   = int(parts[1][1:])
            col   = int(parts[2][1:])
            return level, row, col
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid tile_id format: {tile_id!r} — expected Z##_R######_C######") from e

    # ── Spatial arithmetic ────────────────────────────────────────────────────

    def _cell_utm_bounds(self, level: int, row: int, col: int) -> Tuple[float, float, float, float]:
        """Returns (min_x, min_y, max_x, max_y) in UTM metres for a given cell."""
        self._ensure_boundary()
        cell_size = RESOLUTION_LEVELS[level]
        min_x_grid, min_y_grid, _, _ = self._utm_bounds

        min_x = min_x_grid + col * cell_size
        min_y = min_y_grid + row * cell_size
        max_x = min_x + cell_size
        max_y = min_y + cell_size
        return min_x, min_y, max_x, max_y

    def _utm_to_wgs84_bounds(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> Tuple[float, float, float, float]:
        """Convert UTM bounding box to WGS84 lon/lat bounds."""
        corners_utm = [
            (min_x, min_y), (max_x, min_y),
            (max_x, max_y), (min_x, max_y),
        ]
        lons, lats = [], []
        for x, y in corners_utm:
            lon, lat = self._utm_to_wgs.transform(x, y)
            lons.append(lon)
            lats.append(lat)
        return min(lons), min(lats), max(lons), max(lats)

    # ── Cell construction ─────────────────────────────────────────────────────

    def build_cell(self, level: int, row: int, col: int) -> GridCell:
        """Construct a GridCell with spatial metadata. Does NOT write to DB."""
        tile_id = self.make_tile_id(level, row, col)

        min_x, min_y, max_x, max_y = self._cell_utm_bounds(level, row, col)
        min_lon, min_lat, max_lon, max_lat = self._utm_to_wgs84_bounds(
            min_x, min_y, max_x, max_y
        )

        # Parent tile ID (one level coarser)
        parent_tile_id = None
        if level > 0 and (level - 1) in RESOLUTION_LEVELS:
            parent_cell_size = RESOLUTION_LEVELS[level - 1]
            cell_size        = RESOLUTION_LEVELS[level]
            # How many current cells fit in one parent cell per axis
            ratio = parent_cell_size // cell_size
            parent_row = row // ratio
            parent_col = col // ratio
            parent_tile_id = self.make_tile_id(level - 1, parent_row, parent_col)

        return GridCell(
            tile_id       = tile_id,
            level         = level,
            row           = row,
            col           = col,
            parent_tile_id = parent_tile_id,
            cell_size_m   = float(RESOLUTION_LEVELS[level]),
            min_lon       = min_lon,
            min_lat       = min_lat,
            max_lon       = max_lon,
            max_lat       = max_lat,
            centroid_lon  = (min_lon + max_lon) / 2,
            centroid_lat  = (min_lat + max_lat) / 2,
            status        = CellStatus.PENDING,
        )

    # ── Grid iteration ────────────────────────────────────────────────────────

    def iter_cells_at_level(
        self,
        level: int,
        intersect_only: bool = True
    ) -> Iterator[GridCell]:
        """
        Yield GridCell objects for every tile at `level` that intersects the Sierra boundary.

        intersect_only=True (default): skip cells outside the boundary — saves ~50% compute
        intersect_only=False: include all cells in the bounding box (useful for debug)

        Yields cells in row-major order (left-to-right, bottom-to-top).
        """
        self._ensure_boundary()

        if level not in RESOLUTION_LEVELS:
            raise ValueError(f"Level {level} not in RESOLUTION_LEVELS — check config/settings.py")

        cell_size = RESOLUTION_LEVELS[level]
        min_x, min_y, max_x_bound, max_y_bound = self._utm_bounds

        n_cols = math.ceil((max_x_bound - min_x) / cell_size)
        n_rows = math.ceil((max_y_bound - min_y) / cell_size)

        log.info("Iterating grid level",
                 level=level,
                 cell_size_m=cell_size,
                 n_cols=n_cols,
                 n_rows=n_rows,
                 max_cells=n_cols * n_rows)

        total_yielded = 0
        total_skipped = 0

        for row in range(n_rows):
            for col in range(n_cols):
                cx_min = min_x + col * cell_size
                cy_min = min_y + row * cell_size
                cx_max = cx_min + cell_size
                cy_max = cy_min + cell_size

                cell_box_utm = box(cx_min, cy_min, cx_max, cy_max)

                if intersect_only and not self._boundary_utm.intersects(cell_box_utm):
                    total_skipped += 1
                    continue

                cell = self.build_cell(level, row, col)
                total_yielded += 1
                yield cell

        log.info("Grid iteration complete",
                 level=level,
                 yielded=total_yielded,
                 skipped=total_skipped)

    def cell_count_at_level(self, level: int) -> int:
        """Estimate number of cells at this level that intersect the boundary."""
        self._ensure_boundary()
        cell_size = RESOLUTION_LEVELS[level]
        min_x, min_y, max_x, max_y = self._utm_bounds
        n_cols = math.ceil((max_x - min_x) / cell_size)
        n_rows = math.ceil((max_y - min_y) / cell_size)
        # Rough intersection ratio — boundary area vs bounding box area
        bbox_area = (max_x - min_x) * (max_y - min_y)
        boundary_area = self._boundary_utm.area
        ratio = min(boundary_area / bbox_area, 1.0) if bbox_area > 0 else 1.0
        return int(n_cols * n_rows * ratio)

    # ── Navigation ────────────────────────────────────────────────────────────

    def get_children(self, tile_id: str) -> List[str]:
        """
        Return tile_ids of child cells one level finer.
        Each cell splits into N×N children where N = parent_size / child_size.
        """
        level, row, col = self.parse_tile_id(tile_id)
        next_level = level + 1

        if next_level not in RESOLUTION_LEVELS:
            log.warning("No finer level available", tile_id=tile_id, level=level)
            return []

        ratio = RESOLUTION_LEVELS[level] // RESOLUTION_LEVELS[next_level]
        children = []
        for dr in range(ratio):
            for dc in range(ratio):
                child_row = row * ratio + dr
                child_col = col * ratio + dc
                children.append(self.make_tile_id(next_level, child_row, child_col))
        return children

    def get_parent(self, tile_id: str) -> Optional[str]:
        """Return tile_id of the parent cell one level coarser, or None if at root."""
        level, row, col = self.parse_tile_id(tile_id)
        if level == 0 or (level - 1) not in RESOLUTION_LEVELS:
            return None
        ratio = RESOLUTION_LEVELS[level - 1] // RESOLUTION_LEVELS[level]
        return self.make_tile_id(level - 1, row // ratio, col // ratio)

    def get_utm_bounds_for_tile(self, tile_id: str) -> Tuple[float, float, float, float]:
        """Returns (min_x, min_y, max_x, max_y) in UTM Zone 10N metres."""
        level, row, col = self.parse_tile_id(tile_id)
        return self._cell_utm_bounds(level, row, col)

    def get_wgs84_bounds_for_tile(self, tile_id: str) -> Tuple[float, float, float, float]:
        """Returns (min_lon, min_lat, max_lon, max_lat) in WGS84."""
        self._ensure_boundary()
        min_x, min_y, max_x, max_y = self.get_utm_bounds_for_tile(tile_id)
        return self._utm_to_wgs84_bounds(min_x, min_y, max_x, max_y)

    def storage_estimate(self) -> Dict[int, Dict]:
        """
        Print a storage estimate for all levels.
        Accounts for the fact that high levels are only partially populated.
        """
        self._ensure_boundary()
        estimates = {}
        bytes_per_record = 2048   # ~2KB per cell summary including JSON fields

        cumulative_gb = 0.0
        for level, cell_size in RESOLUTION_LEVELS.items():
            n_cells = self.cell_count_at_level(level)

            # High levels only populated for high-probability sub-areas
            if level <= 5:
                fill_factor = 1.0
            elif level <= 8:
                fill_factor = 0.6
            elif level <= 10:
                fill_factor = 0.25
            else:
                fill_factor = 0.05   # On-demand only

            populated = int(n_cells * fill_factor)
            raw_gb    = (populated * bytes_per_record) / 1e9
            # DuckDB achieves ~5x compression on this kind of data
            compressed_gb = raw_gb / 5

            cumulative_gb += compressed_gb
            estimates[level] = {
                "cell_size_m":    cell_size,
                "est_cells":      n_cells,
                "fill_factor":    fill_factor,
                "populated":      populated,
                "raw_gb":         round(raw_gb, 3),
                "compressed_gb":  round(compressed_gb, 3),
                "cumulative_gb":  round(cumulative_gb, 3),
            }

        return estimates


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this rather than instantiating GridBuilder everywhere.
grid = GridBuilder()
