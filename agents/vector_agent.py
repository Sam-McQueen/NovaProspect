"""
sierra_prospector/agents/vector_agent.py
=====================================
Vector agent — processes all vector/shapefile datasets and OSM data.

Handles:
    Qfaults_US_Database.shp  — USGS Quaternary fault database
    fault_areas.shp           — Fault zone polygons
    OSM PBF files             — Roads, trails, power lines, springs
    GNIS API                  — Named geographic features (streams, springs, peaks)

Purpose of each dataset:
    Faults     — structural geology context
    Roads/trails — proximity to human access (noted, NOT used to disqualify)
    Springs    — surface water / hydrology indicators
    GNIS       — named features for field navigation context

Road proximity scoring philosophy:
    Data is recorded as distances only.
    No cell is penalized or excluded based on road proximity.
    The reasoning agent decides what road proximity means for each target.

Road categories tracked:
    paved_road_m     — nearest paved road (highway, primary, secondary, tertiary)
    dirt_road_m      — nearest unpaved road (track, unclassified, service)
    trail_m          — nearest walkable trail (path, footway, bridleway)
    powerline_m      — nearest power line clearing (noted, low weight)
    spring_m         — nearest spring

OSM format: .pbf binary — requires pyosmium
Install: pip install osmium
"""

import math
import requests
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field

from config.settings import CRS_WGS84
from core.ontology import GridCell, CellStatus
from core.logger import get_logger
from agents.base_agent import BaseAgent

log = get_logger("vector_agent")

FAULT_SHP       = Path("/mnt/c/Geodata/Fault/SHP/Qfaults_US_Database.shp")
FAULT_AREAS_SHP = Path("/mnt/c/Geodata/Fault/SHP/fault_areas.shp")
OSM_CA          = Path("/mnt/c/Geodata/OSM (California & Nevada)/california-260322.osm.pbf")
OSM_NV          = Path("/mnt/c/Geodata/OSM (California & Nevada)/nevada-260322.osm.pbf")

GNIS_API = "https://edits.nationalmap.gov/apps/gaz-domestic/rest/search/features"


# ── Vector summary dataclass ──────────────────────────────────────────────────

@dataclass
class VectorSummary:
    """Output of vector_agent — one per GridCell."""

    # ── Faults ────────────────────────────────────────────────────────────────
    fault_count:              int   = 0
    fault_zone_pct:           Optional[float] = None   # % of cell in fault zone
    nearest_fault_m:          Optional[float] = None
    nearest_fault_name:       Optional[str]   = None
    nearest_fault_type:       Optional[str]   = None   # strike-slip, thrust, normal
    nearest_fault_age:        Optional[str]   = None   # Holocene, Pleistocene, etc.
    fault_density_km_per_km2: Optional[float] = None

    # ── Roads / trails (proximity only — not used for scoring) ────────────────
    paved_road_m:             Optional[float] = None   # nearest paved road
    dirt_road_m:              Optional[float] = None   # nearest dirt road/track
    trail_m:                  Optional[float] = None   # nearest walking trail
    powerline_m:              Optional[float] = None   # nearest power line clearing
    spring_m:                 Optional[float] = None   # nearest spring

    # Road counts within cell
    paved_road_count:         int = 0
    dirt_road_count:          int = 0
    trail_count:              int = 0
    spring_count:             int = 0

    # ── GNIS named features ───────────────────────────────────────────────────
    gnis_features:            Optional[str]   = None   # JSON list of named features
    gnis_stream_count:        int = 0
    gnis_spring_count:        int = 0
    nearest_named_stream:     Optional[str]   = None


# ── Spatial helpers ───────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def centroid_to_feature_distance(
    cell_lat: float, cell_lon: float,
    geom,   # shapely geometry
) -> float:
    """Approximate distance from cell centroid to nearest point on a geometry."""
    from shapely.geometry import Point
    cell_pt = Point(cell_lon, cell_lat)
    nearest = geom.centroid if geom.geom_type == 'Polygon' else geom
    # Rough degrees → metres conversion
    deg_dist = cell_pt.distance(nearest)
    return deg_dist * 111_000


# ── Fault processing ──────────────────────────────────────────────────────────

def process_faults(cells: List[GridCell]) -> Dict[str, VectorSummary]:
    """
    Load Qfault shapefile and compute per-cell fault statistics.
    Returns dict of tile_id → VectorSummary with fault fields populated.
    """
    summaries: Dict[str, VectorSummary] = {}

    try:
        import geopandas as gpd
        from shapely.geometry import box as sbox

        if not FAULT_SHP.exists():
            log.warning("Qfault shapefile not found", path=str(FAULT_SHP))
            return summaries

        log.info("Loading Qfault database", path=str(FAULT_SHP))
        faults = gpd.read_file(str(FAULT_SHP))
        if faults.crs and faults.crs.to_epsg() != 4326:
            faults = faults.to_crs("EPSG:4326")
        log.info("Qfault loaded", features=len(faults))

        # Load fault areas if available
        fault_areas = None
        if FAULT_AREAS_SHP.exists():
            fault_areas = gpd.read_file(str(FAULT_AREAS_SHP))
            if fault_areas.crs and fault_areas.crs.to_epsg() != 4326:
                fault_areas = fault_areas.to_crs("EPSG:4326")

        for cell in cells:
            cell_box = sbox(cell.min_lon, cell.min_lat,
                            cell.max_lon, cell.max_lat)

            # Find intersecting faults
            intersecting = faults[faults.geometry.intersects(cell_box)]
            s = summaries.setdefault(cell.tile_id, VectorSummary())
            s.fault_count = len(intersecting)

            if len(intersecting) > 0:
                # Find nearest fault
                min_dist = float("inf")
                nearest_row = None
                for _, row in intersecting.iterrows():
                    try:
                        d = centroid_to_feature_distance(
                            cell.centroid_lat, cell.centroid_lon,
                            row.geometry
                        )
                        if d < min_dist:
                            min_dist = d
                            nearest_row = row
                    except Exception:
                        pass

                if nearest_row is not None:
                    s.nearest_fault_m = round(min_dist, 1)
                    # Try common Qfault column names
                    for col in ("fault_name", "name", "FAULT_NAME", "NAME",
                                "flt_name", "FaultName"):
                        if col in nearest_row.index and nearest_row[col]:
                            s.nearest_fault_name = str(nearest_row[col])[:100]
                            break
                    for col in ("slip_sense", "type", "SLIP_SENSE",
                                "fault_type", "FLT_TYPE", "SlipSense"):
                        if col in nearest_row.index and nearest_row[col]:
                            s.nearest_fault_type = str(nearest_row[col])
                            break
                    for col in ("age", "AGEGROUP", "age_group",
                                "agegrp", "Age", "AgeGroup"):
                        if col in nearest_row.index and nearest_row[col]:
                            s.nearest_fault_age = str(nearest_row[col])
                            break

                # Fault density
                try:
                    total_len_deg = sum(
                        row.geometry.length
                        for _, row in intersecting.iterrows()
                        if row.geometry is not None
                    )
                    total_len_km = total_len_deg * 111
                    cell_area_km2 = (cell.cell_size_m / 1000) ** 2
                    s.fault_density_km_per_km2 = round(
                        total_len_km / max(cell_area_km2, 1), 4
                    )
                except Exception:
                    pass

            # Fault zone coverage
            if fault_areas is not None:
                try:
                    fa_intersecting = fault_areas[
                        fault_areas.geometry.intersects(cell_box)
                    ]
                    if len(fa_intersecting) > 0:
                        total_area = sum(
                            row.geometry.intersection(cell_box).area
                            for _, row in fa_intersecting.iterrows()
                        )
                        cell_area = cell_box.area
                        s.fault_zone_pct = round(
                            min(total_area / max(cell_area, 1e-10), 1.0), 4
                        )
                except Exception:
                    pass

        log.info("Fault processing complete",
                 cells_with_faults=sum(1 for s in summaries.values()
                                       if s.fault_count > 0))

    except ImportError:
        log.error("geopandas not installed — fault processing skipped")
    except Exception as e:
        log.exception("Fault processing failed", exc=e)

    return summaries


# ── OSM processing ────────────────────────────────────────────────────────────

# OSM highway tags → road category
PAVED_TAGS = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
    "tertiary_link", "residential", "living_street", "road",
}
DIRT_TAGS = {
    "unclassified", "service", "track",
}
TRAIL_TAGS = {
    "path", "footway", "bridleway", "cycleway", "steps",
}
POWERLINE_TAG = "power"   # way with power=line


def process_osm(cells: List[GridCell]) -> Dict[str, VectorSummary]:
    """
    Parse OSM PBF files and compute road/trail proximity per cell.
    Uses pyosmium if available, falls back to a note if not installed.
    """
    summaries: Dict[str, VectorSummary] = {}

    try:
        import osmium
    except ImportError:
        log.warning(
            "pyosmium not installed — OSM processing skipped. "
            "Install with: pip install osmium"
        )
        return summaries

    # Sierra Nevada bounding box for pre-filter
    SN_W, SN_E = -122.5, -117.0
    SN_S, SN_N =   35.5,  42.0

    # Accumulators: tile_id → {paved: [(lat,lon)], dirt: [...], trail: [...], ...}
    cell_features: Dict[str, Dict] = {
        c.tile_id: {
            "paved": [], "dirt": [], "trail": [],
            "powerline": [], "spring": [],
        }
        for c in cells
    }

    # Build fast cell lookup grid
    cell_list = cells

    class OSMHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.processed = 0

        def _classify_way(self, tags) -> Optional[str]:
            hw = tags.get("highway")
            if hw in PAVED_TAGS:
                return "paved"
            if hw in DIRT_TAGS:
                return "dirt"
            if hw in TRAIL_TAGS:
                return "trail"
            if tags.get("power") == "line":
                return "powerline"
            return None

        def way(self, w):
            category = self._classify_way(w.tags)
            if category is None:
                return

            # Get way nodes — use first node as proxy location
            try:
                nodes = list(w.nodes)
                if not nodes:
                    return
                # Use midpoint node
                mid = nodes[len(nodes)//2]
                lat, lon = mid.lat, mid.lon

                # Pre-filter
                if not (SN_S <= lat <= SN_N and SN_W <= lon <= SN_E):
                    return

                # Find cell
                for cell in cell_list:
                    if (cell.min_lat <= lat <= cell.max_lat and
                            cell.min_lon <= lon <= cell.max_lon):
                        cell_features[cell.tile_id][category].append((lat, lon))
                        break

                self.processed += 1
                if self.processed % 100_000 == 0:
                    log.debug("OSM ways processed", count=self.processed)

            except Exception:
                pass

        def node(self, n):
            # Springs and natural water sources
            if (n.tags.get("natural") == "spring" or
                    n.tags.get("amenity") == "drinking_water"):
                lat, lon = n.location.lat, n.location.lon
                if not (SN_S <= lat <= SN_N and SN_W <= lon <= SN_E):
                    return
                for cell in cell_list:
                    if (cell.min_lat <= lat <= cell.max_lat and
                            cell.min_lon <= lon <= cell.max_lon):
                        cell_features[cell.tile_id]["spring"].append((lat, lon))
                        break

    handler = OSMHandler()

    for pbf_path in [OSM_CA, OSM_NV]:
        if pbf_path.exists():
            log.info("Processing OSM file", path=pbf_path.name)
            try:
                handler.apply_file(str(pbf_path), locations=True)
            except Exception as e:
                log.error("OSM processing failed", file=pbf_path.name, error=str(e))

    log.info("OSM processing complete", total_ways=handler.processed)

    # Convert accumulators to summaries
    for cell in cells:
        feats = cell_features.get(cell.tile_id, {})
        s = summaries.setdefault(cell.tile_id, VectorSummary())

        cx, cy = cell.centroid_lon, cell.centroid_lat

        def nearest_m(pts):
            if not pts:
                return None
            dists = [haversine_m(cy, cx, lat, lon) for lat, lon in pts]
            return round(min(dists), 1)

        s.paved_road_m    = nearest_m(feats.get("paved", []))
        s.dirt_road_m     = nearest_m(feats.get("dirt", []))
        s.trail_m         = nearest_m(feats.get("trail", []))
        s.powerline_m     = nearest_m(feats.get("powerline", []))
        s.spring_m        = nearest_m(feats.get("spring", []))

        s.paved_road_count = len(feats.get("paved", []))
        s.dirt_road_count  = len(feats.get("dirt", []))
        s.trail_count      = len(feats.get("trail", []))
        s.spring_count     = len(feats.get("spring", []))

    return summaries


# ── GNIS API ──────────────────────────────────────────────────────────────────

GNIS_FEATURE_CLASSES = [
    "Stream", "Spring", "Lake", "Reservoir",
    "Valley", "Ridge", "Summit", "Mine", "Locale",
]


def query_gnis_for_cell(cell: GridCell) -> Optional[Dict]:
    """
    Query USGS GNIS API for named features within cell bounds.
    Returns summary dict or None on failure.
    Rate-limited to avoid hammering the API.
    """
    try:
        params = {
            "bbox":          f"{cell.min_lon},{cell.min_lat},{cell.max_lon},{cell.max_lat}",
            "featureClass":  ",".join(GNIS_FEATURE_CLASSES),
            "maxResults":    100,
            "format":        "json",
        }
        r = requests.get(GNIS_API, params=params, timeout=15)
        if r.status_code != 200:
            return None

        data = r.json()
        features = data.get("features") or data.get("items") or []

        if not features:
            return None

        streams = []
        springs = []
        others  = []

        for f in features:
            props = f.get("properties", f)
            name  = props.get("feature_name") or props.get("name", "")
            ftype = props.get("feature_class") or props.get("featureClass", "")

            if not name:
                continue

            if "Stream" in ftype or "River" in ftype:
                streams.append(name)
            elif "Spring" in ftype:
                springs.append(name)
            else:
                others.append(f"{name} ({ftype})")

        return {
            "streams": streams[:10],
            "springs": springs[:10],
            "others":  others[:10],
        }

    except Exception as e:
        log.debug("GNIS query failed", tile_id=cell.tile_id, error=str(e))
        return None


def process_gnis(cells: List[GridCell]) -> Dict[str, VectorSummary]:
    """Query GNIS API for all cells. Rate-limited."""
    summaries: Dict[str, VectorSummary] = {}

    log.info("Querying GNIS API", cells=len(cells))

    for i, cell in enumerate(cells):
        result = query_gnis_for_cell(cell)
        if result:
            s = summaries.setdefault(cell.tile_id, VectorSummary())
            s.gnis_stream_count  = len(result["streams"])
            s.gnis_spring_count  = len(result["springs"])
            s.nearest_named_stream = result["streams"][0] if result["streams"] else None

            import json
            all_features = (
                [f"stream: {x}" for x in result["streams"]] +
                [f"spring: {x}" for x in result["springs"]] +
                result["others"]
            )
            s.gnis_features = json.dumps(all_features[:20])

        # Polite rate limiting — 2 requests/second max
        time.sleep(0.5)

        if (i + 1) % 10 == 0:
            log.info("GNIS progress", done=i+1, total=len(cells))

    log.info("GNIS complete",
             cells_with_features=len(summaries))
    return summaries


# ── Main agent ────────────────────────────────────────────────────────────────

class VectorAgent(BaseAgent):
    """
    Processes all vector datasets per grid cell:
    - Qfault shapefile (fault proximity, type, density)
    - OSM PBF (road/trail proximity — noted only, not scored)
    - GNIS API (named features)

    Road proximity is recorded but NOT used to penalize or exclude cells.
    The reasoning agent decides what access means for each target.
    """

    agent_name  = "vector_agent"
    description = "Faults, roads/trails, springs, named features — vector spatial queries"

    def __init__(self):
        super().__init__()
        self.log.info("VectorAgent ready",
                      fault_shp=FAULT_SHP.exists(),
                      osm_ca=OSM_CA.exists(),
                      osm_nv=OSM_NV.exists())

    def process_cell(self, cell: GridCell, **kwargs) -> GridCell:
        """Single cell interface — used when called via broker on-demand."""
        results = self.run_all_cells([cell])
        return results[0] if results else cell

    def run_all_cells(
        self,
        cells: List[GridCell],
        dry_run: bool = False,
    ) -> Dict:
        """
        Primary entry point. Processes all vector datasets for all cells.
        More efficient than cell-by-cell for shapefile operations.
        """
        from core.database import db
        import json as _json

        self.log.info("Starting vector agent", cells=len(cells))

        # Merge summaries from all sources
        all_summaries: Dict[str, VectorSummary] = {}

        def merge(source: Dict[str, VectorSummary]):
            for tile_id, s in source.items():
                if tile_id not in all_summaries:
                    all_summaries[tile_id] = VectorSummary()
                existing = all_summaries[tile_id]
                # Merge non-None fields
                for field_name, val in s.__dict__.items():
                    if val is not None and val != 0:
                        if getattr(existing, field_name, None) is None:
                            setattr(existing, field_name, val)
                        elif field_name.endswith("_count"):
                            setattr(existing, field_name,
                                    getattr(existing, field_name, 0) + val)

        # ── Run all processors ────────────────────────────────────────────────
        print("\n--- Processing faults ---")
        merge(process_faults(cells))

        print("\n--- Processing OSM roads/trails/springs ---")
        merge(process_osm(cells))

        print("\n--- Querying GNIS named features ---")
        merge(process_gnis(cells))

        # ── Write to cells ────────────────────────────────────────────────────
        updated = 0
        batch   = []

        for cell in cells:
            s = all_summaries.get(cell.tile_id)
            if s is None:
                continue

            # Store as point_data extension — reuse nearest_fault fields
            # that already exist in PointDataSummary, plus add vector note
            from core.ontology import PointDataSummary
            if cell.point_data is None:
                cell.point_data = PointDataSummary()

            pd = cell.point_data
            pd.fault_count             = s.fault_count
            pd.nearest_fault_m         = s.nearest_fault_m
            pd.nearest_fault_name      = s.nearest_fault_name
            pd.nearest_fault_type      = s.nearest_fault_type
            pd.nearest_fault_age       = s.nearest_fault_age
            pd.fault_density_km_per_km2= s.fault_density_km_per_km2

            # Store road/trail/GNIS data in llm_notes as structured JSON
            vector_note = {
                "agent": "vector_agent",
                "roads": {
                    "paved_road_m":   s.paved_road_m,
                    "dirt_road_m":    s.dirt_road_m,
                    "trail_m":        s.trail_m,
                    "powerline_m":    s.powerline_m,
                    "spring_m":       s.spring_m,
                    "note": "proximity only — not used for scoring"
                },
                "gnis": {
                    "stream_count":      s.gnis_stream_count,
                    "spring_count":      s.gnis_spring_count,
                    "nearest_stream":    s.nearest_named_stream,
                    "features":          _json.loads(s.gnis_features)
                                         if s.gnis_features else [],
                },
                "fault_zone_pct": s.fault_zone_pct,
            }

            from datetime import datetime, timezone
            cell.llm_notes.append({
                "note":      _json.dumps(vector_note),
                "confidence": 1.0,
                "timestamp":  datetime.now(timezone.utc).isoformat(),
                "model":      "vector_agent",
            })

            # Print summary for cells with faults
            if s.fault_count > 0:
                print(f"\n{cell.tile_id} | {cell.centroid_lat:.2f}N "
                      f"{cell.centroid_lon:.2f}W")
                print(f"  Faults:  {s.fault_count} — "
                      f"{s.nearest_fault_name or 'unnamed'} "
                      f"({s.nearest_fault_type or 'unknown type'})")
                if s.nearest_fault_m:
                    print(f"  Nearest: {s.nearest_fault_m:.0f}m")
                if s.paved_road_m:
                    print(f"  Paved road: {s.paved_road_m:.0f}m")
                if s.trail_m:
                    print(f"  Trail: {s.trail_m:.0f}m")
                if s.nearest_named_stream:
                    print(f"  Stream: {s.nearest_named_stream}")

            batch.append(cell)
            updated += 1

        if not dry_run and batch:
            db.upsert_cells_batch(batch)

        stats = {
            "cells_processed": len(cells),
            "cells_updated":   updated,
            "cells_with_faults": sum(1 for s in all_summaries.values()
                                     if s.fault_count > 0),
        }
        self.log.info("Vector agent complete", **stats)
        return stats
