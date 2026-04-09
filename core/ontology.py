"""
sierra_prospector/core/ontology.py
=====================================
Defines GridCell — the atomic unit of the prospecting ontology.

Every piece of information in the system eventually gets attached to a GridCell.
The reasoning LLM never sees raw raster data — it sees serialized GridCells.
Adding new data sources means adding new fields here and a new agent to populate them.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any
import json


# ── Cell status flags ─────────────────────────────────────────────────────────
class CellStatus:
    PENDING    = "PENDING"     # Created in grid, not yet analyzed
    PROCESSING = "PROCESSING"  # Agent currently working on it
    COMPLETE   = "COMPLETE"    # All scheduled agents have run
    EXCLUDED   = "EXCLUDED"    # Low probability AND/OR confirmed depleted
                               # Kept in DB — excluded cells still inform spatial inference
    ERROR      = "ERROR"       # Agent failed — see error_log field


@dataclass
class SpectralSummary:
    """Output of the spectral agent — one per GridCell."""
    iron_oxide_ratio:     Optional[float] = None   # B4/B2
    hydroxyl_ratio:       Optional[float] = None   # B6/B7
    ferric_iron_ratio:    Optional[float] = None   # B6/B5
    gossan_ratio:         Optional[float] = None   # (B4+B6)/(B3+B5)
    ndvi:                 Optional[float] = None   # (B5-B4)/(B5+B4)
    clay_alteration:      Optional[float] = None   # B6/B5

    # Per-band statistics (mean, std, valid_pixel_pct)
    band_stats:           Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Derived alteration classification
    alteration_type:      Optional[str]  = None    # argillic | propylitic | silicic | unaltered
    alteration_confidence: Optional[float] = None  # 0-1

    # Data quality
    cloud_cover_pct:      Optional[float] = None
    valid_pixel_pct:      Optional[float] = None
    scene_date:           Optional[str]   = None
    landsat_scene_id:     Optional[str]   = None


@dataclass
class TerrainSummary:
    """Output of the terrain agent — one per GridCell."""
    mean_elevation_m:     Optional[float] = None
    max_elevation_m:      Optional[float] = None
    min_elevation_m:      Optional[float] = None
    mean_slope_deg:       Optional[float] = None
    dominant_aspect:      Optional[str]   = None   # N/NE/E/SE/S/SW/W/NW
    drainage_density:     Optional[float] = None   # stream length / area
    nearest_ridge_m:      Optional[float] = None
    nearest_drainage_m:   Optional[float] = None
    topographic_wetness:  Optional[float] = None   # TWI
    terrain_roughness:    Optional[float] = None   # Std dev of elevation within cell


@dataclass
class PointDataSummary:
    """Output of the point_data_agent — coordinate-based datasets."""
    # Gravity
    gravity_bouguer_mgal:     Optional[float] = None   # Bouguer anomaly mGal
    gravity_sample_count:     int   = 0
    gravity_nearest_m:        Optional[float] = None

    # Magnetics
    magnetic_intensity_nt:    Optional[float] = None   # nanoTesla
    magnetic_sample_count:    int   = 0
    magnetic_gradient:        Optional[float] = None   # Rate of change

    # Faults (Qfault)
    fault_count:              int   = 0
    nearest_fault_m:          Optional[float] = None
    nearest_fault_name:       Optional[str]   = None
    nearest_fault_type:       Optional[str]   = None   # strike-slip, thrust, normal
    nearest_fault_age:        Optional[str]   = None   # Holocene, Pleistocene, etc.
    fault_density_km_per_km2: Optional[float] = None

    # Hydrology
    stream_count:             int   = 0
    nearest_stream_m:         Optional[float] = None
    stream_order_max:         Optional[int]   = None   # Strahler order

    # Historic mines / prospects (coordinate points)
    historic_mine_count:      int   = 0
    nearest_mine_m:           Optional[float] = None
    nearest_mine_name:        Optional[str]   = None
    nearest_mine_commodity:   Optional[str]   = None   # gold, silver, copper

    # Boreholes
    borehole_count:           int   = 0
    nearest_borehole_m:       Optional[float] = None
    max_borehole_depth_m:     Optional[float] = None

    # Depletion signal from point data
    depletion_score:          Optional[float] = None   # 0-1


@dataclass
class HyperspectralSummary:
    """Output of the hyperspectral_agent — EMIT L2B mineral identification."""
    # Top minerals found in cell (mineral name → mean abundance 0-1)
    dominant_mineral_1:    Optional[str]   = None
    dominant_mineral_1_abundance: Optional[float] = None
    dominant_mineral_2:    Optional[str]   = None
    dominant_mineral_2_abundance: Optional[float] = None

    # Gold pathfinder minerals present
    goethite_score:        Optional[float] = None   # Iron oxide — gossan indicator
    jarosite_score:        Optional[float] = None   # Iron sulfate — oxidised sulphides
    kaolinite_score:       Optional[float] = None   # Clay — argillic alteration
    alunite_score:         Optional[float] = None   # Advanced argillic — high-sulf epithermal
    calcite_score:         Optional[float] = None   # Carbonate
    chlorite_score:        Optional[float] = None   # Propylitic alteration

    # Coverage
    valid_pixel_pct:       Optional[float] = None
    files_used:            int = 0
    alteration_class:      Optional[str]   = None   # argillic | propylitic | gossan | unaltered
    grok_note:             Optional[str]   = None
    model:                 Optional[str]   = None



    """Output of the geochemistry_agent — stream sediment and soil samples."""
    au_anomaly_score:         Optional[float] = None   # 0-1 normalised
    au_ppb_max:               Optional[float] = None
    au_ppb_mean:              Optional[float] = None
    nearest_au_sample_m:      Optional[float] = None
    au_sample_count:          int   = 0
    ag_anomaly_score:         Optional[float] = None
    as_anomaly_score:         Optional[float] = None   # Arsenic pathfinder
    sb_anomaly_score:         Optional[float] = None   # Antimony pathfinder
    hg_anomaly_score:         Optional[float] = None   # Mercury pathfinder


@dataclass
class StructuralSummary:
    """Output of structural_agent — synthesizes Round 1 data for structural interpretation."""
    structural_setting:       Optional[str]   = None
    deformation_style:        Optional[str]   = None
    structural_score:         Optional[float] = None   # 0-1
    key_observations:         Optional[str]   = None
    notable_features:         Optional[str]   = None   # features worth drilling down on
    investigation_rounds:     int             = 0      # how many rounds the investigation took
    model:                    Optional[str]   = None


@dataclass
class HistorySummary:
    """Output of history_agent — runs last, reads all prior data."""
    active_claims:            int   = 0
    historic_claims:          int   = 0
    recorded_mines:           int   = 0
    recorded_placer:          int   = 0
    hydraulic_mining_pct:     Optional[float] = None
    depletion_score:          Optional[float] = None   # Final depletion estimate
    depletion_reason:         Optional[str]   = None
    historical_notes:         Optional[str]   = None   # LLM summary of 1800s docs
    model:                    Optional[str]   = None


@dataclass
class GeologyNote:
    """Free-text note written back by the reasoning LLM."""
    note:         str  = ""
    confidence:   float = 0.0
    timestamp:    str   = field(default_factory=lambda: datetime.utcnow().isoformat())
    model:        str   = ""     # Which LLM wrote this


@dataclass
class GridCell:
    """
    The atomic ontology object — one geographic tile at one resolution level.

    tile_id format:  "Z{level}_R{row}_C{col}"
    Example:         "Z5_R120_C340"

    Parent/child relationships allow the reasoning agent to drill up or down the pyramid.
    All *Summary fields start None and get populated as agents run.
    The reasoning LLM reads serialized GridCells — it never reads raw rasters.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    tile_id:          str   = ""
    level:            int   = 0       # Resolution level (0 = coarsest)
    row:              int   = 0
    col:              int   = 0
    parent_tile_id:   Optional[str]  = None
    cell_size_m:      float = 0.0     # Edge length in metres

    # ── Spatial bounds (WGS84) ────────────────────────────────────────────────
    min_lon:          float = 0.0
    min_lat:          float = 0.0
    max_lon:          float = 0.0
    max_lat:          float = 0.0
    centroid_lon:     float = 0.0
    centroid_lat:     float = 0.0

    # ── Status & scoring ──────────────────────────────────────────────────────
    status:           str   = CellStatus.PENDING
    probability_score: Optional[float] = None    # 0-1 composite gold probability
    opportunity_score: Optional[float] = None    # probability × (1 - depletion_score)
    confidence:        Optional[float] = None    # Data coverage confidence 0-1

    # ── Agent outputs (populated progressively) ───────────────────────────────
    spectral:         Optional[SpectralSummary]       = None
    terrain:          Optional[TerrainSummary]         = None
    hyperspectral:    Optional[HyperspectralSummary]   = None
    geochemistry:     Optional[GeochemistrySummary]    = None
    point_data:       Optional[PointDataSummary]       = None
    structural:       Optional[StructuralSummary]      = None
    history:          Optional[HistorySummary]         = None

    # ── Reasoning LLM write-back ──────────────────────────────────────────────
    llm_notes:        list = field(default_factory=list)   # List[GeologyNote]
    llm_probability:  Optional[float] = None    # LLM's own probability estimate
    llm_reasoning:    Optional[str]   = None    # Reasoning chain summary

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at:       str  = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at:       str  = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ── Error tracking ────────────────────────────────────────────────────────
    error_log:        list = field(default_factory=list)   # List[str] of error messages

    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a flat dict for DB storage. Nested objects become JSON strings."""
        d = asdict(self)
        # Flatten nested dataclasses to JSON strings for columnar storage
        for key in ("spectral", "terrain", "hyperspectral", "geochemistry", "point_data", "structural", "history"):
            if d[key] is not None:
                d[key] = json.dumps(d[key])
        d["llm_notes"]  = json.dumps(d["llm_notes"])
        d["error_log"]  = json.dumps(d["error_log"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GridCell":
        """Deserialize from DB row dict."""
        import copy
        d = copy.deepcopy(d)
        for key, klass in [
            ("spectral",       SpectralSummary),
            ("terrain",        TerrainSummary),
            ("hyperspectral",  HyperspectralSummary),
            ("geochemistry",   GeochemistrySummary),
            ("point_data",     PointDataSummary),
            ("structural",     StructuralSummary),
            ("history",        HistorySummary),
        ]:
            if d.get(key) and isinstance(d[key], str):
                raw = json.loads(d[key])
                d[key] = klass(**raw) if raw else None
        if isinstance(d.get("llm_notes"), str):
            d["llm_notes"] = json.loads(d["llm_notes"])
        if isinstance(d.get("error_log"), str):
            d["error_log"] = json.loads(d["error_log"])
        return cls(**d)

    def to_llm_prompt(self) -> str:
        """
        Render this cell as a structured natural-language summary
        suitable for injection into an LLM context window.
        The LLM never sees raw numbers without units or context.
        """
        lines = [
            f"=== Grid Cell {self.tile_id} ===",
            f"Location:  {self.centroid_lat:.4f}°N, {self.centroid_lon:.4f}°W",
            f"Cell size: {self.cell_size_m/1000:.1f} km × {self.cell_size_m/1000:.1f} km",
            f"Status:    {self.status}",
        ]

        if self.probability_score is not None:
            lines.append(f"Gold probability score:   {self.probability_score:.3f} / 1.0")
        if self.opportunity_score is not None:
            lines.append(f"Net opportunity score:    {self.opportunity_score:.3f} / 1.0")

        if self.spectral:
            s = self.spectral
            lines.append("\n[Spectral / Alteration]")
            if s.alteration_type:
                lines.append(f"  Alteration type:     {s.alteration_type} (conf: {s.alteration_confidence:.2f})")
            if s.iron_oxide_ratio is not None:
                lines.append(f"  Iron oxide ratio:    {s.iron_oxide_ratio:.3f}")
            if s.hydroxyl_ratio is not None:
                lines.append(f"  Hydroxyl ratio:      {s.hydroxyl_ratio:.3f}")
            if s.gossan_ratio is not None:
                lines.append(f"  Gossan index:        {s.gossan_ratio:.3f}")
            if s.ndvi is not None:
                lines.append(f"  NDVI (vegetation):   {s.ndvi:.3f}")
            if s.cloud_cover_pct is not None:
                lines.append(f"  Cloud cover:         {s.cloud_cover_pct:.1f}%")

        if self.terrain:
            t = self.terrain
            lines.append("\n[Terrain]")
            if t.mean_elevation_m is not None:
                lines.append(f"  Elevation:           {t.mean_elevation_m:.0f}m mean ({t.min_elevation_m:.0f}–{t.max_elevation_m:.0f}m)")
            if t.mean_slope_deg is not None:
                lines.append(f"  Slope:               {t.mean_slope_deg:.1f}° mean")
            if t.dominant_aspect:
                lines.append(f"  Dominant aspect:     {t.dominant_aspect}")
            if t.nearest_drainage_m is not None:
                lines.append(f"  Nearest drainage:    {t.nearest_drainage_m:.0f}m")

        if self.geochemistry:
            g = self.geochemistry
            lines.append("\n[Geochemistry]")
            if g.au_anomaly_score is not None:
                lines.append(f"  Au anomaly score:    {g.au_anomaly_score:.3f} (raw: {g.au_ppb_max:.1f} ppb)")
            if g.nearest_au_sample_m is not None:
                lines.append(f"  Nearest Au sample:   {g.nearest_au_sample_m:.0f}m away")

        if self.history:
            h = self.history
            lines.append("\n[Mining History]")
            lines.append(f"  Active claims:       {h.active_claims}")
            lines.append(f"  Historic mines:      {h.recorded_mines}")
            if h.depletion_score is not None:
                lines.append(f"  Depletion score:     {h.depletion_score:.3f} ({h.depletion_reason or 'unknown reason'})")

        if self.llm_notes:
            lines.append("\n[Previous LLM Notes]")
            for note in self.llm_notes[-3:]:   # Show last 3 only to save context
                if isinstance(note, dict):
                    lines.append(f"  [{note.get('timestamp','')}] {note.get('note','')}")

        if self.status == CellStatus.EXCLUDED:
            lines.append("\n** EXCLUDED — retained for spatial inference but not a drill target **")

        return "\n".join(lines)

@dataclass
class GeochemistrySummary:
    """Output of the geochemistry_agent."""
    au_anomaly_score:    Optional[float] = None
    au_ppb_max:          Optional[float] = None
    au_ppb_mean:         Optional[float] = None
    nearest_au_sample_m: Optional[float] = None
    au_sample_count:     int   = 0
    ag_anomaly_score:    Optional[float] = None
    as_anomaly_score:    Optional[float] = None
    sb_anomaly_score:    Optional[float] = None
    hg_anomaly_score:    Optional[float] = None
