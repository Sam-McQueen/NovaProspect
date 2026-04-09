# How to Add a New Data Source (Agent)

Every data source in this system is an "agent" — a self-contained Python file
that knows how to read one type of data and write summaries to the database.

The spectral agent reads Landsat imagery.
The terrain agent (when you build it) will read DEM elevation files.
The geochemistry agent will read USGS stream sediment data.
And so on.

Adding a new agent does NOT require changing any existing code.

---

## The Pattern — Copy, Rename, Fill In

### Step 1 — Copy the template

```bash
cp agents/spectral_agent.py agents/MY_NEW_AGENT.py
```

### Step 2 — Change the class name and metadata at the top

```python
class TerrainAgent(BaseAgent):

    agent_name  = "terrain_agent"            # Must be unique
    description = "Computes terrain features from DEM elevation data"
```

### Step 3 — Implement process_cell()

This is the only function you must write. It receives a GridCell with spatial
bounds already set, and must return that same cell with your data attached.

```python
def process_cell(self, cell: GridCell, **kwargs) -> GridCell:

    # cell.min_lon, cell.min_lat, cell.max_lon, cell.max_lat
    # give you the geographic bounds of this cell.

    # 1. Read your data file clipped to those bounds
    # 2. Compute whatever statistics you need
    # 3. Attach results to the right field on the cell
    # 4. Return the cell

    bounds = (cell.min_lon, cell.min_lat, cell.max_lon, cell.max_lat)

    # Example: read a DEM and compute mean slope
    slope = compute_slope_from_dem(bounds)

    # Attach to the TerrainSummary field
    cell.terrain = TerrainSummary(
        mean_slope_deg = slope,
        mean_elevation_m = ...
    )

    return cell
```

### Step 4 — Register it in main.py

Find this section in main.py:

```python
# ── Initialise agents ──────────────────────────────────────────────────────
from agents.spectral_agent import SpectralAgent
spectral = SpectralAgent()
broker.register_agent("spectral_agent", spectral)
```

Add your agent below it:

```python
from agents.terrain_agent import TerrainAgent
terrain = TerrainAgent()
broker.register_agent("terrain_agent", terrain)
```

That's it. The broker, the database, the error logging, the task queue —
all of that is inherited automatically from BaseAgent.

---

## The Fields Available on GridCell

When your agent runs, the cell arrives with these fields already set:

```
cell.tile_id          — "Z05_R000120_C000340"
cell.level            — 5
cell.cell_size_m      — 1000.0  (metres)
cell.min_lon          — -119.834
cell.min_lat          — 38.481
cell.max_lon          — -119.825
cell.max_lat          — 38.490
cell.centroid_lon     — -119.829
cell.centroid_lat     — 38.485
```

Your agent fills in one of these output fields:

```
cell.spectral         — SpectralSummary     (filled by spectral_agent)
cell.terrain          — TerrainSummary      (filled by terrain_agent)
cell.geochemistry     — GeochemistrySummary (filled by geochemistry_agent)
cell.history          — HistorySummary      (filled by history_agent)
```

The summary dataclasses are defined in `core/ontology.py`.
If you need a field that doesn't exist yet, add it there.

---

## Error Handling — You Don't Need to Write Any

BaseAgent wraps your process_cell() in a try/except automatically.
If your code throws any exception:

- The cell is marked ERROR in the database
- The full traceback is saved to cell.error_log
- The error is written to logs/agents/YOUR_AGENT.log
- Processing continues with the next cell — one bad cell doesn't stop the run

You just write the happy path. The framework handles all failure cases.

---

## Testing Your New Agent on One Cell

Before running on 100,000 cells, test on one:

```python
# test_my_agent.py  (run with: python test_my_agent.py)

import sys
sys.path.insert(0, '.')

from core.database import db
from core.grid import grid
from agents.terrain_agent import TerrainAgent

db.connect()

# Build one test cell at a known location
cell = grid.build_cell(level=5, row=120, col=340)

# Run your agent on it
agent = TerrainAgent()
result = agent.process_cell(cell)

# Print the human-readable summary
print(result.to_llm_prompt())

# Check for errors
if result.error_log:
    print("ERRORS:", result.error_log)
```

If the output makes geological sense for that location, your agent is working.
