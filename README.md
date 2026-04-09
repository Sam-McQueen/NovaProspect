# Sierra Nevada Gold Prospecting System
### Plain-English Guide — Start Here

---

## What This System Does

This system analyzes satellite imagery and geological data to build a probability
map of where gold is likely to be found in the Sierra Nevada — and where it has
likely already been mined out.

It works like a pyramid of specialists:
- **Bottom layer:** Raw satellite images, elevation data, geological maps
- **Middle layer:** Specialist agents that read that raw data and write plain summaries
- **Top layer:** A reasoning AI that reads those summaries and scores every square kilometer

You never touch the raw data directly. You just run a command, wait, then open the
output in QGIS to see a heatmap.

---

## First Time Setup (Do This Once)

### Step 1 — Open your WSL terminal

Press the Windows key, type `Ubuntu` (or `WSL`), press Enter.
A black terminal window opens. This is where you type all commands.

### Step 2 — Navigate to this project folder

When you first open WSL, you are in your home folder (`~`).
Type this to go into the project:

```bash
cd ~/sierra_prospector
```

If you get "No such file or directory", you need to copy the project there first.
See the **Moving Files** section at the bottom of this guide.

### Step 3 — Create a virtual environment (sandbox for Python packages)

A virtual environment keeps this project's packages separate from everything else
on your system. You only do this once.

```bash
python3 -m venv venv
```

You will now see a `venv/` folder appear inside the project. Don't touch it.

### Step 4 — Activate the virtual environment

You must do this EVERY TIME you open a new terminal before running any commands.

```bash
source venv/bin/activate
```

You will see `(venv)` appear at the start of your command line. That means it worked.
If you don't see `(venv)`, the packages won't be found and things will break.

### Step 5 — Install the required packages

```bash
pip install -r requirements.txt
```

This downloads and installs everything the system needs. It takes a few minutes.
You only do this once (or after pulling new code).

### Step 6 — Put your data files in the right place

Your data files go in specific folders. Here's where everything belongs:

```
sierra_prospector/
└── data/
    ├── boundaries/
    │   └── sierra_nevada.geojson    ← YOUR SIERRA NEVADA BOUNDARY FILE GOES HERE
    └── raw/
        ├── landsat/                 ← LANDSAT GEOTIFF FILES GO HERE
        │   └── (any folder name)/
        │       └── scene_stack.tif
        ├── dem/                     ← ELEVATION (DEM) FILES GO HERE
        ├── usgs/                    ← USGS GEOCHEMISTRY DATA GOES HERE
        └── geochemistry/            ← OTHER GEOCHEMISTRY FILES GO HERE
```

The system will tell you clearly if a file is missing and where it expected to find it.

---

## Every Day Usage — The Normal Workflow

### Step 1 — Open WSL terminal

Press Windows key → type Ubuntu → Enter.

### Step 2 — Activate the environment (do this every time)

```bash
cd ~/sierra_prospector
source venv/bin/activate
```

You must see `(venv)` before the cursor. If you don't, type the source command again.

### Step 3 — Choose what you want to do

---

## Commands Reference

Think of these like menu options. You run them from inside the `sierra_prospector` folder
with the `(venv)` environment active.

---

### Check everything is set up correctly
```bash
python main.py --check
```
Checks that your data files are in the right place, the database is reachable,
and all agents are ready. Run this first if anything seems broken.

---

### See how much disk space this will need
```bash
python main.py --storage-estimate
```
Prints a table showing how many grid cells will be created at each zoom level
and how much disk space it will use. Run this before a big ingest to make sure
you have enough space.

---

### Run the first analysis pass (the main command)
```bash
python main.py
```
This runs the full ingest at zoom levels 0 through 5 (cells from 50km down to 1km).
It will take a while depending on how many Landsat scenes you have.

You can watch the progress in the terminal. Every cell processed is logged.

---

### Run only specific zoom levels
```bash
python main.py --min-level 3 --max-level 5
```
Useful if levels 0-2 are already done and you want to continue.

---

### Test without actually saving anything
```bash
python main.py --dry-run
```
Runs everything but does NOT write to the database. Use this to test that your
data files are being read correctly before committing to a full run.

---

### Check current progress
```bash
python main.py --status
```
Shows a table of how many cells have been processed at each zoom level,
how many are complete, excluded, or errored.

Example output:
```
LEVEL    SIZE     TOTAL     DONE     EXCL   ERR   AVG PROB   MAX OPP
    0   50.0km      40       40        8     0     0.3241    0.8821
    1   25.0km     160      160       31     0     0.3108    0.8654
    2   10.0km    1000      987       204    3     0.2891    0.8821
    3    5.0km    4000     3921       812    8     0.2744    0.8821
    4    2.0km   25000    24102      5231   41     0.2601    0.8950
    5    1.0km  100000    97341     21044  122     0.2489    0.9012
```

---

### Export results to QGIS
```bash
python main.py --export-geojson 5
```
Exports level 5 (1km cells) as a GeoJSON file you can load directly into QGIS.
The file will appear at: `outputs/geojson/level_05.geojson`

To see the heatmap in QGIS:
1. Open QGIS
2. Drag the `level_05.geojson` file onto the map
3. Right-click the layer → Properties → Symbology
4. Change from "Single Symbol" to "Graduated"
5. Set the column to `probability_score`
6. Click "Classify" → Apply
7. Areas in red/orange = high probability. Areas in blue = low probability.

---

### Zoom into a promising area (drill down)
```bash
python main.py --drill-down Z05_R000120_C000340
```
Takes a specific cell (you get the ID from QGIS or the status output) and
processes all the child cells at the next finer zoom level.

---

### Read the human summary for a specific cell
```bash
python main.py --read-cell Z05_R000120_C000340
```
Prints a plain-English summary of everything the system knows about that cell.
This is what the reasoning AI reads. If it makes sense to you as a geologist,
it's working correctly.

Example output:
```
=== Grid Cell Z05_R000120_C000340 ===
Location:  38.4821°N, -119.8234°W
Cell size: 1.0 km × 1.0 km
Status:    COMPLETE
Gold probability score:   0.724 / 1.0
Net opportunity score:    0.581 / 1.0

[Spectral / Alteration]
  Alteration type:     argillic (conf: 0.78)
  Iron oxide ratio:    2.341
  Hydroxyl ratio:      1.823
  Gossan index:        2.104
  NDVI (vegetation):   0.182
  Cloud cover:         3.2%

[Terrain]
  Elevation:    2140m mean (1980–2380m)
  Slope:        18.4° mean
  Aspect:       SW facing

[Geochemistry]
  Au anomaly:   0.681 (48.3 ppb raw)

[Mining History]
  Active claims:   0
  Historic mines:  1
  Depletion:       0.21 (placer_exhausted)
```

---

### If something goes wrong
```bash
python main.py --check
```

Then look at the log files. They are in:
```
logs/
├── system/
│   └── prospector.log       ← Overall system log
└── agents/
    ├── spectral_agent.log   ← Spectral agent specific log
    ├── terrain_agent.log    ← Terrain agent specific log
    └── ...
```

Each log entry is a single line of JSON. To read it in a human-friendly way:
```bash
cat logs/agents/spectral_agent.log | python3 -m json.tool | less
```
Press `q` to quit the `less` viewer.

To search logs for errors only:
```bash
grep '"level": "ERROR"' logs/system/prospector.log
```

To see the last 50 lines of a log:
```bash
tail -50 logs/system/prospector.log
```

---

## Understanding the Numbers

### probability_score (0 to 1)
How likely is it that this cell contains gold mineralization based on spectral
and geochemical signals. 1.0 = maximum signal. 0.0 = no signal.

| Score | Meaning |
|-------|---------|
| 0.0 – 0.1 | No significant indicators. Likely unaltered rock or vegetation. |
| 0.1 – 0.3 | Weak signal. Worth noting but not a priority. |
| 0.3 – 0.5 | Moderate signal. Warrants closer look at finer zoom levels. |
| 0.5 – 0.7 | Strong signal. Multiple indicators align. High priority. |
| 0.7 – 1.0 | Very strong signal. Drill down here first. |

### opportunity_score (0 to 1)
Same as probability_score but discounted by how depleted the area is from
historic mining. A cell with probability 0.9 but depletion 0.8 gets
a much lower opportunity score — it's probably already been worked.

### confidence (0 to 1)
How much data was available to compute the scores. A cell with only one
partial Landsat scene has low confidence even if the score is high.
Low confidence cells need more data before trusting the score.

### Status values
| Status | Meaning |
|--------|---------|
| PENDING | Cell created but not yet analyzed |
| PROCESSING | An agent is currently working on it |
| COMPLETE | All scheduled agents have run |
| EXCLUDED | Low probability or confirmed depleted. Kept for spatial reference. |
| ERROR | Something went wrong. Check the logs. |

---

## Moving Files From Windows to WSL

Your Windows files are accessible from WSL at `/mnt/c/Users/YourWindowsUsername/`.

To copy your Sierra Nevada GeoJSON into the project:
```bash
cp /mnt/c/Users/YourName/Downloads/sierra_nevada.geojson ~/sierra_prospector/data/boundaries/
```

To copy a folder of Landsat scenes:
```bash
cp -r /mnt/c/Users/YourName/Downloads/landsat_scenes/ ~/sierra_prospector/data/raw/landsat/
```

Replace `YourName` with your actual Windows username.

---

## Quick Reference Card

```
SETUP (once only):
  python3 -m venv venv
  pip install -r requirements.txt

EVERY TIME (start here):
  cd ~/sierra_prospector
  source venv/bin/activate        ← must see (venv) before cursor

COMMON COMMANDS:
  python main.py --check          ← is everything set up?
  python main.py --status         ← how far along am I?
  python main.py                  ← run the main analysis
  python main.py --dry-run        ← test without saving
  python main.py --export-geojson 5  ← export to QGIS
  python main.py --read-cell Z05_R000120_C000340  ← inspect one cell

IF BROKEN:
  cat logs/system/prospector.log | python3 -m json.tool | less
  grep '"level": "ERROR"' logs/agents/spectral_agent.log
```

---

## Adding New Data Sources Later

The system is designed to grow. When you're ready to add terrain (DEM) analysis,
geochemistry, or historic mine records, each one is a new "agent" — a self-contained
Python file that reads one data type and writes summaries to the database.

You don't need to change any existing code. Just add a new file in `agents/`
and register it in `main.py`. The README in `agents/HOW_TO_ADD_AN_AGENT.md`
walks through this step by step.
