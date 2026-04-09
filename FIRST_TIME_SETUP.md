# First Time Setup Guide
### Read this before anything else

---

## What You Need Before Starting

1. **WSL2 installed** (Ubuntu) — the Linux environment inside Windows
2. **This project folder** — extracted from the .tar.gz file
3. **Your data files** — Landsat GeoTIFFs and Sierra Nevada boundary GeoJSON

---

## Step-by-Step: Getting WSL2 Ready

### 1. Open Ubuntu (WSL)

Press the **Windows key**, type **Ubuntu**, press **Enter**.
A black terminal window opens. You'll see something like:

```
username@DESKTOP-XXXXX:~$
```

The `~` means you are in your home folder. This is your starting point every time.

---

### 2. Copy the project into WSL

Your downloaded file is somewhere on Windows (probably Downloads).
In the Ubuntu terminal, run this — replace `YourWindowsUsername` with your actual Windows username:

```bash
cp -r /mnt/c/Users/YourWindowsUsername/Downloads/sierra_prospector ~/
```

Now the project lives at `~/sierra_prospector` inside WSL.

---

### 3. Go into the project folder

```bash
cd ~/sierra_prospector
```

Your terminal prompt will change to show the folder name. You should be here
for all future commands.

---

### 4. Create a virtual environment

This is a private sandbox that keeps the project's Python packages
from interfering with anything else on your system.

```bash
python3 -m venv venv
```

A folder called `venv/` will appear. Don't touch or delete it.

---

### 5. Activate the virtual environment

```bash
source venv/bin/activate
```

You MUST see `(venv)` at the start of your prompt:

```
(venv) username@DESKTOP-XXXXX:~/sierra_prospector$
```

**You need to do this every single time you open a new terminal.**
If you don't see `(venv)`, your packages won't work.

---

### 6. Install required packages

```bash
pip install -r requirements.txt
```

This downloads everything the system needs. Takes 3-5 minutes.
You'll see a lot of text scrolling past — that's normal.

If you see red error text at the end, copy it and ask for help.

---

### 7. Put your data files in the right place

The project expects data in specific folders. Create them:

```bash
mkdir -p data/boundaries data/raw/landsat data/raw/dem data/raw/usgs
```

Then copy your files in. From the Ubuntu terminal:

**Your Sierra Nevada boundary GeoJSON:**
```bash
cp /mnt/c/Users/YourWindowsUsername/path/to/sierra_nevada.geojson data/boundaries/
```

**Your Landsat GeoTIFF files:**
```bash
cp /mnt/c/Users/YourWindowsUsername/path/to/landsat/*.tif data/raw/landsat/
```

---

### 8. Run the setup check

```bash
python main.py --check
```

This will print a list like:

```
=== Sierra Prospector — Setup Check ===

Python packages:
  [ OK ]  rasterio
  [ OK ]  numpy
  [ OK ]  shapely
  [ OK ]  pyproj
  [ OK ]  duckdb
  [ OK ]  geopandas

Data files:
  [ OK ]  Sierra Nevada boundary GeoJSON
  [ OK ]  Landsat scenes (3 found)

Directories:
  [ OK ]  data/raw/landsat
  [ OK ]  data/boundaries
  ...

Database:
  [ OK ]  DuckDB main database

  Everything looks good. Run: python main.py --dry-run
```

Fix any MISSING items before continuing.

---

### 9. Do a dry run (optional but recommended)

This runs the full analysis but doesn't save anything.
Good way to confirm your data files are being read correctly.

```bash
python main.py --dry-run --max-level 2
```

Watch the terminal output. You should see cells being processed with
no red ERROR lines. If you see errors, check the log file:

```bash
cat logs/agents/spectral_agent.log | python3 -m json.tool | grep -A5 '"level": "ERROR"'
```

---

### 10. Run the real first analysis

```bash
python main.py --max-level 5
```

This processes the Sierra Nevada at zoom levels 0 through 5
(cells from 50km down to 1km). Let it run. Check progress any time with:

```bash
python main.py --status
```

---

### 11. View results in QGIS

When the run finishes, export the results:

```bash
python main.py --export-geojson 5
```

The output file is at: `outputs/geojson/level_05.geojson`

To find this file in Windows Explorer:
- Open File Explorer
- In the address bar type: `\\wsl$\Ubuntu\home\YourWSLUsername\sierra_prospector\outputs\geojson`
- Drag `level_05.geojson` into QGIS

In QGIS, style the layer by `probability_score` (Graduated symbology)
to see the heatmap.

---

## The Two Commands You Will Use Most

```bash
# Starting a session (do this every time you open a terminal):
cd ~/sierra_prospector
source venv/bin/activate

# Check where things are at:
python main.py --status
```

---

## If Something Breaks

**"Command not found" or "No module named X"**
→ You forgot to activate the virtual environment.
→ Run: `source venv/bin/activate`

**"No such file or directory"**
→ You're in the wrong folder.
→ Run: `cd ~/sierra_prospector`

**"Permission denied"**
→ Run: `chmod +x main.py`

**Red error text during a run**
→ The cell is logged as ERROR and processing continues.
→ Check: `grep '"level": "ERROR"' logs/system/prospector.log`

**Everything seems frozen**
→ It's probably just processing. Large Landsat files take time.
→ Press Ctrl+C to stop safely. Run `--status` to see what got done.
→ Resume with `--min-level N` where N is the last completed level.
