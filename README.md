# API222
Final Project for HKS API222: Machine Learning and Big Data Analytics on National Security

## Local data files (not tracked by git)

- Place any `.xlsx` or other large/private data files in the `data/` directory at the project root.
- The `data/` directory is ignored by git via `.gitignore`, so files placed there will not be committed or pushed.

Example structure:

```
API222/
  data/
    terrorism_data.xlsx
    gtd_cleaned.csv      # cleaned GTD (for maps/EDA), not tracked by git
    gtd_model_ready.csv  # model‑ready ML dataset, not tracked by git
  visualization_data_processor.py
  world_map_visualization.py
  gtd_cleaner.py
  regression_models.py
```

Accessing the local file from code or notebooks (example path):

```python
local_path = "data/terrorism_data.xlsx"
# Load with your preferred library, e.g., pandas:
# import pandas as pd
# df = pd.read_excel(local_path)
```

## GTD cleaning and model‑ready datasets

The repository includes a modular GTD cleaner at `gtd_cleaner.py` that prepares the Global Terrorism Database for ML and can also produce a **model‑ready CSV** for regression models (LASSO, Random Forest, etc.).

- What it does (core cleaning):
  - Cleans `nkill`, `nwound` (treats -99/missing as 0; outputs integers)
  - Ensures binary integers for `success`, `suicide`
  - Cleans `latitude`, `longitude` (sets -99/invalid to NaN), adds `geo_missing` flag; optional row drop
  - Removes duplicates by `eventid`
  - Drops columns with >50% missing values (keeps key columns)
  - Optionally drops very high‑cardinality categoricals (`gname`, `city`, `provstate`, `location`)
  - One-hot encodes medium‑cardinality categoricals: `attacktype1_txt`, `weaptype1_txt`, `region_txt`, `country_txt`, `targtype1_txt`
  - Adds `severity = nkill + nwound`

- Input formats: CSV or Excel (`.xlsx`/`.xls`).

- Example usage 1: write **cleaned GTD** into `data/` (for maps/EDA):

```bash
source .venv/bin/activate
python gtd_cleaner.py \
  "/path/to/globalterrorismdb.xlsx" \
  -o data/gtd_cleaned.csv --drop-missing-geo --drop-first

- Example usage 2: starting from `gtd_cleaned.csv`, create a **model‑ready CSV** for regression models:

```bash
source .venv/bin/activate
python gtd_cleaner.py \
  data/gtd_cleaned.csv \
  -o data/gtd_model_ready.csv --drop-first
```

Notes:
- The `data/` directory is ignored by git, so the generated `data/gtd_cleaned.csv` and `data/gtd_model_ready.csv` will not be committed.
- The `data/gtd_model_ready.csv` file is intended to be re‑used across multiple modeling experiments.

## World Map Visualization
<img width="1816" height="834" alt="Screenshot 2025-11-12 at 11 46 26 PM" src="https://github.com/user-attachments/assets/034d0ba1-21ea-4a62-b71c-baa4e28e1194" />


The repository includes an interactive world map visualization that displays terrorist attack locations with bubble sizes and colors representing attack severity.

### Generating the Map

To create the interactive world map visualization:

```bash
python3 world_map_visualization.py
```

This will:
- Load the cleaned terrorism data from `data/gtd_cleaned.csv`
- Filter and prepare data with valid coordinates
- Generate an interactive HTML map file: `world_map_attacks.html`

### Viewing the Map

**Option 1: Open directly**
```bash
open world_map_attacks.html
```

**Option 2: Double-click** the `world_map_attacks.html` file in Finder

**Option 3: Drag and drop** the file into any web browser window

**Option 4: Right-click** → "Open With" → Choose your browser (Chrome, Firefox, Safari, etc.)

### Interacting with the Map

Once the map is open in your browser, you can:

- **Zoom**: Use your mouse wheel or trackpad pinch gesture to zoom in/out
- **Pan**: Click and drag to move around the map
- **Hover**: Hover over any bubble to see attack details:
  - Location (city and country)
  - Severity score
  - Year of attack
  - Additional metadata
- **Color Scale**: The color bar on the right shows the severity scale (yellow = low, red = high)
- **Bubble Size**: Larger bubbles indicate higher severity attacks

### Map Features

- **4,925 attacks** across **80 countries** visualized
- **Bubble size and color** both represent attack severity
- **Interactive tooltips** with detailed attack information
- **Natural Earth projection** for accurate geographic representation
- **Responsive design** that works in all modern web browsers

### Requirements

The visualization requires:
- Python 3.6+
- `plotly>=5.18.0` (install via `pip install -r requirements.txt`)
