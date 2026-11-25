# SBM-STWR Full Pipeline Analysis

A Streamlit-based academic research tool for **energy efficiency evaluation** and **spatiotemporal driving factor evolution analysis**, integrating SBM efficiency measurement, STWR spatiotemporal weighted regression, and AI-assisted Q&A in a single web application.

## Workflow

```
┌─────────────────────────┐    ┌──────────────────────────────────┐    ┌─────────────────────────────┐
│   Step 1: SBM           │    │   Step 2: STWR                   │    │   Step 3: Path Evolution    │
│   Efficiency Evaluation  │ → │   Spatiotemporal Regression      │ → │   Analysis                   │
│                          │    │                                  │    │                             │
│   Inputs / Desirable     │    │   Year-by-year independent fit   │    │   Dominant factor per year   │
│   Outputs / Undesirable  │    │   OLS → GWR → STWR              │    │   Transition matrix          │
│   Outputs → Efficiency   │    │   Auto bandwidth search         │    │   Sankey diagram + map       │
└─────────────────────────┘    └──────────────────────────────────┘    └─────────────────────────────┘
```

## Project Structure

| File | Description |
|---|---|
| `app.py` | Streamlit main application (UI, state management, AI chat, visualizations) |
| `core.py` | Core algorithm library (SBM solver, STWR modeling, path analysis, Sankey data prep) |
| `All_SBM_Data_Merged.xlsx` | Example SBM input data (province × year × inputs/outputs) |
| `STWR_Merged_Input1.csv` | Example STWR driver data (province × year × lat/lon × explanatory vars) |
| `download_geojson.py` | One-time script to download `china_provinces.geojson` for map rendering |

## Installation

```bash
pip install streamlit pandas numpy cvxpy scipy matplotlib seaborn plotly statsmodels requests openpyxl
```

The `stwr` library (spatiotemporal weighted regression) should be available locally. If not, install it from your research group's repository.

## Quick Start

```bash
streamlit run app.py
```

### First-time setup

Download the China province GeoJSON file for map visualization:

```bash
python download_geojson.py
```

## Usage Guide

### 1. Configure Parameters (Sidebar)

| Parameter | Description | Default |
|---|---|---|
| Returns to Scale (RTS) | VRS (Variable) or CRS (Constant) | VRS |
| P-value Threshold | Significance level for Step 3 path analysis | 0.05 |
| Path Frequency Threshold (%) | Minimum frequency for Sankey diagram paths | 1% |

### 2. Upload Data

- **SBM Data** (xlsx/csv): Inputs, desirable outputs, undesirable outputs, with ID and time columns
- **STWR Driver Data** (xlsx/csv): Explanatory variables with latitude, longitude, ID, and time columns

### 3. Run the Pipeline

| Step | Button | What It Does |
|---|---|---|
| Step 1 | **Run SBM Calculation** | Computes efficiency scores via Slacks-Based Measure with undesirable outputs |
| Step 2 | **Run STWR Regression** | Fits OLS → GWR → STWR for each year independently; outputs coefficients, t-values, p-values, local R² |
| Step 3 | **Discover Evolution Paths** | Identifies dominant driving factors per province-year and builds transition matrices |

### 4. Visualizations

- **Choropleth Map**: STWR coefficient spatial distribution across Chinese provinces (requires `china_provinces.geojson`)
- **Sankey Diagram**: Driving factor evolution flows across years with macaron color palette

### 5. AI Assistant

After completing the computation, ask questions in the sidebar. The AI assistant has access to all computed data (SBM efficiency, regression diagnostics, coefficients, path matrices) and can answer domain-specific questions.

> ⚠️ Replace `INTERNAL_API_KEY` and `ENDPOINT_ID` in `app.py` with your own API credentials.

## Data Format

### SBM Input (Excel or CSV)

Expected columns with any names (you select them in the UI):

| Column Role | Examples |
|---|---|
| ID | Province/city name (e.g., `Beijing`, `Shanghai`) |
| Time | Year (e.g., `2004`, `2005`, ...) |
| Inputs | Labor, capital, energy consumption, etc. |
| Desirable Outputs | Power generation, GDP, etc. |
| Undesirable Outputs | CO₂ emissions, pollutants, etc. |

### STWR Driver Input (Excel or CSV)

| Column Role | Examples |
|---|---|
| ID | Province/city name (matches SBM ID) |
| Time | Year (matches SBM time) |
| Latitude | Decimal degrees |
| Longitude | Decimal degrees |
| Explanatory Variables | Industrial structure, population, energy intensity, etc. |

## Output Files

After Step 2, a ZIP package is available containing yearly directories:

```
Stage2_STWR/
├── AICc.csv              # Model comparison: OLS vs GWR vs STWR
├── R2.csv                # R² comparison
├── RSS.csv               # Residual sum of squares comparison
├── Sigma.csv             # Sigma comparison
└── {year}/
    ├── stwr_params.csv      # STWR coefficients
    ├── stwr_t-Values.csv    # t-values
    ├── stwr_p-Values.csv    # p-values
    ├── stwr_z-Scores.csv    # z-scores
    ├── stwr_residuals.csv   # Residuals + local R²
    ├── gwr_params.csv       # GWR coefficients
    ├── ols_params.csv       # OLS coefficients
    └── summary.txt          # Yearly model summary
```

After Step 3, download the province evolution matrix and path statistics.

## Province Name Mapping

The app supports both **Chinese** and **English** province IDs. English names are automatically mapped to Chinese full names required by the GeoJSON map. Supported English IDs:

`Beijing`, `Tianjin`, `Hebei`, `Shanxi`, `Inner Mongolia`, `InnerMongolia`, `Liaoning`, `Jilin`, `Heilongjiang`, `Shanghai`, `Jiangsu`, `Zhejiang`, `Anhui`, `Fujian`, `Jiangxi`, `Shandong`, `Henan`, `Hubei`, `Hunan`, `Guangdong`, `Guangxi`, `Hainan`, `Chongqing`, `Sichuan`, `Guizhou`, `Yunnan`, `Tibet`, `Xizang`, `Shaanxi`, `Gansu`, `Qinghai`, `Ningxia`, `Xinjiang`

## Algorithm Details

### SBM (Slacks-Based Measure)

- Non-radial, non-oriented DEA model with undesirable outputs
- Normalizes input-output matrices to [0.1, 1.0] range
- Solver strategy: ECOS → SCS fallback via CVXPY

### STWR (Spatiotemporally Weighted Regression)

- Year-by-year independent fitting with auto bandwidth selection
- Primary: `Sel_Spt_BW` (spatiotemporal bandwidth)
- Fallback: `Sel_BW` (spatial-only) + default temporal decay
- Outputs: parameters, t-values, p-values, z-scores, residuals, local R²

### Path Evolution Analysis

- Dominant factor per province-year (highest |coefficient| with p < threshold)
- Adjacent-year transition matrix (e.g., `Industrial+ → Energy-`)
- Frequency statistics + Sankey diagram visualization# SBM-STWR-LLM


---

## v2: RAG retrieval + Agent orchestration + numerical verification

New modules
- `retrieval.py` — fine-grained province/year retrieval, stable evidence IDs, JSON evidence serialization.
- `agent_layer.py` — LLM planner (JSON task plan) + validated tool registry + executor.
- `verification.py` — numerical claim extraction, deterministic verification, and answer repair.
- `stwr_multislice.py` — correct multi-slice STWR (temporal kernel active).
- `prompts.py` — retrieval/analysis/synthesis/output-contract prompts.

Model / API key
The QA pipeline uses DeepSeek-Chat (OpenAI-compatible API). The key is NOT stored in the repo.
Provide it via environment variable, or create a git-ignored `local_secrets.py`:

```bash
export DEEPSEEK_API_KEY=your_key_here
# optional:
export DEEPSEEK_MODEL=deepseek-chat
export DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
```

`local_secrets.py` (git-ignored):
```python
DEEPSEEK_API_KEY = "your_key_here"
```

Run
```bash
pip install -r requirements.txt
python download_geojson.py      # fetch Natural Earth shapefiles for maps
streamlit run app.py
```

Data
- `结果文件/` — precomputed SBM/STWR/path snapshot used by the retrieval layer at startup.
- `data/` — raw input tables.

Agent mode is available as a sidebar toggle ("Agent-driven planning + numerical verification").
