# Predicting Weather-Induced Infrastructure Failures

Production-ready hybrid modelling pipeline that forecasts infrastructure failures driven by weather extremes. Uses real-world data from NOAA APIs, US Census, BEA, and FRED for comprehensive, credible predictions.

## Highlights
- **Hybrid ensemble** blending a PyTorch LSTM (temporal sequences) with RandomForest + XGBoost on engineered tabular features.
- **Comprehensive data sources**: Weather (NOAA GHCN), storm events, infrastructure age, population density, and economic indicators.
- **Real API data integration**: NOAA weather and storm events, US Census demographics, BEA/FRED economic indicators.
- **Feature engineering**: rolling stats, lagged impacts, vulnerability context, and sliding-window sequences.
- **Artifacts**: saved weights, metrics JSON, prediction tables, and evaluation plots (actual vs predicted, residuals).
- **Interactive Dashboard**: Web-based visualization with maps, charts, and predictions.
- **Risk mapping**: choropleth-style point map of regional risk using GeoPandas.
- **Quality gates**: pytest unit tests, black/flake8 compliant code.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root with your API keys:
```bash
NOAA_API_TOKEN=your_noaa_token_here
CENSUS_API_KEY=your_census_key_here
BEA_API_KEY=your_bea_key_here
FRED_API_KEY=your_fred_key_here
```

See [docs/API_QUICK_START.md](docs/API_QUICK_START.md) for detailed setup instructions.

### 3. Train the Model

**Full Training:**
```bash
python -m src.models.train_ensemble --config configs/default.yaml --mode real
```

**Quick Run (Faster Testing):**
```bash
python -m src.models.train_ensemble --mode real --quick-run
```

### 4. View Results

**Interactive Dashboard (Recommended)**
```bash
python -m src.dashboard.app
# Then open http://localhost:8050 in your browser
# (Use localhost or 127.0.0.1, NOT 0.0.0.0)
```

**OR Command-Line Visualization**
```bash
python -m src.visualization.map_failures --artifacts models/latest/metrics.json
```

📖 **For detailed instructions, see [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md)**

Artifacts are stored in `models/<run_name>_<timestamp>/` and mirrored to `models/latest/`. Metrics, predictions, and plots are ready for inspection immediately after training.

## Configuration
All behaviour is driven from `configs/default.yaml`:
- `data.*`: real data loader settings (NOAA Storm Events, GHCN weather stations, API configurations).
- `features.*`: lags, rolling windows, and base weather columns for engineering.
- `sequence.*`: sequence length & stride for the LSTM windows.
- `target.*`: choose `regression` or `classification` (80th percentile threshold by default).
- `training.*`: LSTM hyperparameters, tabular model grids, ensemble weights, device selection.
- `paths.*`: where to place artifacts and processed data.

Override values via CLI-friendly YAML edits or create additional config files for experiments.

## Pipeline Overview
1. **Data adapters (`src/data/download_and_preprocess.py`)**
   - Downloads NOAA Storm Events damage reports and GHCN daily weather observations via APIs
   - Aggregates data to state/day level and caches raw files automatically
   - Integrates additional data sources: Census demographics, BEA/FRED economic indicators
2. **Feature engineering (`src/data/data_pipeline.py`)**
   - Continuous timeline alignment, missing data imputation, rolling means/max/std, lag features, growth rates, and deterministic time-based splits.
   - Builds both tabular matrices and scaled sliding-window tensors with aligned metadata for ensembling.
3. **Models (`src/models/`)**
   - `lstm_model.py`: configurable, optionally bidirectional LSTM with dropout.
   - `ensemble.py`: tabular RandomForest/XGBoost bundle plus blending utilities.
   - `train_ensemble.py`: CLI trainer handling seeding, dataloaders, early stopping, metrics, artifact export, and plotting.
4. **Visualisation (`src/visualization/`)**
   - `plot_results.py`: reusable actual-vs-predicted and residual plotting helpers.
   - `map_failures.py`: GeoPandas-based mapping fed by saved prediction tables.

## Data Sources & Caching

### Primary Data Sources (✅ Already Working - Real Data)
- **Weather observations**: Global Historical Climatology Network (GHCN) daily station files from NOAA (`https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/`).
  - ✅ **Active**: 10 US weather stations, 2021-2024
  - 🆕 **API Support**: Now supports NOAA APIs (CDO, NCEI) with automatic fallback to file downloads
  - **See**: `docs/NOAA_DATASETS_DOCUMENTED.md` and `docs/NOAA_API_INTEGRATION.md`
  
- **Infrastructure impact proxy**: NOAA Storm Events "details" files (`https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/`), aggregated to state/day with damage estimates used as the failure target.
  - ✅ **Active**: Storm events database, 2021-2024
  - **Note**: Uses file downloads (no direct API available)
  - **See**: `docs/NOAA_DATASETS_DOCUMENTED.md` for complete documentation

### Additional Data Sources (New!)
- **Infrastructure Age & Condition**: Average infrastructure age, maintenance scores, and condition ratings by region.
- **Population Data**: Population density, urban percentage, and total population.
- **Economic Indicators**: GDP per capita, infrastructure investment, and poverty rates.

Data is cached under `data/raw/real/` (configurable in `configs/default.yaml`) so repeated runs reuse downloads. Delete the directory to force a refresh.

**See `docs/ADDING_MORE_DATA.md` for details on adding more data sources.**

**✅ Real API Data**: All data sources use real APIs (NOAA, Census, BEA, FRED). API keys are required - see setup instructions above. 

**🆕 Real API Integration**:
- `docs/REAL_API_SOURCES_VERIFIED.md` - Verified government API sources (2024)
- `docs/API_QUICK_START.md` - Get real data in 10 minutes
- `docs/API_INTEGRATION_GUIDE.md` - Complete integration guide
- `docs/DATA_SOURCES_EXPLAINED.md` - What's real vs fabricated

## Switching to Classification Mode
Set `target.type: classification` in the config. The pipeline will:
- Derive the threshold from the configured percentile.
- Train `RandomForestClassifier`, `XGBClassifier`, and use `BCEWithLogitsLoss` for the LSTM.
- Report accuracy, precision/recall, F1, ROC-AUC, PR-AUC.
- Persist probabilities and predicted classes for downstream analysis.

## Integrating Real Data Sources
1. Provide loader paths in the config:
   ```yaml
   data:
     real:
       weather_path: /path/to/weather.csv
       outages_path: /path/to/outages.csv
       context_path: /path/to/context.csv
   ```
2. Implement the TODO sections in `_load_real_dataset` (e.g., API fetches, quality checks, custom joins).
3. Launch training with `--mode real`.
4. Extend feature engineering or adapters as needed—docstrings describe expected schemas (`region_id`, `timestamp`, features…, `failures`).

## Testing & Quality
```bash
pytest
flake8
black --check .
```
Key tests cover data-pipeline shapes, LSTM forward pass, and a quick smoke run of the trainer (writes artifacts to a temp directory).

For rapid iteration use the built-in quick settings:
```bash
python -m src.models.train_ensemble --config configs/default.yaml --mode demo --quick-run
```
This trims the synthetic dataset, reduces estimator counts, and limits LSTM epochs for fast feedback.

## Adding a New Adapter
1. Create a loader function that returns a DataFrame with at least `region_id`, `timestamp`, features, and `failures`.
2. Register it inside `load_dataset` (e.g., `mode="my_source"`).
3. Update the config to pass adapter-specific parameters.
4. Optionally, extend `data_pipeline.py` to craft custom features for your source.

## Repository Structure
```
configs/           Default experiment configuration
data/              Storage bucket for raw/processed data (gitignored)
docs/              Documentation (architecture, process flow, diagrams)
models/            Saved models, metrics, and plots (gitignored)
src/
  data/            Adapters and feature engineering
  models/          Model definitions and trainer CLI
  utils/           I/O helpers and metrics
  visualization/   Plots and geospatial risk map
tests/             Pytest suite
```

## Interactive Dashboard

The project includes an interactive Dash dashboard for visualizing predictions and model performance:

```bash
# Start the dashboard
python -m src.dashboard.app
```

Then open your browser to `http://localhost:8050`

**Features:**
- 📊 Overview with summary statistics
- 🗺️ Interactive risk map
- 📈 Filterable predictions table and charts
- 📉 Model performance metrics
- 🔍 Model comparison visualizations

See [docs/DASHBOARD.md](docs/DASHBOARD.md) for detailed dashboard documentation.

## Documentation

### 🆕 Start Here
- **[PROJECT_EXPLAINED.md](docs/PROJECT_EXPLAINED.md)** - Simple, clear explanation of what the project does (perfect for understanding!)
- **[NOAA_DATASETS_DOCUMENTED.md](docs/NOAA_DATASETS_DOCUMENTED.md)** - Complete documentation of NOAA datasets you're already using! ✅
- **[NOAA_API_INTEGRATION.md](docs/NOAA_API_INTEGRATION.md)** - 🆕 Use NOAA APIs instead of file downloads! 🚀
- **[API_QUICK_START.md](docs/API_QUICK_START.md)** - Get real API data in 10 minutes! 🚀
- **[REAL_API_SOURCES_VERIFIED.md](docs/REAL_API_SOURCES_VERIFIED.md)** - Verified government API sources (2024)
- **[ADDING_MORE_DATA.md](docs/ADDING_MORE_DATA.md)** - Comprehensive guide for adding more data sources

### Technical Documentation
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design details
- **[ARCHITECTURE_DIAGRAM.md](docs/ARCHITECTURE_DIAGRAM.md)** - Visual architecture diagrams
- **[PROCESS_FLOW.md](docs/PROCESS_FLOW.md)** - Detailed step-by-step process documentation
- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Project organization guide
- **[DASHBOARD.md](docs/DASHBOARD.md)** - Interactive dashboard guide
- **[REAL_DATA_GUIDE.md](docs/REAL_DATA_GUIDE.md)** - Guide for working with real NOAA data
- **[HOW_TO_RUN.md](docs/HOW_TO_RUN.md)** - Complete step-by-step guide to running the code
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common errors and solutions

## License
[MIT License](LICENSE)
