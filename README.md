# Predicting Weather-Induced Infrastructure Failures

Production-ready hybrid modelling pipeline that forecasts infrastructure failures driven by weather extremes. The repository ships with synthetic demo data so the full workflow—data ingestion, feature engineering, model training, evaluation, and geospatial visualisation—runs out of the box.

## Highlights
- **Hybrid ensemble** blending a PyTorch LSTM (temporal sequences) with RandomForest + XGBoost on engineered tabular features.
- **Configurable data adapters** with a synthetic demo generator and pluggable hooks for real NOAA/ERA5/outage datasets.
- **Feature engineering**: rolling stats, lagged impacts, vulnerability context, and sliding-window sequences.
- **Artifacts**: saved weights, metrics JSON, prediction tables, and evaluation plots (actual vs predicted, residuals).
- **Risk mapping**: choropleth-style point map of regional risk using GeoPandas.
- **Quality gates**: pytest unit tests, black/flake8 compliant code.

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Train the hybrid ensemble on synthetic demo data
python -m src.models.train_ensemble --config configs/default.yaml --mode demo

# 2b. (Optional) Pull NOAA + GHCN data and train on real events
python -m src.models.train_ensemble --config configs/default.yaml --mode real

# 3. Plot regional risk using the generated artifacts
python -m src.visualization.map_failures --artifacts models/latest/metrics.json
```

Artifacts are stored in `models/<run_name>_<timestamp>/` and mirrored to `models/latest/`. Metrics, predictions, and plots are ready for inspection immediately after training.

## Configuration
All behaviour is driven from `configs/default.yaml`:
- `data.*`: loader settings, including the demo generator parameters and time alignment options.
- `features.*`: lags, rolling windows, and base weather columns for engineering.
- `sequence.*`: sequence length & stride for the LSTM windows.
- `target.*`: choose `regression` or `classification` (80th percentile threshold by default).
- `training.*`: LSTM hyperparameters, tabular model grids, ensemble weights, device selection.
- `paths.*`: where to place artifacts and processed data.

Override values via CLI-friendly YAML edits or create additional config files for experiments. Use `--mode real` once real adapters are wired (see below).

## Pipeline Overview
1. **Data adapters (`src/data/download_and_preprocess.py`)**
   - `demo` mode fabricates weather + failure sequences across multiple regions with injected seasonality, extremes, and SVI context.
   - `real` mode downloads NOAA Storm Events damage reports and GHCN daily observations, aggregates them to state/day, and caches the raw files automatically.
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

## Real Data Sources & Caching
- **Weather observations**: Global Historical Climatology Network (GHCN) daily station files from NOAA (`https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/`).
- **Infrastructure impact proxy**: NOAA Storm Events “details” files (`https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/`), aggregated to state/day with damage estimates used as the failure target.
- Data is cached under `data/raw/real/` (configurable in `configs/default.yaml`) so repeated runs reuse downloads. Delete the directory to force a refresh.

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
models/            Saved models, metrics, and plots
src/
  data/            Adapters and feature engineering
  models/          Model definitions and trainer CLI
  utils/           I/O helpers and metrics
  visualization/   Plots and geospatial risk map
tests/             Pytest suite
```

## License
[MIT License](LICENSE)
