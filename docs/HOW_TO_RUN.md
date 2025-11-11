# How to Run the Code - Complete Guide

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Pipeline](#running-the-pipeline)
4. [Running the Dashboard](#running-the-dashboard)
5. [Understanding Outputs](#understanding-outputs)
6. [Common Commands](#common-commands)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Python 3.8+** (Check: `python --version`)
2. **pip** (Python package manager)
3. **Git** (if cloning repository)

### System Requirements

- **OS**: Windows, macOS, or Linux
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: ~2GB for data and models
- **Internet**: Required for downloading real data (first time)

---

## Installation

### Step 1: Clone or Navigate to Project

```bash
# If you have the project folder, navigate to it:
cd Predictive-Analytics-for-Weather-Induced-Infrastructure-Failures

# Or if cloning:
git clone <repository-url>
cd Predictive-Analytics-for-Weather-Induced-Infrastructure-Failures
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**What this installs:**
- `numpy`, `pandas` - Data processing
- `scikit-learn` - Machine learning utilities
- `xgboost` - Gradient boosting model
- `torch` - PyTorch for LSTM
- `matplotlib`, `seaborn` - Plotting
- `geopandas`, `shapely` - Geospatial visualization
- `dash`, `plotly` - Interactive dashboard
- And more...

**Expected time:** 5-10 minutes (depending on internet speed)

### Step 3: Verify Installation

```bash
# Test Python imports
python -c "import torch, pandas, numpy, xgboost; print('✅ All packages installed!')"
```

---

## Running the Pipeline

### Option 1: Quick Start (Demo Data)

**Best for:** Testing, learning, quick experiments

```bash
python -m src.models.train_ensemble --config configs/default.yaml --mode demo
```

**What happens:**
1. Generates synthetic weather + failure data
2. Engineers features (lags, rolling stats)
3. Trains LSTM + Random Forest + XGBoost
4. Creates hybrid ensemble
5. Saves models and metrics

**Time:** ~2-5 minutes

### Option 2: Quick Run (Faster Testing)

**Best for:** Quick testing with reduced settings

```bash
python -m src.models.train_ensemble --config configs/default.yaml --mode demo --quick-run
```

**What's different:**
- Smaller dataset
- Fewer model iterations
- Faster training (~1-2 minutes)

### Option 3: Real Data (Production)

**Best for:** Real-world predictions, production use

```bash
python -m src.models.train_ensemble --config configs/default.yaml --mode real
```

**What happens:**
1. Downloads NOAA Storm Events data (if not cached)
2. Downloads GHCN weather station data (if not cached)
3. Processes 4 years of real data (2021-2024)
4. Trains models on 10 US states
5. Generates predictions based on actual events

**Time:** ~10-30 minutes (first time includes downloads)

**Note:** Data is cached in `data/raw/real/` after first download

### Option 4: Real Data Quick Run

```bash
python -m src.models.train_ensemble --config configs/default.yaml --mode real --quick-run
```

**Time:** ~2-5 minutes

---

## Command-Line Options

### Full Command Syntax

```bash
python -m src.models.train_ensemble [OPTIONS]
```

### Available Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config` | Path to config file | `--config configs/default.yaml` |
| `--mode` | Data mode: `demo` or `real` | `--mode real` |
| `--quick-run` | Use reduced settings | `--quick-run` |
| `--run-name` | Custom name for run | `--run-name my_experiment` |
| `--output-dir` | Custom output directory | `--output-dir custom_models/` |

### Examples

```bash
# Full training with custom name
python -m src.models.train_ensemble --config configs/default.yaml --mode real --run-name production_v1

# Quick demo test
python -m src.models.train_ensemble --mode demo --quick-run

# Custom config file
python -m src.models.train_ensemble --config my_config.yaml --mode demo
```

---

## Running the Dashboard

### Step 1: Train a Model First

```bash
# Train with demo data
python -m src.models.train_ensemble --mode demo

# OR train with real data
python -m src.models.train_ensemble --mode real
```

### Step 2: Start Dashboard

```bash
python -m src.dashboard.app
```

### Step 3: Open Browser

The dashboard will start and show:
```
🚀 Dashboard Starting...
📊 Open your browser and go to:
   http://localhost:8050
   or
   http://127.0.0.1:8050
```

**Important:** Use `localhost` or `127.0.0.1` in your browser, NOT `0.0.0.0`

Open your browser and go to: **http://localhost:8050**

### Step 4: Explore

- **Overview Tab**: Summary statistics and time series
- **Risk Map**: Interactive geographic visualization
- **Predictions**: Filterable table and charts
- **Model Performance**: Metrics comparison
- **Model Comparison**: Scatter plots

### Stop Dashboard

Press `Ctrl+C` in the terminal to stop the dashboard.

---

## Understanding Outputs

### Where Files Are Saved

All outputs are saved to: `models/<run_name>_<timestamp>/`

Example: `models/hybrid_weather_failure_20251110_232932/`

### Output Files

| File | Description |
|------|-------------|
| `metrics.json` | Model performance metrics (RMSE, MAE, R²) |
| `predictions.csv` | All predictions with actual vs predicted |
| `lstm.pt` | Trained LSTM model weights |
| `random_forest.joblib` | Trained Random Forest model |
| `xgboost.joblib` | Trained XGBoost model |
| `sequence_scaler.joblib` | Feature scaler for sequences |
| `actual_vs_predicted.png` | Scatter plot visualization |
| `residuals.png` | Residual analysis plot |
| `config_used.yaml` | Configuration used for training |
| `feature_columns.json` | List of feature names |
| `lstm_history.json` | LSTM training history |

### Latest Run

The most recent run is also copied to: `models/latest/`

This makes it easy to access the latest results without finding the timestamped folder.

---

## Common Commands

### 1. Train Model (Demo)

```bash
python -m src.models.train_ensemble --mode demo
```

### 2. Train Model (Real Data)

```bash
python -m src.models.train_ensemble --mode real
```

### 3. Quick Test Run

```bash
python -m src.models.train_ensemble --mode demo --quick-run
```

### 4. Start Dashboard

```bash
python -m src.dashboard.app
```

### 5. View Risk Map (Command Line)

```bash
python -m src.visualization.map_failures --artifacts models/latest/metrics.json
```

### 6. Run Tests

```bash
pytest
```

### 7. Check Code Style

```bash
black --check .
flake8
```

---

## Step-by-Step Workflow

### Complete Workflow Example

#### 1. First Time Setup

```bash
# Navigate to project
cd Predictive-Analytics-for-Weather-Induced-Infrastructure-Failures

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('✅ Ready!')"
```

#### 2. Quick Test (Demo Data)

```bash
# Quick test with demo data
python -m src.models.train_ensemble --mode demo --quick-run

# Check results
ls models/latest/
```

#### 3. View Results

```bash
# Option A: Start dashboard
python -m src.dashboard.app
# Then open http://localhost:8050

# Option B: View metrics file
cat models/latest/metrics.json

# Option C: View predictions
python -c "import pandas as pd; print(pd.read_csv('models/latest/predictions.csv').head())"
```

#### 4. Train on Real Data

```bash
# Train with real NOAA data
python -m src.models.train_ensemble --mode real

# This will:
# - Download data (first time only)
# - Process 4 years of data
# - Train models
# - Save results
```

#### 5. Analyze Results

```bash
# Start dashboard to explore
python -m src.dashboard.app

# Or view specific files
python -c "import json; print(json.dumps(json.load(open('models/latest/metrics.json')), indent=2))"
```

---

## Understanding the Process

### What Happens When You Run Training

```
1. Configuration Loading
   └─> Reads configs/default.yaml
   
2. Data Ingestion
   ├─> Demo: Generates synthetic data
   └─> Real: Loads/downloads NOAA data
   
3. Feature Engineering
   ├─> Creates lag features
   ├─> Calculates rolling statistics
   └─> Prepares sequences and tabular formats
   
4. Data Splitting
   ├─> Train: 70%
   ├─> Validation: 15%
   └─> Test: 15%
   
5. Model Training
   ├─> LSTM (temporal patterns)
   ├─> Random Forest (feature interactions)
   └─> XGBoost (gradient boosting)
   
6. Ensemble Creation
   └─> Combines all models
   
7. Evaluation
   ├─> Computes metrics
   ├─> Generates predictions
   └─> Creates visualizations
   
8. Save Artifacts
   └─> Saves everything to models/
```

---

## Troubleshooting

### Problem: "Module not found"

**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "Port 8050 already in use" (Dashboard)

**Solution:**
```python
# Edit src/dashboard/app.py, change port:
app.run_server(port=8080)  # Use different port
```

### Problem: "No data available" (Dashboard)

**Solution:**
```bash
# Train a model first
python -m src.models.train_ensemble --mode demo
```

### Problem: "Out of memory"

**Solution:**
```bash
# Use quick-run mode
python -m src.models.train_ensemble --mode demo --quick-run

# Or reduce batch size in config
```

### Problem: "Data download failed" (Real mode)

**Solution:**
- Check internet connection
- Verify NOAA websites are accessible
- Check firewall settings
- Try again later (servers may be busy)

### Problem: "CUDA out of memory" (GPU)

**Solution:**
```yaml
# In configs/default.yaml, change:
training:
  device: cpu  # Use CPU instead
```

---

## Quick Reference Card

### Most Common Commands

```bash
# Install
pip install -r requirements.txt

# Train (Demo)
python -m src.models.train_ensemble --mode demo

# Train (Real)
python -m src.models.train_ensemble --mode real

# Quick Test
python -m src.models.train_ensemble --mode demo --quick-run

# Dashboard
python -m src.dashboard.app

# Tests
pytest
```

### File Locations

```
Project Root/
├── configs/default.yaml          # Configuration
├── models/latest/                # Latest results
├── data/raw/real/                # Cached real data
└── src/                          # Source code
```

---

## Next Steps After Running

1. **View Dashboard**: `python -m src.dashboard.app`
2. **Check Metrics**: `models/latest/metrics.json`
3. **Analyze Predictions**: `models/latest/predictions.csv`
4. **View Plots**: `models/latest/*.png`
5. **Read Documentation**: `docs/` folder

---

## Tips for Success

1. **Start with Demo**: Test with demo data first
2. **Use Quick-Run**: Faster iteration during development
3. **Check Logs**: Watch terminal output for progress
4. **Monitor Resources**: Real data training uses more memory
5. **Save Configs**: Create custom configs for experiments
6. **Use Dashboard**: Easier than reading JSON files

---

## Getting Help

- Check `docs/` folder for detailed guides
- Review error messages in terminal
- Check `models/latest/` for outputs
- Verify data in `data/raw/real/` for real mode

---

This guide covers everything you need to run the code successfully! 🚀

