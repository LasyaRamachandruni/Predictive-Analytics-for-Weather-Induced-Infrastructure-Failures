# Detailed Process Flow Documentation
## Predictive Analytics for Weather-Induced Infrastructure Failures

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Phase-by-Phase Process](#phase-by-phase-process)
3. [Data Transformations](#data-transformations)
4. [Model Architecture Details](#model-architecture-details)
5. [Execution Flow](#execution-flow)
6. [Input/Output Specifications](#inputoutput-specifications)

---

## Overview

This project implements an end-to-end machine learning pipeline that predicts infrastructure failures (power outages, damage, etc.) based on weather patterns. The system uses a hybrid ensemble approach combining:

- **LSTM** (Long Short-Term Memory) for temporal pattern recognition
- **Random Forest** for feature interaction modeling
- **XGBoost** for gradient boosting
- **Ensemble Blending** to combine predictions

---

## Phase-by-Phase Process

### 🔵 Phase 1: Data Ingestion

**Objective**: Collect and prepare raw data for processing

#### Step 1.1: Mode Selection
- **Input**: Configuration file (`configs/default.yaml`)
- **Decision**: Choose between `demo` or `real` mode
- **Output**: Mode flag

#### Step 1.2: Data Generation/Download

**Demo Mode**:
```
1. Generate timestamp range
   - Start: 2024-01-01
   - Frequency: 6 hours
   - Periods: 240
   - Result: 240 timestamps

2. Generate regions
   - Number: 8 regions
   - Each region gets:
     * region_id: "Region_00", "Region_01", ...
     * latitude: random(30, 47)
     * longitude: random(-105, -70)
     * elevation: random(5, 500)
     * SVI score: random(0.1, 0.9)

3. Generate weather data for each region×timestamp
   - temperature: seasonal pattern + noise
   - humidity: correlated with temperature
   - precipitation: gamma distribution
   - wind_speed: normal distribution
   - snow_depth: conditional on temperature
   - extreme_event: 5% chance

4. Generate failures
   - Formula: base_rate + weather_effects + extreme_events
   - Poisson distribution for count
```

**Real Mode**:
```
1. Download Storm Events
   - Source: NOAA Storm Events database
   - Files: StormEvents_details-ftp_v1.0_d{year}_c{timestamp}.csv.gz
   - Years: 2021-2024
   - Extract: damage, event type, location, date

2. Download GHCN Weather
   - Source: Global Historical Climatology Network
   - Stations: 10 US weather stations
   - Extract: temperature, precipitation, wind, snow
   - Format: Fixed-width .dly files

3. Parse and aggregate
   - Storm events → aggregate by state/day
   - Weather → daily averages
   - Merge on state + date
```

#### Step 1.3: Data Merging
```
1. Create base index
   - MultiIndex: [region_id, timestamp]
   - Date range: start_date to end_date
   - Frequency: Daily

2. Merge weather data
   - Left join on [region_id, timestamp]
   - Forward fill missing values
   - Backward fill remaining gaps

3. Merge failure data
   - Left join on [region_id, timestamp]
   - Fill missing with 0

4. Add coordinates
   - Map region_id → latitude/longitude
   - From station metadata or demo generation
```

**Output**: Raw DataFrame
- Shape: [n_regions × n_timestamps, columns]
- Columns: region_id, timestamp, weather features, failures, lat, lon

---

### 🟢 Phase 2: Feature Engineering

**Objective**: Create predictive features from raw data

#### Step 2.1: Time Series Alignment

**Process**:
```python
For each region:
  1. Sort by timestamp
  2. Set timestamp as index
  3. Infer frequency (or use config)
  4. Create full date range (min to max)
  5. Reindex to full range
  6. Forward fill missing values
  7. Fill remaining NaN with 0
  8. Reset index
```

**Example**:
```
Before:
  region_id  timestamp  temp
  CA         2024-01-01  50
  CA         2024-01-03  52  ← Missing 2024-01-02

After:
  region_id  timestamp  temp
  CA         2024-01-01  50
  CA         2024-01-02  50  ← Forward filled
  CA         2024-01-03  52
```

#### Step 2.2: Lag Feature Creation

**Process**:
```python
For each numeric column (weather + target):
  For each lag in [1, 3, 6, 12, 24]:
    Create: column_lag_{lag} = column.shift(lag)
    Fill NaN with 0
```

**Example**:
```
timestamp  temp  temp_lag_1  temp_lag_3  failures  failures_lag_1
2024-01-01  50     0           0           2         0
2024-01-02  52    50           0           3         2
2024-01-03  48    52           0           1         3
2024-01-04  55    48          50           4         1
```

**Rationale**: Captures delayed effects (e.g., heavy rain today → failure tomorrow)

#### Step 2.3: Rolling Statistics

**Process**:
```python
For each numeric column:
  For each window in [3, 6, 12, 24]:
    Create rollmean_{window} = rolling(window).mean()
    Create rollmax_{window} = rolling(window).max()
    Create rollstd_{window} = rolling(window).std()
```

**Example**:
```
timestamp  temp  temp_rollmean_3  temp_rollmax_3  temp_rollstd_3
2024-01-01  50      50.0            50.0            0.0
2024-01-02  52      51.0            52.0            1.41
2024-01-03  48      50.0            52.0            2.0
2024-01-04  55      51.7            55.0            3.51
```

**Rationale**: Captures trends and volatility over time windows

#### Step 2.4: Target Transformations

**Process**:
```python
1. failures_diff_1 = failures - failures.shift(1)
   → Change from previous time step

2. failures_pct_change = (failures - prev) / prev * 100
   → Percentage change
   → Replace inf/-inf with 0
   → Fill NaN with 0
```

**Example**:
```
timestamp  failures  failures_diff_1  failures_pct_change
2024-01-01    2         0               0.0
2024-01-02    3         1              50.0
2024-01-03    1        -2             -66.7
2024-01-04    4         3             300.0
```

**Rationale**: Captures growth rates and changes

#### Step 2.5: Feature Collection

**Process**:
```python
1. Collect all new features in dictionary
2. Convert to DataFrame with matching index
3. Concatenate with original DataFrame
4. Fill NaN in lag features with 0
5. Drop rows where essential columns are NaN
   (region_id, timestamp, failures)
```

**Output**: Engineered DataFrame
- Original columns: ~10
- Engineered features: ~140
- Total columns: ~150
- Rows: Same as input (minus dropped NaN rows)

---

### 🟡 Phase 3: Data Splitting & Formatting

**Objective**: Prepare data for model training

#### Step 3.1: Chronological Split

**Process**:
```python
1. Get unique timestamps, sort chronologically
2. Calculate cutoffs:
   train_cutoff = int(total * 0.70)  # 70%
   val_cutoff = int(total * 0.85)    # 15% more
3. Assign splits:
   - timestamps[:train_cutoff] → "train"
   - timestamps[train_cutoff:val_cutoff] → "val"
   - timestamps[val_cutoff:] → "test"
4. Map back to DataFrame rows
```

**Example**:
```
Total timestamps: 240
Train: 168 timestamps (70%)
Val: 36 timestamps (15%)
Test: 36 timestamps (15%)

For 8 regions:
Train: 168 × 8 = 1,344 rows
Val: 36 × 8 = 288 rows
Test: 36 × 8 = 288 rows
```

**Rationale**: Time series must respect temporal order to prevent data leakage

#### Step 3.2: Feature Selection

**Process**:
```python
1. Get all numeric columns
2. Exclude:
   - target column (failures)
   - split column
   - metadata (longitude, latitude)
   - any columns in exclude_auto list
3. Result: ~150 feature columns
```

#### Step 3.3: Tabular Format Creation

**Process**:
```python
For each split (train/val/test):
  1. Filter DataFrame by split
  2. Extract feature columns → features DataFrame
  3. Extract target column → target Series
  4. Extract metadata → metadata DataFrame
     (region_id, timestamp, latitude, longitude)
```

**Output**: TabularDataset objects
- Features: [n_samples, n_features] DataFrame
- Target: [n_samples] Series
- Metadata: [n_samples, 4] DataFrame

#### Step 3.4: Sequence Format Creation

**Process**:
```python
For each region:
  For each split:
    1. Filter data for region + split
    2. Sort by timestamp
    3. Create sliding windows:
       - Window length: 24 time steps
       - Stride: 1 (overlapping windows)
    4. For each window:
       - Features: [24, n_features] array
       - Target: value at end of window
    5. Only include windows entirely within split
```

**Example**:
```
Region CA, Train split:
  Timestamps: [t0, t1, t2, ..., t167]
  
  Window 1: [t0:t23] → target = t23
  Window 2: [t1:t24] → target = t24
  Window 3: [t2:t25] → target = t25
  ...
  
  Result: ~145 windows (168 - 24 + 1)
```

**Output**: SequenceDataset objects
- Features: [n_samples, 24, n_features] numpy array
- Target: [n_samples] numpy array
- Metadata: [n_samples, 4] DataFrame

#### Step 3.5: Feature Scaling

**Process**:
```python
1. Fit StandardScaler on train sequence features
   - Compute mean and std for each feature
   
2. Transform all sequences (train/val/test)
   - Formula: (x - mean) / std
   
3. Store scaler for inference
```

**Rationale**: Neural networks (LSTM) require normalized inputs

---

### 🔴 Phase 4: Model Training

**Objective**: Train multiple models and combine them

#### Step 4.1: LSTM Training

**Architecture**:
```
Input: [batch_size, sequence_length=24, n_features=151]
  │
  ▼
LSTM Layer 1
  - Hidden size: 128
  - Number of layers: 2
  - Dropout: 0.2
  - Bidirectional: False (configurable)
  │
  ▼
Dropout (0.2)
  │
  ▼
Fully Connected Layer
  - Input: 128
  - Output: 1
  │
  ▼
Output: [batch_size, 1] (failure prediction)
```

**Training Process**:
```python
1. Create DataLoader
   - Batch size: 64
   - Shuffle: True (train), False (val)

2. Initialize model
   - Random weights
   - Move to device (CPU/GPU)

3. Training loop (max 25 epochs):
   For each epoch:
     For each batch:
       a. Forward pass → predictions
       b. Compute loss (MSE for regression)
       c. Backward pass → gradients
       d. Update weights (Adam optimizer)
     Evaluate on validation set
     Check early stopping (patience=5)
     
4. Save best model weights
```

**Early Stopping**:
- Monitor: Validation loss
- Patience: 5 epochs
- Min delta: 0.0005
- Stop if no improvement for 5 epochs

#### Step 4.2: Tabular Ensemble Training

**Random Forest**:
```python
Parameters:
  - n_estimators: 400 trees
  - max_depth: 12
  - min_samples_leaf: 2
  - n_jobs: -1 (use all CPUs)
  - random_state: 42

Training:
  - Fit on train tabular features
  - Predict on all splits
```

**XGBoost**:
```python
Parameters:
  - n_estimators: 500 trees
  - learning_rate: 0.05
  - max_depth: 5
  - subsample: 0.9
  - colsample_bytree: 0.8
  - reg_lambda: 1.0 (L2 regularization)
  - random_state: 42

Training:
  - Fit on train tabular features
  - Predict on all splits
```

**Blending**:
```python
For each sample:
  blended_tabular = 0.5 × RF_pred + 0.5 × XGB_pred
```

**Rationale**: Combines strengths of both tree-based models

#### Step 4.3: Hybrid Ensemble Creation

**Process**:
```python
For each split (train/val/test):
  For each sample:
    1. Get LSTM prediction (from sequences)
    2. Get tabular blend prediction (from features)
    3. Combine:
       hybrid_pred = weight_tabular × tabular_pred + 
                     (1 - weight_tabular) × lstm_pred
       
       Default: weight_tabular = 0.5
       → 50% tabular + 50% LSTM
```

**Rationale**: Combines temporal patterns (LSTM) with feature interactions (trees)

---

### 🟣 Phase 5: Evaluation & Output

**Objective**: Assess model performance and save results

#### Step 5.1: Prediction Generation

**Process**:
```python
For each split:
  1. Get LSTM predictions
     - Load sequences
     - Forward pass through model
     - Extract predictions
     
  2. Get tabular predictions
     - Load features
     - RF.predict()
     - XGB.predict()
     - Blend: 0.5 × RF + 0.5 × XGB
     
  3. Create hybrid predictions
     - 0.5 × tabular + 0.5 × LSTM
     
  4. Store in DataFrame:
     - region_id, timestamp
     - target (actual)
     - rf_pred, xgb_pred
     - tabular_ensemble
     - lstm_pred
     - hybrid_pred
     - split
```

#### Step 5.2: Metrics Computation

**Regression Metrics** (if target_type = "regression"):
```python
For each model (RF, XGB, Tabular, LSTM, Hybrid):
  RMSE = sqrt(mean((y_true - y_pred)²))
  MAE = mean(|y_true - y_pred|)
  R² = 1 - (SS_res / SS_tot)
```

**Classification Metrics** (if target_type = "classification"):
```python
For each model:
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  Precision = TP / (TP + FP)
  Recall = TP / (TP + FN)
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ROC-AUC = Area under ROC curve
  PR-AUC = Area under Precision-Recall curve
```

**Output**: metrics.json
```json
{
  "metrics": {
    "train": {
      "hybrid_ensemble": {"rmse": 1.23, "mae": 0.98, "r2": 0.85},
      "lstm": {...},
      "tabular_ensemble": {...}
    },
    "val": {...},
    "test": {...}
  }
}
```

#### Step 5.3: Visualization Generation

**Actual vs Predicted Plot**:
```python
1. Extract test set predictions
2. Create scatter plot:
   - X-axis: Actual values
   - Y-axis: Predicted values
   - Diagonal line: Perfect prediction
3. Save as PNG
```

**Residuals Plot**:
```python
1. Compute residuals = actual - predicted
2. Plot residuals vs predicted
3. Check for patterns (should be random)
4. Save as PNG
```

**Geospatial Risk Map** (optional):
```python
1. Load predictions with lat/lon
2. Create GeoDataFrame
3. Plot on map with color-coded risk
4. Save as PNG
```

#### Step 5.4: Artifact Saving

**Saved Files**:
```
models/<run_name>_<timestamp>/
  ├── lstm.pt                    # LSTM model weights
  ├── random_forest.joblib       # RF model
  ├── xgboost.joblib             # XGBoost model
  ├── sequence_scaler.joblib     # Feature scaler
  ├── metrics.json               # Performance metrics
  ├── predictions.csv            # All predictions
  ├── actual_vs_predicted.png    # Plot
  ├── residuals.png               # Plot
  ├── config_used.yaml           # Configuration
  ├── feature_columns.json        # Feature names
  └── lstm_history.json          # Training history
```

**Also copied to**: `models/latest/` for easy access

---

## Data Transformations

### Input → Output Flow

```
Raw Data (Demo Mode)
  Input: None (generated)
  Output: DataFrame
    - 8 regions × 240 timestamps = 1,920 rows
    - Columns: region_id, timestamp, temp, humidity, precip, wind, failures, lat, lon

↓

Feature Engineering
  Input: 1,920 rows × 10 columns
  Output: 1,920 rows × 150 columns
  Added: ~140 engineered features

↓

Data Splitting
  Input: 1,920 rows × 150 columns
  Output:
    - Train: 1,344 rows
    - Val: 288 rows
    - Test: 288 rows

↓

Format Creation
  Tabular:
    - Train: [1,344, 150]
    - Val: [288, 150]
    - Test: [288, 150]
  
  Sequences:
    - Train: [~81, 24, 150]
    - Val: [~18, 24, 150]
    - Test: [~18, 24, 150]

↓

Model Training
  Input: Formatted data
  Output: Trained models + predictions

↓

Evaluation
  Input: Predictions + actuals
  Output: Metrics + plots + saved artifacts
```

---

## Model Architecture Details

### LSTM Architecture

```python
class LSTMOutagePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,      # 151 features
            hidden_size=hidden_size,    # 128
            num_layers=num_layers,      # 2
            dropout=dropout,            # 0.2
            bidirectional=bidirectional # False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)  # Output: 1 value
    
    def forward(self, x):
        # x: [batch, seq_len=24, features=151]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch, 24, 128]
        last_output = lstm_out[:, -1, :]  # Take last time step
        # last_output: [batch, 128]
        dropped = self.dropout(last_output)
        output = self.fc(dropped)
        # output: [batch, 1]
        return output
```

### Tabular Ensemble Architecture

```python
class TabularEnsemble:
    def __init__(self):
        self.rf = RandomForestRegressor(...)
        self.xgb = XGBRegressor(...)
    
    def fit(self, X, y):
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
    
    def predict(self, X, alpha=0.5):
        rf_pred = self.rf.predict(X)
        xgb_pred = self.xgb.predict(X)
        blended = alpha * rf_pred + (1 - alpha) * xgb_pred
        return EnsemblePrediction(rf_pred, xgb_pred, blended)
```

### Hybrid Ensemble

```python
def blend_with_lstm(tabular_pred, lstm_pred, weight_tabular=0.5):
    weight_lstm = 1 - weight_tabular
    hybrid = weight_tabular * tabular_pred + weight_lstm * lstm_pred
    return hybrid
```

---

## Execution Flow

### Command Line Execution

```bash
python -m src.models.train_ensemble --config configs/default.yaml --mode demo
```

### Internal Execution

```python
1. main()
   ├─ parse_args()
   ├─ load_yaml_config()
   ├─ apply_quick_run_overrides()  # If --quick-run
   └─ train_and_evaluate()
       ├─ set_seed()
       ├─ get_device()
       ├─ run_data_pipeline()      # Phases 1-3
       │   ├─ load_dataset()
       │   ├─ _align_time_series()
       │   ├─ _engineer_features()
       │   ├─ _assign_splits()
       │   ├─ _build_sequence_datasets()
       │   └─ _build_tabular_datasets()
       ├─ prepare_dataloader()
       ├─ train_lstm()             # Phase 4.1
       ├─ TabularEnsemble.fit()    # Phase 4.2
       ├─ predict_lstm()
       ├─ ensemble.predict()
       ├─ blend_with_lstm()         # Phase 4.3
       ├─ compute_metrics()         # Phase 5.2
       ├─ plot_actual_vs_predicted() # Phase 5.3
       ├─ plot_residuals()
       └─ save artifacts            # Phase 5.4
```

---

## Input/Output Specifications

### Input Specifications

**Configuration File** (`configs/default.yaml`):
- YAML format
- Contains: data settings, feature engineering params, model hyperparameters

**Data (Real Mode)**:
- Storm Events: CSV.gz files from NOAA
- GHCN Weather: Fixed-width .dly files
- Station metadata: Text file with lat/lon/elevation

### Output Specifications

**Models**:
- LSTM: PyTorch .pt file (state dict + metadata)
- RF/XGB: Joblib .joblib files
- Scaler: Joblib .joblib file

**Metrics**:
- JSON format
- Contains: RMSE, MAE, R² for each model and split

**Predictions**:
- CSV format
- Columns: region_id, timestamp, target, predictions, split

**Visualizations**:
- PNG format
- 300 DPI resolution
- Actual vs Predicted, Residuals plots

---

This detailed process flow provides a complete understanding of how the system works from start to finish!

