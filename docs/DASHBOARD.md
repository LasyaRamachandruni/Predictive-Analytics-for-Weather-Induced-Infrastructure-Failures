# Interactive Dashboard Guide

## 🎯 Overview

The project includes an interactive Dash dashboard for visualizing infrastructure failure predictions, model performance, and risk analysis.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `dash` - Web framework
- `dash-bootstrap-components` - UI components
- `plotly` - Interactive charts

### 2. Run the Dashboard

```bash
# Option 1: Run as module
python -m src.dashboard.app

# Option 2: Run directly
python src/dashboard/app.py
```

### 3. Access the Dashboard

**Important:** Use `localhost` or `127.0.0.1`, NOT `0.0.0.0`

Open your browser and navigate to:
```
http://localhost:8050
```

**OR**

```
http://127.0.0.1:8050
```

**Note:** `0.0.0.0` is a server bind address, not a browser URL. Always use `localhost` or `127.0.0.1` in your browser.

## 📊 Dashboard Features

### 1. **Overview Tab**
- Summary statistics (total predictions, regions, averages)
- Model performance metrics (RMSE, MAE, R²)
- Time series plot of predictions vs actuals

### 2. **Risk Map Tab**
- Interactive geographic map showing failure risk by region
- Filter by split (train/val/test)
- Filter by region
- Color-coded risk levels
- Size of markers indicates prediction magnitude

### 3. **Predictions Tab**
- Filterable predictions table
- Filters:
  - Split (train/val/test/all)
  - Region
  - Date range
- Time series chart of filtered predictions
- Shows actual vs predicted values

### 4. **Model Performance Tab**
- Performance metrics comparison across models
- Charts for:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (Coefficient of Determination)
- Comparison across train/val/test splits
- Comparison across models (Hybrid, LSTM, Tabular)

### 5. **Model Comparison Tab**
- Scatter plots comparing models
- Actual vs Predicted scatter plot
- LSTM vs Tabular comparison
- Perfect prediction line for reference

## 🎨 Features

### Interactive Elements
- **Filters**: Dropdown menus and date pickers
- **Hover Tooltips**: Detailed information on hover
- **Zoom & Pan**: Interactive charts
- **Responsive Design**: Works on different screen sizes

### Data Loading
- Automatically loads from `models/latest/`
- Loads:
  - `predictions.csv` - Prediction data
  - `metrics.json` - Performance metrics
  - `feature_columns.json` - Feature information

### Error Handling
- Graceful handling of missing data
- Warning messages when data is unavailable
- Fallback displays

## 🔧 Customization

### Change Default Artifacts Path

Edit `src/dashboard/app.py`:

```python
DEFAULT_ARTIFACTS_PATH = Path("models/latest")  # Change this
```

### Change Port

Edit the last line in `src/dashboard/app.py`:

```python
app.run_server(debug=True, host="0.0.0.0", port=8050)  # Change port
```

### Add New Tabs

1. Add tab to the tabs list:
```python
dbc.Tab(label="New Tab", tab_id="newtab"),
```

2. Add rendering function:
```python
def render_newtab() -> html.Div:
    return dbc.Row([...])
```

3. Update callback:
```python
elif active_tab == "newtab":
    return render_newtab()
```

## 📱 Usage Tips

1. **First Time**: Make sure you've trained a model first to generate artifacts
2. **Data Updates**: Refresh the page after training a new model
3. **Filters**: Use filters to focus on specific regions or time periods
4. **Charts**: Click legend items to show/hide data series
5. **Export**: Use browser screenshot or print to PDF for reports

## 🐛 Troubleshooting

### Dashboard won't start
- Check if port 8050 is available
- Try a different port: `app.run_server(port=8080)`

### No data showing
- Ensure you've run training: `python -m src.models.train_ensemble --mode demo`
- Check that `models/latest/` contains the required files

### Charts not loading
- Check browser console for errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### Map not showing
- Ensure `geopandas` and `plotly` are installed
- Check that predictions have latitude/longitude columns

## 🎯 Example Workflow

1. **Train Model**:
   ```bash
   python -m src.models.train_ensemble --config configs/default.yaml --mode demo
   ```

2. **Start Dashboard**:
   ```bash
   python -m src.dashboard.app
   ```

3. **Explore**:
   - Check Overview for summary
   - View Risk Map for geographic insights
   - Analyze Predictions with filters
   - Compare Model Performance
   - Review Model Comparison charts

## 📚 Additional Resources

- [Dash Documentation](https://dash.plotly.com/)
- [Plotly Python](https://plotly.com/python/)
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)

