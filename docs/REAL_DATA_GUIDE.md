# Working with Real Data Guide

## 📊 Current Real Data Status

Your project already has real data downloaded and ready to use:

- **Total Rows**: 14,610
- **Regions**: 10 US states (AZ, CA, CO, FL, GA, IL, NC, NY, TX, WA)
- **Date Range**: 2021-01-01 to 2024-12-31 (4 years)
- **Data Sources**:
  - NOAA Storm Events (infrastructure damage/failures)
  - GHCN Weather Stations (temperature, precipitation, wind, etc.)

## 🚀 Training with Real Data

### Quick Start

```bash
# Train on real data
python -m src.models.train_ensemble --config configs/default.yaml --mode real
```

### What Happens

1. **Data Loading**: Loads real weather and storm event data
2. **Feature Engineering**: Creates lag features, rolling statistics
3. **Model Training**: Trains LSTM + Random Forest + XGBoost
4. **Evaluation**: Generates metrics and predictions
5. **Artifacts**: Saves models and results to `models/`

### Expected Results

- **Training Time**: Longer than demo (more data)
- **Data Quality**: Real-world patterns and noise
- **Predictions**: Based on actual historical events
- **Metrics**: More realistic performance indicators

## 📋 Real Data Details

### Regions (States)

| State | Station ID | Location |
|-------|-----------|----------|
| AZ | USW00023183 | Phoenix |
| CA | USW00023234 | Los Angeles |
| CO | USW00023062 | Denver |
| FL | USW00012839 | Miami |
| GA | USW00053863 | Atlanta |
| IL | USW00094846 | Chicago |
| NC | USW00013723 | Charlotte |
| NY | USW00094728 | New York |
| TX | USW00013904 | Dallas/Fort Worth |
| WA | USW00024233 | Seattle |

### Weather Features

- `tavg_c`: Average temperature (°C)
- `tmax_c`: Maximum temperature (°C)
- `tmin_c`: Minimum temperature (°C)
- `prcp_mm`: Precipitation (mm)
- `awnd_ms`: Average wind speed (m/s)
- `snwd_mm`: Snow depth (mm)
- `snow_mm`: Snowfall (mm)

### Failure Data

- `failures`: Infrastructure damage (thousands USD)
- `event_count`: Total storm events
- `wind_event_count`: Wind-related events
- `flood_event_count`: Flood events
- `winter_event_count`: Winter weather events
- `hail_event_count`: Hail events
- `tornado_event_count`: Tornado events
- `lightning_event_count`: Lightning events

## ⚙️ Configuration

The real data configuration is in `configs/default.yaml`:

```yaml
real:
  start_date: "2021-01-01"
  end_date: "2024-12-31"
  cache_dir: data/raw/real
  storm_events:
    years: [2021, 2022, 2023, 2024]
  ghcn:
    stations:
      CA: USW00023234
      # ... other stations
```

### Customizing Real Data

#### Add More Regions

1. Find GHCN station IDs: https://www.ncei.noaa.gov/data/ghcn-daily/
2. Add to config:
```yaml
ghcn:
  stations:
    OR: USW00024229  # Portland, Oregon
    NV: USW00023169  # Las Vegas, Nevada
```

#### Change Date Range

```yaml
real:
  start_date: "2020-01-01"  # Earlier start
  end_date: "2023-12-31"   # Earlier end
```

#### Add More Years

```yaml
storm_events:
  years: [2020, 2021, 2022, 2023, 2024]  # Add 2020
```

## 🔍 Data Quality Notes

### Real Data Characteristics

1. **Missing Values**: Some weather stations may have gaps
   - Handled by forward/backward filling
   - Check data quality after loading

2. **Data Quality**: Real data has noise and outliers
   - Normal for real-world datasets
   - Models should handle this

3. **Temporal Patterns**: Real seasonal patterns
   - More realistic than synthetic data
   - Better for production use

4. **Failure Distribution**: Based on actual events
   - May be sparse (many zeros)
   - Reflects real-world patterns

## 📈 Training Tips

### For Real Data

1. **Longer Training**: Real data is larger, training takes longer
   - Use `--quick-run` for testing
   - Full training may take 10-30 minutes

2. **Monitor Performance**: Check metrics carefully
   - Real data may have different patterns
   - Adjust hyperparameters if needed

3. **Feature Engineering**: May need adjustments
   - Real data has different scales
   - Consider feature selection

4. **Validation**: Use proper train/val/test splits
   - Chronological split is important
   - Don't leak future data

## 🎯 Quick Test

Test real data loading:

```bash
python test_real_data.py
```

This will show:
- Number of rows and regions
- Date range
- Column names
- Data summary statistics

## 🐛 Troubleshooting

### Data Not Loading

1. Check internet connection (for downloads)
2. Verify cache directory exists: `data/raw/real/`
3. Check file permissions

### Missing Data

1. Some weather stations may have gaps
2. System handles this automatically
3. Check logs for warnings

### Slow Loading

1. First time downloads data (slower)
2. Subsequent runs use cached data (faster)
3. Large files take time to process

## 📊 Comparing Demo vs Real

| Aspect | Demo Data | Real Data |
|--------|-----------|-----------|
| **Size** | ~2,000 rows | ~14,600 rows |
| **Regions** | 8 synthetic | 10 US states |
| **Time Range** | 60 days | 4 years |
| **Patterns** | Simulated | Real-world |
| **Training Time** | ~2-5 min | ~10-30 min |
| **Use Case** | Testing | Production |

## 🚀 Next Steps

1. **Train Model**: `python -m src.models.train_ensemble --mode real`
2. **View Dashboard**: `python -m src.dashboard.app`
3. **Analyze Results**: Check `models/latest/metrics.json`
4. **Compare Models**: Use dashboard comparison tab

## 📚 Additional Resources

- [NOAA Storm Events](https://www.ncdc.noaa.gov/stormevents/)
- [GHCN Daily Data](https://www.ncei.noaa.gov/data/ghcn-daily/)
- [Station Metadata](https://www.ncei.noaa.gov/data/ghcn-daily/)

