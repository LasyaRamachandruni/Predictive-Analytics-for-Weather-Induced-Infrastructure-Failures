# NOAA Datasets - Complete Documentation

## 📊 Overview

Your project **already uses real NOAA datasets** for weather and infrastructure failure data. These are **production-ready, real-world data sources** that are actively downloaded and used in your pipeline.

---

## ✅ Currently Used NOAA Datasets

### 1. **NOAA GHCN Daily Weather Data** ✅

**Status**: ✅ **ACTIVE - Already Integrated**

#### **What It Is**:
- **GHCN** = Global Historical Climatology Network
- Daily weather observations from weather stations worldwide
- One of the most comprehensive weather datasets available

#### **Official URLs**:
- **Base URL**: `https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/`
- **Station Metadata**: `https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt`
- **Documentation**: https://www.ncei.noaa.gov/data/ghcn-daily/
- **Data Format**: Fixed-width ASCII files (`.dly` format)

#### **What Data You Get**:
- `tavg_c`: Average temperature (°C)
- `tmax_c`: Maximum temperature (°C)
- `tmin_c`: Minimum temperature (°C)
- `prcp_mm`: Precipitation (mm)
- `awnd_ms`: Average wind speed (m/s)
- `snwd_mm`: Snow depth (mm)
- `snow_mm`: Snowfall (mm)

#### **How It Works in Your Project**:
1. **Station Selection**: 10 US weather stations (one per state)
2. **Download**: Files downloaded from NOAA servers
3. **Parsing**: Fixed-width format parsed into DataFrame
4. **Caching**: Files cached in `data/raw/real/ghcn/`
5. **Merging**: Combined with storm events data

#### **Current Stations** (from `configs/default.yaml`):
| State | Station ID | Location |
|-------|-----------|----------|
| CA | USW00023234 | Los Angeles |
| TX | USW00013904 | Dallas/Fort Worth |
| FL | USW00012839 | Miami |
| NY | USW00094728 | New York |
| IL | USW00094846 | Chicago |
| GA | USW00053863 | Atlanta |
| WA | USW00024233 | Seattle |
| AZ | USW00023183 | Phoenix |
| NC | USW00013723 | Charlotte |
| CO | USW00023062 | Denver |

#### **Data Format**:
```
Fixed-width format:
- Station ID: 11 characters
- Date: 8 characters (YYYYMMDD)
- Element: 4 characters (TMAX, TMIN, PRCP, etc.)
- Value: 5 characters
- Quality flags: 3 characters
```

#### **Update Frequency**: Daily (new data added daily)

#### **Historical Coverage**: 
- Some stations have data back to 1800s
- Your project uses: 2021-2024 (4 years)

#### **Code Location**:
- `src/data/download_and_preprocess.py`
- Functions: `_download_ghcn_station()`, `_parse_ghcn_daily()`, `_load_ghcn_weather()`

---

### 2. **NOAA Storm Events Database** ✅

**Status**: ✅ **ACTIVE - Already Integrated**

#### **What It Is**:
- Comprehensive database of storm events in the US
- Includes damage estimates, event types, locations, dates
- Used as a **proxy for infrastructure failures**

#### **Official URLs**:
- **Base URL**: `https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/`
- **Documentation**: https://www.ncdc.noaa.gov/stormevents/
- **Data Format**: CSV files (compressed `.csv.gz`)

#### **What Data You Get**:
- `failures`: Infrastructure damage (thousands USD)
- `event_count`: Total storm events per day/state
- `wind_event_count`: Wind-related events
- `flood_event_count`: Flood events
- `winter_event_count`: Winter weather events
- `hail_event_count`: Hail events
- `tornado_event_count`: Tornado events
- `lightning_event_count`: Lightning events
- `deaths`: Fatalities
- `injuries`: Injuries
- `avg_magnitude`: Average event magnitude

#### **How It Works in Your Project**:
1. **Catalog Fetch**: Scans NOAA directory for latest files
2. **Download**: Downloads compressed CSV files by year
3. **Parsing**: Extracts damage, event types, locations
4. **Aggregation**: Aggregates by state and day
5. **Caching**: Files cached in `data/raw/real/storm_events/`

#### **File Naming**:
```
StormEvents_details-ftp_v1.0_d{year}_c{timestamp}.csv.gz
Example: StormEvents_details-ftp_v1.0_d2024_c20241215.csv.gz
```

#### **Current Years Used**:
- 2021, 2022, 2023, 2024 (4 years of data)

#### **Event Types Tracked**:
- Wind (Thunderstorm Wind, High Wind, etc.)
- Flood (Flash Flood, Flood, etc.)
- Winter (Winter Storm, Blizzard, etc.)
- Hail
- Tornado
- Lightning
- And more...

#### **Damage Parsing**:
- Handles formats like: "1.5K", "2.3M", "500"
- Converts to absolute USD values
- Used as `failures` target variable

#### **Code Location**:
- `src/data/download_and_preprocess.py`
- Functions: `_fetch_storm_events_catalog()`, `_download_storm_events_files()`, `_load_storm_events_dataset()`, `_parse_damage_value()`

---

## 🔄 Data Flow

### **Complete Pipeline**:

```
1. User runs: python -m src.models.train_ensemble --mode real

2. Data Loading:
   ├─ Check cache: data/raw/real/
   ├─ If not cached:
   │  ├─ Download GHCN files from NOAA
   │  └─ Download Storm Events files from NOAA
   └─ If cached: Use existing files

3. Data Parsing:
   ├─ Parse GHCN .dly files → Weather DataFrame
   └─ Parse Storm Events CSV → Failures DataFrame

4. Data Merging:
   ├─ Create base index (region_id × timestamp)
   ├─ Merge weather data
   ├─ Merge storm events data
   └─ Add coordinates (lat/lon)

5. Feature Engineering:
   ├─ Time series alignment
   ├─ Missing value imputation
   ├─ Lag features
   ├─ Rolling statistics
   └─ Additional data sources (infrastructure, population, economic)

6. Model Training:
   └─ Use processed data for LSTM + Random Forest + XGBoost
```

---

## 📊 Data Statistics

### **Current Dataset**:
- **Total Rows**: ~14,610
- **Regions**: 10 US states
- **Date Range**: 2021-01-01 to 2024-12-31 (4 years)
- **Weather Variables**: 7 (temperature, precipitation, wind, snow)
- **Failure Variables**: 10+ (damage, event counts, etc.)

### **Data Quality**:
- ✅ Real-world data (not synthetic)
- ✅ Daily resolution
- ✅ Multiple weather stations
- ✅ Comprehensive storm event coverage
- ⚠️ Some missing values (handled automatically)

---

## 🎯 How NOAA Data Is Used

### **Weather Data (GHCN)**:
- **Purpose**: Predict infrastructure failures based on weather patterns
- **Usage**: 
  - Direct features (temperature, precipitation, wind)
  - Lagged features (weather from previous days)
  - Rolling statistics (moving averages, max, std)
  - Seasonal patterns

### **Storm Events (Infrastructure Failures)**:
- **Purpose**: Target variable (what we're predicting)
- **Usage**:
  - `failures`: Main target (damage in USD)
  - `event_count`: Total events (alternative target)
  - Event type counts: Feature engineering
  - Magnitude: Feature engineering

---

## 🔧 Configuration

### **Current Config** (`configs/default.yaml`):

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
      TX: USW00013904
      FL: USW00012839
      NY: USW00094728
      IL: USW00094846
      GA: USW00053863
      WA: USW00024233
      AZ: USW00023183
      NC: USW00013723
      CO: USW00023062
```

### **Customizing NOAA Data**:

#### **Add More Stations**:
1. Find station IDs: https://www.ncei.noaa.gov/data/ghcn-daily/
2. Add to config:
```yaml
ghcn:
  stations:
    OR: USW00024229  # Portland, Oregon
    NV: USW00023169  # Las Vegas, Nevada
```

#### **Change Date Range**:
```yaml
real:
  start_date: "2020-01-01"  # Earlier
  end_date: "2023-12-31"    # Earlier
```

#### **Add More Years**:
```yaml
storm_events:
  years: [2020, 2021, 2022, 2023, 2024]  # Add 2020
```

---

## 📚 Additional NOAA Resources

### **NOAA Climate Data API** (Could Add):
- **URL**: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
- **What**: Programmatic access to climate data
- **Status**: API available (not currently used)
- **Use Case**: Alternative to file downloads

### **NOAA Weather API** (Could Add):
- **URL**: https://www.weather.gov/documentation/services-web-api
- **What**: Real-time and forecast weather data
- **Status**: API available (not currently used)
- **Use Case**: Real-time predictions

### **NOAA Integrated Surface Database (ISD)**:
- **URL**: https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
- **What**: Hourly weather observations
- **Status**: Available (not currently used)
- **Use Case**: Higher temporal resolution

---

## ✅ Summary

### **What You Have**:
- ✅ **GHCN Daily Weather Data** - Real weather from 10 stations
- ✅ **NOAA Storm Events** - Real infrastructure damage data
- ✅ **Automatic Downloads** - Cached for efficiency
- ✅ **Production Ready** - Real-world data, not synthetic

### **What You Could Add**:
- 🔄 **NOAA Climate Data API** - Programmatic access
- 🔄 **NOAA Weather API** - Real-time forecasts
- 🔄 **ISD Data** - Hourly resolution
- 🔄 **More Stations** - Expand coverage

### **Current Status**:
- ✅ **Working perfectly**
- ✅ **Real data**
- ✅ **Well-documented**
- ✅ **Ready for production**

---

## 🚀 Next Steps

1. **Continue using current NOAA data** (it's working great!)
2. **Consider adding NOAA APIs** for real-time data
3. **Add more stations** for better coverage
4. **Extend date range** for more historical data

**Your NOAA data integration is solid and production-ready!** 🎉

