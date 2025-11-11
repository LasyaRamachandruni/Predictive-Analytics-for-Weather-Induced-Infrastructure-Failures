# Adding More Data Sources - Comprehensive Project Enhancement

## 🎯 Goal: Make This a Full-Rounded Infrastructure Failure Prediction System

Currently, we have:
- ✅ Weather data (temperature, precipitation, wind)
- ✅ Storm event damage data

To make it comprehensive, we should add:

---

## 📊 Additional Data Sources to Add

### 1. **Power Outage Records** ⚡
**Why:** Direct failure data instead of using storm damage as proxy

**Sources:**
- **EIA (Energy Information Administration)**: Power outage statistics
- **DOE (Department of Energy)**: Grid reliability data
- **Utility company APIs**: Real-time outage data
- **HIFLD (Homeland Infrastructure Foundation-Level Data)**: Power infrastructure

**What to Add:**
- Number of customers affected
- Outage duration
- Cause of outage
- Geographic location

**Implementation:**
```python
# Add to download_and_preprocess.py
def _load_power_outage_data():
    # Download from EIA or utility APIs
    # Parse outage records
    # Merge with weather data
```

---

### 2. **Infrastructure Age & Maintenance Data** 🏗️
**Why:** Older infrastructure fails more often

**Sources:**
- **ASCE (American Society of Civil Engineers)**: Infrastructure report cards
- **FHWA (Federal Highway Administration)**: Bridge and road condition data
- **EPA**: Water infrastructure age data
- **State DOT databases**: Infrastructure maintenance records

**What to Add:**
- Average infrastructure age by region
- Maintenance frequency
- Infrastructure condition scores
- Last maintenance date

**Features:**
- `infrastructure_age`: Average age of infrastructure
- `maintenance_score`: Maintenance quality score
- `condition_rating`: Overall condition (1-10)

---

### 3. **Population Density & Demographics** 👥
**Why:** More people = more infrastructure = more potential failures

**Sources:**
- **US Census Bureau**: Population data
- **Census API**: Real-time demographics
- **WorldPop**: High-resolution population data

**What to Add:**
- Population density
- Urban vs rural classification
- Critical infrastructure density
- Demographics (age, income)

**Features:**
- `population_density`: People per square mile
- `urban_percentage`: % urban population
- `critical_infrastructure_count`: Number of critical facilities

---

### 4. **Economic Indicators** 💰
**Why:** Economic factors affect infrastructure investment and maintenance

**Sources:**
- **Bureau of Economic Analysis**: Regional economic data
- **FRED (Federal Reserve)**: Economic indicators
- **Census Economic Data**: Business statistics

**What to Add:**
- GDP per capita
- Infrastructure investment
- Poverty rates
- Employment rates

**Features:**
- `gdp_per_capita`: Economic strength
- `infrastructure_investment`: Spending on infrastructure
- `poverty_rate`: Economic vulnerability

---

### 5. **Historical Failure Records** 📋
**Why:** Direct historical data improves predictions

**Sources:**
- **FEMA**: Disaster declarations
- **NOAA**: Historical damage reports
- **State emergency management**: Local failure records
- **Utility company historical data**: Past outages

**What to Add:**
- Historical failure counts
- Failure patterns
- Seasonal trends
- Regional failure rates

---

### 6. **Additional Weather Variables** 🌦️
**Why:** More weather context = better predictions

**Sources:**
- **NOAA**: Extended weather data
- **NASA**: Satellite data
- **Weather APIs**: Real-time extended forecasts

**What to Add:**
- Humidity
- Barometric pressure
- UV index
- Heat index
- Wind direction
- Visibility
- Cloud cover

---

### 7. **Geographic & Environmental Factors** 🗺️
**Why:** Geography affects infrastructure vulnerability

**Sources:**
- **USGS**: Geological data
- **FEMA**: Flood zones
- **NOAA**: Coastal data

**What to Add:**
- Elevation
- Distance to coast
- Flood zone classification
- Soil type
- Seismic activity
- Wildfire risk

---

## 🛠️ Implementation Plan

### Phase 1: Power Outage Data (Highest Priority)

**Step 1: Create Data Loader**
```python
# src/data/additional_sources.py

def load_power_outage_data(start_date, end_date, regions):
    """
    Load power outage data from EIA or utility APIs.
    
    Returns DataFrame with:
    - region_id
    - timestamp
    - customers_affected
    - outage_duration_hours
    - cause
    """
    pass
```

**Step 2: Integrate with Pipeline**
```python
# In download_and_preprocess.py
def _load_real_dataset():
    # Existing weather + storm data
    weather_df, coords = _load_ghcn_weather(...)
    storm_df = _load_storm_events_dataset(...)
    
    # NEW: Power outage data
    outage_df = load_power_outage_data(...)
    
    # Merge all together
    merged = combine_all_data(weather_df, storm_df, outage_df, ...)
```

**Step 3: Update Features**
```python
# In data_pipeline.py
# Add outage-related features:
# - outage_history_lag_1, lag_3, etc.
# - average_outage_duration
# - outage_frequency
```

---

### Phase 2: Infrastructure Age Data

**Step 1: Create Loader**
```python
def load_infrastructure_age_data(regions):
    """
    Load infrastructure age and condition data.
    
    Returns DataFrame with:
    - region_id
    - avg_infrastructure_age
    - maintenance_score
    - condition_rating
    """
    # Could use ASCE report card data
    # Or scrape state DOT websites
    pass
```

**Step 2: Merge as Static Features**
```python
# These don't change daily, so merge once per region
infrastructure_df = load_infrastructure_age_data(regions)
merged = weather_df.merge(infrastructure_df, on='region_id', how='left')
```

---

### Phase 3: Population & Economic Data

**Step 1: Create Loaders**
```python
def load_population_data(regions):
    """Load population density and demographics."""
    pass

def load_economic_data(regions):
    """Load economic indicators."""
    pass
```

**Step 2: Merge as Static Features**
```python
# Merge once per region (static data)
pop_df = load_population_data(regions)
econ_df = load_economic_data(regions)
merged = merged.merge(pop_df, on='region_id')
merged = merged.merge(econ_df, on='region_id')
```

---

## 📋 Data Source Details

### Power Outage Data Sources

1. **EIA Form 861**
   - Annual utility statistics
   - Outage duration and frequency
   - Customer counts
   - URL: https://www.eia.gov/electricity/data/eia861/

2. **DOE Grid Modernization**
   - Grid reliability metrics
   - Outage causes
   - Recovery times
   - URL: https://www.energy.gov/oe/services/electricity-policy-coordination-and-implementation

3. **Utility Company APIs**
   - Real-time outage data
   - Historical records
   - Geographic precision

### Infrastructure Age Sources

1. **ASCE Infrastructure Report Card**
   - Grades by category and state
   - Condition assessments
   - Investment needs
   - URL: https://infrastructurereportcard.org/

2. **FHWA Bridge Data**
   - Bridge condition ratings
   - Age and maintenance
   - URL: https://www.fhwa.dot.gov/bridge/nbi.cfm

### Population Data Sources

1. **US Census API**
   - Population by county/state
   - Demographics
   - Real-time updates
   - URL: https://www.census.gov/data/developers/data-sets.html

2. **Census Bureau QuickFacts**
   - Easy-to-access summaries
   - By state/county

### Economic Data Sources

1. **Bureau of Economic Analysis**
   - GDP by state
   - Economic indicators
   - URL: https://www.bea.gov/data

2. **FRED (Federal Reserve)**
   - Economic time series
   - Regional data
   - URL: https://fred.stlouisfed.org/

---

## 🔧 Implementation Example

### Adding Power Outage Data

```python
# src/data/additional_sources.py

import pandas as pd
from pathlib import Path
from typing import Dict, List

def load_eia_outage_data(years: List[int], cache_dir: Path) -> pd.DataFrame:
    """
    Load power outage data from EIA Form 861.
    
    This is a placeholder - you would implement actual EIA API calls
    or file downloads here.
    """
    # TODO: Implement EIA data download
    # For now, return empty DataFrame
    return pd.DataFrame(columns=['region_id', 'timestamp', 'customers_affected', 'outage_duration'])

def load_infrastructure_age_data(regions: List[str]) -> pd.DataFrame:
    """
    Load infrastructure age and condition data.
    
    Could use:
    - ASCE report card data (scraped or API)
    - State DOT databases
    - FEMA infrastructure assessments
    """
    # Example structure:
    data = []
    for region in regions:
        data.append({
            'region_id': region,
            'avg_infrastructure_age': 25.5,  # Would come from real data
            'maintenance_score': 6.5,
            'condition_rating': 7.0,
        })
    return pd.DataFrame(data)

def load_population_data(regions: List[str]) -> pd.DataFrame:
    """Load population density data from Census."""
    # Would use Census API
    data = []
    for region in regions:
        data.append({
            'region_id': region,
            'population_density': 150.0,  # People per sq mile
            'urban_percentage': 85.0,
            'total_population': 1000000,
        })
    return pd.DataFrame(data)
```

### Integrating into Pipeline

```python
# In download_and_preprocess.py

def _load_real_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    # ... existing code ...
    
    # NEW: Load additional data sources
    from .additional_sources import (
        load_infrastructure_age_data,
        load_population_data,
        load_eia_outage_data
    )
    
    # Load static features (don't change daily)
    infrastructure_df = load_infrastructure_age_data(region_ids)
    population_df = load_population_data(region_ids)
    
    # Load time-series data
    outage_df = load_eia_outage_data(storm_years, cache_dir)
    
    # Merge everything
    merged = _combine_real_dataset(
        weather_df, coords, storm_df, region_ids, start_date, end_date
    )
    
    # Add static features
    merged = merged.merge(infrastructure_df, on='region_id', how='left')
    merged = merged.merge(population_df, on='region_id', how='left')
    
    # Add outage data
    if not outage_df.empty:
        merged = merged.merge(
            outage_df,
            on=['region_id', 'timestamp'],
            how='left'
        )
    
    return merged
```

---

## 📊 Enhanced Feature Set

After adding more data, you'll have:

### Weather Features (Existing)
- Temperature, precipitation, wind, snow

### Infrastructure Features (New)
- `avg_infrastructure_age`
- `maintenance_score`
- `condition_rating`
- `last_maintenance_date`

### Population Features (New)
- `population_density`
- `urban_percentage`
- `critical_infrastructure_count`

### Economic Features (New)
- `gdp_per_capita`
- `infrastructure_investment`
- `poverty_rate`

### Outage Features (New)
- `customers_affected`
- `outage_duration_hours`
- `outage_frequency`

### Geographic Features (New)
- `elevation`
- `distance_to_coast`
- `flood_zone`
- `seismic_risk`

---

## 🎯 Priority Order

### High Priority (Add First)
1. ✅ **Power Outage Data** - Direct failure records
2. ✅ **Infrastructure Age** - Critical vulnerability factor
3. ✅ **Population Density** - Affects failure impact

### Medium Priority
4. **Economic Indicators** - Affects maintenance quality
5. **Additional Weather Variables** - More context

### Lower Priority
6. **Geographic Factors** - Nice to have
7. **Historical Failure Records** - Redundant with outages

---

## 🚀 Quick Start: Adding One Data Source

Let's start with **Infrastructure Age Data** (easiest to add):

### Step 1: Create the loader
```python
# src/data/additional_sources.py
def load_infrastructure_age_simple(regions: List[str]) -> pd.DataFrame:
    """Simple infrastructure age data (can be enhanced later)."""
    # For now, use estimated values based on state averages
    age_data = {
        'CA': {'age': 30, 'score': 6.5},
        'TX': {'age': 25, 'score': 7.0},
        'FL': {'age': 28, 'score': 6.8},
        # ... etc
    }
    
    data = []
    for region in regions:
        info = age_data.get(region, {'age': 27, 'score': 6.5})
        data.append({
            'region_id': region,
            'avg_infrastructure_age': info['age'],
            'maintenance_score': info['score'],
        })
    return pd.DataFrame(data)
```

### Step 2: Integrate
```python
# In _load_real_dataset or _combine_real_dataset
infrastructure_df = load_infrastructure_age_simple(region_ids)
merged = merged.merge(infrastructure_df, on='region_id', how='left')
```

### Step 3: Use in Features
The feature engineering will automatically pick up these new columns!

---

## 📚 Data Source URLs

### Power Outages
- EIA: https://www.eia.gov/electricity/data/eia861/
- DOE: https://www.energy.gov/oe/services/electricity-policy-coordination-and-implementation

### Infrastructure
- ASCE Report Card: https://infrastructurereportcard.org/
- FHWA Bridges: https://www.fhwa.dot.gov/bridge/nbi.cfm

### Population
- Census API: https://www.census.gov/data/developers/data-sets.html
- QuickFacts: https://www.census.gov/quickfacts/

### Economic
- BEA: https://www.bea.gov/data
- FRED: https://fred.stlouisfed.org/

---

## 🎯 Next Steps

1. **Choose a data source** to add first (recommend: Infrastructure Age)
2. **Create loader function** in `src/data/additional_sources.py`
3. **Integrate into pipeline** in `download_and_preprocess.py`
4. **Test** with existing pipeline
5. **Add more sources** incrementally

This will make your project much more comprehensive and realistic! 🚀

