# Real API Integration Guide

## 🎯 Overview

This guide shows you how to replace the **placeholder/fabricated data** with **real data from free APIs**.

---

## 📋 Current Status

### What's Real ✅
- Weather data (NOAA GHCN) - **Already working!**
- Storm events (NOAA) - **Already working!**

### What's Fabricated ❌
- Infrastructure age data - **Placeholder values**
- Population data - **Hardcoded (based on real estimates, but not from API)**
- Economic data - **Placeholder values**

---

## 🚀 Quick Start: Replace with Real APIs

### Step 1: Get Free API Keys (5 minutes)

1. **Census API** (Population Data)
   - Go to: https://api.census.gov/data/key_signup.html
   - Sign up (free)
   - Copy your API key

2. **BEA API** (Economic Data)
   - Go to: https://apps.bea.gov/API/signup/
   - Sign up (free)
   - Copy your API key

3. **FRED API** (Economic Data - Alternative)
   - Go to: https://fred.stlouisfed.org/docs/api/api_key.html
   - Sign up (free)
   - Copy your API key

### Step 2: Set Environment Variables

**Windows (PowerShell)**:
```powershell
$env:CENSUS_API_KEY="your_census_key_here"
$env:BEA_API_KEY="your_bea_key_here"
$env:FRED_API_KEY="your_fred_key_here"
```

**Windows (Command Prompt)**:
```cmd
set CENSUS_API_KEY=your_census_key_here
set BEA_API_KEY=your_bea_key_here
set FRED_API_KEY=your_fred_key_here
```

**Linux/Mac**:
```bash
export CENSUS_API_KEY="your_census_key_here"
export BEA_API_KEY="your_bea_key_here"
export FRED_API_KEY="your_fred_key_here"
```

**Or create a `.env` file** (recommended):
```bash
# .env file
CENSUS_API_KEY=your_census_key_here
BEA_API_KEY=your_bea_key_here
FRED_API_KEY=your_fred_key_here
```

Then install `python-dotenv`:
```bash
pip install python-dotenv
```

And load in your code:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Step 3: Update Code to Use Real APIs

I've created `src/data/api_loaders.py` with example implementations. Update `src/data/additional_sources.py` to use them:

```python
# In src/data/additional_sources.py

from .api_loaders import load_population_data_census, load_economic_data_bea

def load_population_data(regions: List[str]) -> pd.DataFrame:
    """Load population data from Census API."""
    # Try API first
    df = load_population_data_census(regions)
    if not df.empty:
        return df
    
    # Fall back to placeholder if API fails
    logger.warning("Using placeholder population data")
    return load_population_data_placeholder(regions)
```

---

## 📡 Available APIs

### 1. US Census Bureau API

**What it provides:**
- Population by state/county
- Demographics
- Economic data
- Housing data

**API Documentation:**
- https://www.census.gov/data/developers/data-sets.html
- https://api.census.gov/data.html

**Example Request:**
```python
import requests

url = "https://api.census.gov/data/2023/acs/acs5"
params = {
    "get": "B01001_001E",  # Total population
    "for": "state:06",     # California (FIPS code)
    "key": "YOUR_API_KEY"
}
response = requests.get(url, params=params)
data = response.json()
```

**Free Tier:** Unlimited requests (with rate limits)

---

### 2. Bureau of Economic Analysis (BEA) API

**What it provides:**
- GDP by state
- Personal income
- Economic indicators
- Industry data

**API Documentation:**
- https://www.bea.gov/API/
- https://apps.bea.gov/API/signup/

**Example Request:**
```python
url = "https://apps.bea.gov/api/data"
params = {
    "UserID": "YOUR_API_KEY",
    "method": "GetData",
    "datasetname": "Regional",
    "TableName": "SQGDP",
    "LineCode": "1",
    "Year": "2023",
    "ResultFormat": "JSON"
}
response = requests.get(url, params=params)
```

**Free Tier:** Unlimited requests

---

### 3. FRED (Federal Reserve Economic Data) API

**What it provides:**
- GDP per capita
- Employment data
- Economic time series
- State-level indicators

**API Documentation:**
- https://fred.stlouisfed.org/docs/api/fred/
- https://fred.stlouisfed.org/docs/api/api_key.html

**Example Request:**
```python
url = "https://api.stlouisfed.org/fred/series/observations"
params = {
    "series_id": "CASTHPI",  # California State Total Personal Income
    "api_key": "YOUR_API_KEY",
    "file_type": "json"
}
response = requests.get(url, params=params)
```

**Free Tier:** 120 requests per minute

---

### 4. FHWA Bridge Data (No API - File Download)

**What it provides:**
- Bridge condition ratings
- Infrastructure age
- Maintenance records

**Source:**
- https://www.fhwa.dot.gov/bridge/nbi.cfm
- Download National Bridge Inventory files

**Implementation:**
- Download CSV files
- Parse and aggregate by state
- Use as static features

---

### 5. Power Outage Data

**Option A: EIA (Energy Information Administration)**
- Source: https://www.eia.gov/electricity/data/eia861/
- **No API** - Download CSV files
- Annual utility statistics

**Option B: Utility Company APIs**
- Many utilities have public APIs
- Examples:
  - PG&E: https://www.pge.com/pge_global/common/pages/safetyoutage/
  - ConEd: Various APIs available

**Option C: DOE Grid Data**
- Source: https://www.energy.gov/oe/services/electricity-policy-coordination-and-implementation
- Grid reliability metrics

---

## 🛠️ Implementation Steps

### Option 1: Use Provided API Loaders

I've created `src/data/api_loaders.py` with starter code. Update it with your API keys and adjust parsing based on actual API responses.

### Option 2: Manual Integration

1. **Create API loader function:**
```python
def load_population_data_census(regions: List[str]) -> pd.DataFrame:
    api_key = os.getenv("CENSUS_API_KEY")
    # ... implement API calls
    return df
```

2. **Update `additional_sources.py`:**
```python
def load_population_data(regions: List[str]) -> pd.DataFrame:
    # Try API first
    try:
        return load_population_data_census(regions)
    except:
        # Fall back to placeholder
        return load_population_data_placeholder(regions)
```

3. **Test:**
```python
# Set API key
import os
os.environ["CENSUS_API_KEY"] = "your_key"

# Test
from src.data.additional_sources import load_population_data
df = load_population_data(["CA", "TX", "FL"])
print(df)
```

---

## 📊 Benefits of Real API Data

### Current (Placeholder)
- ✅ Concept works
- ✅ Shows feature importance
- ❌ Not accurate
- ❌ Doesn't update

### With Real APIs
- ✅ Accurate, real data
- ✅ Auto-updates when you refresh
- ✅ Better predictions
- ✅ Production-ready
- ✅ Can track changes over time

---

## 🔧 Troubleshooting

### API Key Not Working
- Check environment variable is set: `echo $CENSUS_API_KEY`
- Verify key is correct (no extra spaces)
- Check API key hasn't expired

### Rate Limits
- Add delays between requests
- Cache results to file
- Use batch requests when possible

### API Response Format Changed
- Check API documentation
- Update parsing code
- Add error handling

### No Data Returned
- Check region codes (use FIPS codes for Census)
- Verify year is available
- Check API status page

---

## 🎯 Next Steps

1. **Get API keys** (5 minutes)
2. **Set environment variables**
3. **Test API loaders** (use provided code)
4. **Update `additional_sources.py`** to use real APIs
5. **Retrain model** with real data
6. **Compare results** - should see improved predictions!

---

## 📚 Additional Resources

- **Census API Examples**: https://api.census.gov/data/2023/acs/acs5/examples.html
- **BEA API Guide**: https://apps.bea.gov/API/bea_web_service_api_user_guide.htm
- **FRED API Docs**: https://fred.stlouisfed.org/docs/api/fred/
- **FHWA Bridge Data**: https://www.fhwa.dot.gov/bridge/nbi.cfm

---

**Bottom Line**: Replacing placeholder data with real APIs will make your project **significantly stronger** and **production-ready**! 🚀

