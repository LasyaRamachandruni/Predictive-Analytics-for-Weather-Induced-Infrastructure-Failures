# Real API Sources - Verified and Current (2024)

## 📋 Overview

This document provides verified, current information about real APIs you can use to replace placeholder data in your project. All APIs listed here are **free** and **government-provided**.

---

## 1. US Census Bureau API (Population Data)

### ✅ **Status**: Active, Free, No Rate Limits

### **Official URLs**:
- **API Documentation**: https://www.census.gov/data/developers/data-sets.html
- **API Base URL**: `https://api.census.gov/data/`
- **Key Signup**: https://api.census.gov/data/key_signup.html
- **API Examples**: https://api.census.gov/data/2023/acs/acs5/examples.html

### **What You Can Get**:
- Population by state/county
- Population density
- Demographics (age, race, income)
- Urban/rural classification
- Housing data

### **Example Endpoints**:

**ACS 5-Year Estimates (Most Recent)**:
```
https://api.census.gov/data/2023/acs/acs5
?get=B01001_001E          # Total population
&for=state:06             # California (FIPS code)
&key=YOUR_API_KEY
```

**State Population**:
```
https://api.census.gov/data/2023/acs/acs5
?get=B01001_001E          # Total population
&for=state:*              # All states
&key=YOUR_API_KEY
```

**Urban/Rural Data**:
```
https://api.census.gov/data/2023/acs/acs5
?get=B08301_001E,B08301_021E  # Total workers, Worked from home
&for=state:06
&key=YOUR_API_KEY
```

### **FIPS State Codes**:
- CA: 06, TX: 48, FL: 12, NY: 36, IL: 17, GA: 13, WA: 53, AZ: 04, NC: 37, CO: 08

### **API Key**: Free, unlimited requests

### **Data Format**: JSON

---

## 2. Bureau of Economic Analysis (BEA) API

### ✅ **Status**: Active, Free, No Rate Limits

### **Official URLs**:
- **API Documentation**: https://apps.bea.gov/API/bea_web_service_api_user_guide.htm
- **API Base URL**: `https://apps.bea.gov/api/data`
- **Key Signup**: https://apps.bea.gov/API/signup/
- **API Examples**: https://apps.bea.gov/API/bea_web_service_api_user_guide.htm#Section2

### **What You Can Get**:
- GDP by state
- Personal income by state
- GDP per capita
- Economic indicators
- Industry data

### **Example Endpoints**:

**State GDP (Annual)**:
```
https://apps.bea.gov/api/data
?UserID=YOUR_API_KEY
&method=GetData
&datasetname=Regional
&TableName=SAGDP
&LineCode=1
&GeoFips=STATE
&Year=2023
&ResultFormat=JSON
```

**State Personal Income**:
```
https://apps.bea.gov/api/data
?UserID=YOUR_API_KEY
&method=GetData
&datasetname=Regional
&TableName=SAINC1
&LineCode=1
&GeoFips=STATE
&Year=2023
&ResultFormat=JSON
```

**GDP Per Capita** (Calculate from GDP + Population):
- Get GDP from BEA
- Get Population from Census
- Divide: GDP / Population

### **API Key**: Free, unlimited requests

### **Data Format**: JSON, XML

---

## 3. FRED (Federal Reserve Economic Data) API

### ✅ **Status**: Active, Free, 120 requests/minute

### **Official URLs**:
- **API Documentation**: https://fred.stlouisfed.org/docs/api/fred/
- **API Base URL**: `https://api.stlouisfed.org/fred/`
- **Key Signup**: https://fred.stlouisfed.org/docs/api/api_key.html
- **API Examples**: https://fred.stlouisfed.org/docs/api/fred/series_observations.html

### **What You Can Get**:
- GDP per capita by state
- Personal income
- Employment data
- Poverty rates (from Census, via FRED)
- Economic time series

### **Example Endpoints**:

**State Personal Income Per Capita**:
```
https://api.stlouisfed.org/fred/series/observations
?series_id=CASTHPI        # California State Total Personal Income
&api_key=YOUR_API_KEY
&file_type=json
&limit=1
&sort_order=desc
```

**State GDP**:
```
https://api.stlouisfed.org/fred/series/observations
?series_id=CASTNGSP       # California State GDP
&api_key=YOUR_API_KEY
&file_type=json
```

**Find Series IDs**:
- Search: https://fred.stlouisfed.org/
- Use series search to find state-specific data

### **Common FRED Series IDs**:
- State GDP: `[STATE]STNGSP` (e.g., `CASTNGSP` for California)
- State Personal Income: `[STATE]STHPI` (e.g., `CASTHPI`)
- State Population: `[STATE]POP` (e.g., `CAPOP`)

### **API Key**: Free, 120 requests/minute

### **Data Format**: JSON, XML

---

## 4. FHWA National Bridge Inventory (Infrastructure Condition)

### ⚠️ **Status**: No API - File Download Only

### **Official URLs**:
- **Data Portal**: https://www.fhwa.dot.gov/bridge/nbi.cfm
- **Download Page**: https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm
- **Documentation**: https://www.fhwa.dot.gov/bridge/nbi/format.cfm

### **What You Can Get**:
- Bridge condition ratings
- Bridge age
- Maintenance records
- Structural data

### **How to Access**:
1. Download CSV files from FHWA website
2. Parse files (fixed-width format)
3. Aggregate by state
4. Calculate average condition ratings

### **File Format**: Fixed-width ASCII files

### **Update Frequency**: Annual

### **Alternative**: Use Data.gov API if available

---

## 5. Data.gov APIs (Infrastructure Data)

### ✅ **Status**: Active, Free, Various Rate Limits

### **Official URLs**:
- **Data.gov**: https://www.data.gov/
- **API Catalog**: https://www.data.gov/developers/apis
- **Transportation Data**: https://www.data.gov/transportation/

### **What You Can Get**:
- Bridge condition data
- Road condition data
- Infrastructure projects
- Transportation statistics

### **Example Datasets**:
- National Bridge Inventory (via Data.gov)
- Highway Performance Monitoring System
- Infrastructure Investment data

### **API Access**: Varies by dataset (some have APIs, some are file downloads)

---

## 6. EIA (Energy Information Administration) - Power Outage Data

### ⚠️ **Status**: No API - File Download Only

### **Official URLs**:
- **EIA Form 861**: https://www.eia.gov/electricity/data/eia861/
- **Annual Data**: https://www.eia.gov/electricity/data/eia861/
- **Documentation**: https://www.eia.gov/electricity/data/eia861/

### **What You Can Get**:
- Utility statistics
- Outage duration
- Customer counts
- Reliability metrics

### **How to Access**:
1. Download annual CSV files
2. Parse utility statistics
3. Aggregate by state
4. Calculate outage metrics

### **File Format**: CSV, Excel

### **Update Frequency**: Annual

---

## 7. NOAA Datasets (Already Using - Weather Data) ✅

### ✅ **Status**: Active, Free, **ALREADY INTEGRATED**

### **What You're Already Using**:

#### **1. NOAA GHCN Daily Weather Data** ✅
- **Source**: `https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/`
- **Status**: ✅ **Working - File Downloads**
- **What**: Temperature, precipitation, wind, snow from 10 US stations
- **Coverage**: 2021-2024 (4 years)
- **See**: `docs/NOAA_DATASETS_DOCUMENTED.md` for complete details

#### **2. NOAA Storm Events Database** ✅
- **Source**: `https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/`
- **Status**: ✅ **Working - File Downloads**
- **What**: Infrastructure damage, event types, locations
- **Coverage**: 2021-2024 (4 years)
- **See**: `docs/NOAA_DATASETS_DOCUMENTED.md` for complete details

### **Additional NOAA APIs** (Could Add):
- **NOAA Climate Data API**: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
  - **Use Case**: Programmatic access instead of file downloads
  - **Status**: Available, not currently used
  
- **NOAA Weather API**: https://www.weather.gov/documentation/services-web-api
  - **Use Case**: Real-time weather forecasts
  - **Status**: Available, not currently used

- **NOAA Integrated Surface Database (ISD)**: 
  - **URL**: https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
  - **Use Case**: Hourly weather observations (higher resolution)
  - **Status**: Available, not currently used

---

## 8. Additional Government Data Sources

### **USGS (Geological Survey)**
- **URL**: https://www.usgs.gov/products/data-and-tools/apis
- **What**: Geological, water, infrastructure data
- **Status**: Some APIs available

### **FEMA Data**
- **URL**: https://www.fema.gov/about/openfema/data-feeds
- **What**: Disaster declarations, infrastructure assessments
- **Status**: API available

### **DOT (Department of Transportation)**
- **URL**: https://www.transportation.gov/data
- **What**: Transportation infrastructure data
- **Status**: Various APIs and datasets

---

## 🎯 Recommended Implementation Priority

### **High Priority** (Easy to Implement):
1. ✅ **Census API** - Population data (has API, well-documented)
2. ✅ **BEA API** - Economic data (has API, well-documented)
3. ✅ **FRED API** - Economic time series (has API, well-documented)

### **Medium Priority** (Requires File Processing):
4. ⚠️ **FHWA Bridge Data** - Download and parse files
5. ⚠️ **EIA Power Outage Data** - Download and parse files

### **Lower Priority** (Research Needed):
6. 🔍 **Data.gov APIs** - Explore available datasets
7. 🔍 **FEMA APIs** - Infrastructure assessments
8. 🔍 **USGS APIs** - Additional infrastructure data

---

## 📝 Implementation Notes

### **Census API**:
- Most straightforward to implement
- Well-documented
- Free, unlimited requests
- **Start here!**

### **BEA API**:
- Good for GDP and economic data
- Free, unlimited requests
- Response format may need parsing

### **FRED API**:
- Excellent for time series data
- Free, 120 requests/minute
- Need to find correct series IDs

### **FHWA/EIA**:
- No APIs - file downloads only
- Requires file parsing
- More complex but valuable data

---

## 🔗 Quick Reference Links

### **API Signup Pages**:
- Census: https://api.census.gov/data/key_signup.html
- BEA: https://apps.bea.gov/API/signup/
- FRED: https://fred.stlouisfed.org/docs/api/api_key.html

### **API Documentation**:
- Census: https://www.census.gov/data/developers/data-sets.html
- BEA: https://apps.bea.gov/API/bea_web_service_api_user_guide.htm
- FRED: https://fred.stlouisfed.org/docs/api/fred/

### **Data Download Pages**:
- FHWA: https://www.fhwa.dot.gov/bridge/nbi.cfm
- EIA: https://www.eia.gov/electricity/data/eia861/

---

## ✅ Summary

**Best APIs to Start With**:
1. **Census API** - Population data (easiest)
2. **BEA API** - Economic data (straightforward)
3. **FRED API** - Economic time series (powerful)

**All are free, well-documented, and ready to use!**

See `docs/API_INTEGRATION_GUIDE.md` for step-by-step implementation guide.

