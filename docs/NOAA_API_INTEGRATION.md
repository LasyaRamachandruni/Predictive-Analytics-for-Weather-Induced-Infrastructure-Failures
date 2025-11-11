# NOAA API Integration Guide

## 🎯 Overview

Your project now supports **NOAA APIs** in addition to file downloads! The system automatically tries APIs first, then falls back to file downloads if APIs are unavailable.

---

## ✅ What's Available

### **NOAA APIs You Can Use**:

1. **Climate Data Online (CDO) Web Services API** ✅
   - **URL**: https://www.ncdc.noaa.gov/cdo-web/api/v2
   - **Status**: Available, requires token
   - **What**: GHCN daily weather data
   - **Documentation**: https://www.ncdc.noaa.gov/cdo-web/webservices/v2

2. **NCEI Data Service API** ✅
   - **URL**: https://www.ncei.noaa.gov/access/services/data/v1
   - **Status**: Available, requires token
   - **What**: Environmental data including GHCN
   - **Documentation**: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation

3. **NCEI Search Service API** ✅
   - **URL**: https://www.ncei.noaa.gov/access/services/search/v1
   - **Status**: Available, requires token
   - **What**: Dataset discovery
   - **Documentation**: https://www.ncei.noaa.gov/support/access-search-service-api-user-documentation

### **Note on Storm Events**:
- Storm Events database currently uses **file downloads** (no direct API available)
- File downloads work well and are cached for efficiency

---

## 🚀 Quick Start: Enable NOAA APIs

### Step 1: Get NOAA API Token (5 minutes)

1. **Go to**: https://www.ncdc.noaa.gov/cdo-web/token
2. **Sign up** for a free account (if needed)
3. **Request API token**
4. **Copy your token**

### Step 2: Set Environment Variable

**Windows PowerShell**:
```powershell
$env:NOAA_API_TOKEN="your_token_here"
```

**Windows Command Prompt**:
```cmd
set NOAA_API_TOKEN=your_token_here
```

**Linux/Mac**:
```bash
export NOAA_API_TOKEN="your_token_here"
```

**Or create `.env` file**:
```bash
NOAA_API_TOKEN=your_token_here
```

### Step 3: Test API Connection

```python
from src.data.noaa_api_loaders import test_noaa_api_connection

# Test connection
if test_noaa_api_connection():
    print("✅ NOAA API working!")
else:
    print("❌ NOAA API not configured. Will use file downloads.")
```

### Step 4: Run Your Code!

The system **automatically** uses APIs if token is set:

```bash
python -m src.models.train_ensemble --mode real
```

You'll see in logs:
```
INFO: ✅ Loaded GHCN data via API for station USW00023234 (CA)
```

If API fails, it automatically falls back to file downloads:
```
INFO: API load failed. Falling back to file download.
```

---

## 📊 How It Works

### **Automatic API/File Fallback**:

```
1. Check for NOAA_API_TOKEN environment variable
   ├─ If found: Try API first
   │  ├─ Try CDO API
   │  ├─ If fails: Try NCEI Data Service API
   │  └─ If both fail: Fall back to file download
   └─ If not found: Use file download (current behavior)
```

### **Benefits of Using APIs**:

- ✅ **Faster**: No need to download large files
- ✅ **Selective**: Get only the data you need
- ✅ **Real-time**: Access latest data immediately
- ✅ **Efficient**: Less storage needed

### **Benefits of File Downloads** (Fallback):

- ✅ **Reliable**: Always works, no API limits
- ✅ **Complete**: Get all data at once
- ✅ **Cached**: Fast after first download
- ✅ **No token needed**: Works out of the box

---

## 🔧 Configuration

### **Enable/Disable API Usage**:

The code automatically tries APIs if token is available. To force file downloads only:

```python
# In src/data/download_and_preprocess.py
weather_df, coords = _load_ghcn_weather(
    stations_cfg, 
    start_date, 
    end_date, 
    ghcn_cache,
    use_api=False  # Force file downloads
)
```

### **API Rate Limits**:

- **CDO API**: 5 requests per second, 1000 requests per day (free tier)
- **NCEI APIs**: Varies by endpoint
- **Solution**: Code includes automatic retry logic and caching

---

## 📚 API Documentation Links

### **CDO Web Services API**:
- **Base URL**: https://www.ncdc.noaa.gov/cdo-web/api/v2
- **Documentation**: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
- **Token Signup**: https://www.ncdc.noaa.gov/cdo-web/token
- **Examples**: https://www.ncdc.noaa.gov/cdo-web/webservices/v2#gettingStarted

### **NCEI Data Service API**:
- **Base URL**: https://www.ncei.noaa.gov/access/services/data/v1
- **Documentation**: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
- **Token**: Same as CDO token

### **NCEI Search Service API**:
- **Base URL**: https://www.ncei.noaa.gov/access/services/search/v1
- **Documentation**: https://www.ncei.noaa.gov/support/access-search-service-api-user-documentation

---

## 🎯 Example API Requests

### **CDO API - Get GHCN Daily Data**:

```python
import requests

token = "your_token"
url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
params = {
    "datasetid": "GHCND",
    "stationid": "GHCND:USW00023234",
    "startdate": "2021-01-01",
    "enddate": "2021-12-31",
    "datatypeid": "TMAX,TMIN,PRCP",
    "limit": 1000,
}
headers = {"token": token}

response = requests.get(url, params=params, headers=headers)
data = response.json()
```

### **NCEI Data Service API**:

```python
url = "https://www.ncei.noaa.gov/access/services/data/v1/data"
params = {
    "dataset": "daily-summaries",
    "stations": "USW00023234",
    "startDate": "2021-01-01",
    "endDate": "2021-12-31",
    "dataTypes": "TMAX,TMIN,PRCP",
    "format": "json",
    "token": token,
}

response = requests.get(url, params=params)
data = response.json()
```

---

## ✅ Current Status

### **What's Working**:
- ✅ **File Downloads**: Working perfectly (current default)
- ✅ **API Integration Code**: Ready to use
- ✅ **Automatic Fallback**: Implemented
- ✅ **Error Handling**: Robust

### **What You Need**:
- 🔑 **NOAA API Token**: Get one at https://www.ncdc.noaa.gov/cdo-web/token
- 📝 **Set Environment Variable**: `NOAA_API_TOKEN`

### **What Happens**:
1. **With Token**: Tries API first → Falls back to files if needed
2. **Without Token**: Uses file downloads (current behavior)

---

## 🐛 Troubleshooting

### **"No NOAA API token found"**
- **Solution**: Get token at https://www.ncdc.noaa.gov/cdo-web/token
- **Set**: `export NOAA_API_TOKEN="your_token"`

### **"API request failed"**
- **Check**: Token is valid
- **Check**: Internet connection
- **Fallback**: System automatically uses file downloads

### **"Rate limit exceeded"**
- **Solution**: Code includes retry logic
- **Alternative**: Use file downloads (no rate limits)

### **"API returns empty data"**
- **Check**: Date range is valid
- **Check**: Station ID is correct
- **Fallback**: System automatically uses file downloads

---

## 📊 Comparison: API vs File Downloads

| Aspect | API | File Downloads |
|--------|-----|----------------|
| **Speed** | Fast (selective) | Slower (full files) |
| **Storage** | Minimal | More (cached files) |
| **Token Required** | Yes | No |
| **Rate Limits** | Yes (5/sec) | No |
| **Reliability** | Good | Excellent |
| **Real-time** | Yes | No (cached) |

**Recommendation**: Use APIs for development/testing, file downloads for production.

---

## 🚀 Next Steps

1. **Get API Token**: https://www.ncdc.noaa.gov/cdo-web/token
2. **Set Environment Variable**: `NOAA_API_TOKEN`
3. **Test Connection**: Use `test_noaa_api_connection()`
4. **Run Training**: APIs will be used automatically!

---

## 📚 Additional Resources

- **CDO API Documentation**: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
- **NCEI API Documentation**: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
- **NOAA Data Portal**: https://www.ncei.noaa.gov/
- **API Examples**: https://www.ncdc.noaa.gov/cdo-web/webservices/v2#examples

---

**Your project now supports NOAA APIs with automatic fallback to file downloads!** 🎉

