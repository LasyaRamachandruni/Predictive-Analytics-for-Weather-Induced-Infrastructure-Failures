# API Quick Start Guide - Get Real Data in 10 Minutes

## 🚀 Fastest Way to Replace Placeholder Data

### Step 1: Get API Keys (5 minutes)

#### 1.1 Census API Key (Population Data)
1. Go to: https://api.census.gov/data/key_signup.html
2. Fill out the form (name, email, organization)
3. Check your email for the API key
4. Copy the key

#### 1.2 BEA API Key (Economic Data)
1. Go to: https://apps.bea.gov/API/signup/
2. Fill out the form
3. Check your email for the UserID
4. Copy the UserID

#### 1.3 FRED API Key (Economic Time Series)
1. Go to: https://fred.stlouisfed.org/docs/api/api_key.html
2. Create an account or sign in
3. Go to "My Account" → "API Keys"
4. Generate a new key
5. Copy the key

---

### Step 2: Set Environment Variables (2 minutes)

**Windows PowerShell**:
```powershell
$env:CENSUS_API_KEY="your_census_key_here"
$env:BEA_API_KEY="your_bea_userid_here"
$env:FRED_API_KEY="your_fred_key_here"
```

**Windows Command Prompt**:
```cmd
set CENSUS_API_KEY=your_census_key_here
set BEA_API_KEY=your_bea_userid_here
set FRED_API_KEY=your_fred_key_here
```

**Linux/Mac**:
```bash
export CENSUS_API_KEY="your_census_key_here"
export BEA_API_KEY="your_bea_userid_here"
export FRED_API_KEY="your_fred_key_here"
```

**Or create `.env` file** (recommended):
```bash
# Create .env file in project root
CENSUS_API_KEY=your_census_key_here
BEA_API_KEY=your_bea_userid_here
FRED_API_KEY=your_fred_key_here
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

---

### Step 3: Test API Connection (2 minutes)

Create a test script `test_apis.py`:

```python
import os
from dotenv import load_dotenv
from src.data.api_loaders import (
    load_population_data_census,
    load_economic_data_bea,
    load_economic_data_fred
)

load_dotenv()

# Test Census API
print("Testing Census API...")
try:
    df = load_population_data_census(["CA", "TX", "FL"])
    print(f"✅ Census API working! Got {len(df)} states")
    print(df)
except Exception as e:
    print(f"❌ Census API error: {e}")

# Test BEA API
print("\nTesting BEA API...")
try:
    df = load_economic_data_bea(["CA", "TX", "FL"])
    print(f"✅ BEA API working! Got {len(df)} states")
    print(df)
except Exception as e:
    print(f"❌ BEA API error: {e}")

# Test FRED API
print("\nTesting FRED API...")
try:
    df = load_economic_data_fred(["CA", "TX", "FL"])
    print(f"✅ FRED API working! Got {len(df)} states")
    print(df)
except Exception as e:
    print(f"❌ FRED API error: {e}")
```

Run it:
```bash
python test_apis.py
```

---

### Step 4: Use Real Data in Training (1 minute)

The code **automatically** uses APIs if keys are set! Just run:

```bash
python -m src.models.train_ensemble --mode real
```

You'll see in the logs:
```
INFO: Using real Census API data for population
INFO: Using real BEA API data for economic indicators
```

If APIs aren't working, it falls back to placeholders automatically.

---

## 📊 What Each API Provides

### Census API
- ✅ Total population
- ✅ Population density (calculate from pop + area)
- ✅ Urban/rural classification
- ✅ Demographics

### BEA API
- ✅ GDP by state
- ✅ Personal income
- ✅ GDP per capita (calculate)

### FRED API
- ✅ Economic time series
- ✅ State-level indicators
- ✅ Historical data

---

## 🔧 Troubleshooting

### "API key not found"
- Check environment variable is set: `echo $CENSUS_API_KEY`
- Make sure you're in the same terminal session
- Try using `.env` file instead

### "No data returned"
- Check API key is correct (no extra spaces)
- Verify state codes are correct
- Check API status (sometimes APIs are down)

### "Rate limit exceeded"
- FRED has 120 requests/minute limit
- Add delays between requests
- Cache results to file

---

## ✅ Success Indicators

When APIs are working, you'll see:
- ✅ Log messages: "Using real [API] data"
- ✅ Data in your features (check `models/latest/features.json`)
- ✅ More accurate predictions
- ✅ Up-to-date data (not hardcoded)

---

## 📚 Next Steps

1. ✅ Get API keys (done)
2. ✅ Set environment variables (done)
3. ✅ Test APIs (done)
4. ✅ Train with real data
5. ✅ Compare results with placeholder data

**You're ready to use real data!** 🎉

