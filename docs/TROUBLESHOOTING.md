# Troubleshooting Guide

## Common Errors and Solutions

### 1. Dashboard Error: `app.run_server has been replaced by app.run`

**Error:**
```
dash.exceptions.ObsoleteAttributeException: app.run_server has been replaced by app.run
```

**Solution:**
✅ **Fixed!** The dashboard now uses `app.run()` instead of `app.run_server()`.

If you still see this error, make sure you have the latest version of the code.

---

### 2. Map Visualization: "No prediction records found for split 'test'"

**Error:**
```
ValueError: No prediction records found for split 'test'.
```

**Cause:**
- The predictions file might only have "train" split data
- Quick-run mode might not generate test data
- The split you're requesting doesn't exist

**Solution:**
✅ **Fixed!** The default is now `--split all` which uses all available data.

**Options:**
```bash
# Use all available data (recommended)
python -m src.visualization.map_failures --artifacts models/latest/metrics.json --split all

# Use specific split (if available)
python -m src.visualization.map_failures --artifacts models/latest/metrics.json --split train

# Check what splits are available
python -c "import pandas as pd; df = pd.read_csv('models/latest/predictions.csv'); print(df['split'].value_counts() if 'split' in df.columns else 'No split column')"
```

---

### 3. Module Not Found Errors

**Error:**
```
ModuleNotFoundError: No module named 'dash'
```

**Solution:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific package
pip install dash dash-bootstrap-components plotly
```

---

### 4. "This site can't be reached" - ERR_ADDRESS_INVALID

**Error:**
```
This site can't be reached
The webpage at http://0.0.0.0:8050/ might be temporarily down
Error code: ERR_ADDRESS_INVALID
```

**Solution:**
✅ **Use `localhost` or `127.0.0.1` in your browser, NOT `0.0.0.0`**

- ❌ **Wrong**: `http://0.0.0.0:8050` (this won't work in browser)
- ✅ **Correct**: `http://localhost:8050`
- ✅ **Correct**: `http://127.0.0.1:8050`

**Why:** `0.0.0.0` is a server bind address (tells server to listen on all interfaces), but browsers need `localhost` or `127.0.0.1` to connect.

---

### 5. Port Already in Use (Dashboard)

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```python
# Option 1: Change port in src/dashboard/app.py
app.run(debug=True, host="127.0.0.1", port=8080)  # Use different port

# Option 2: Kill process using port 8050
# Windows:
netstat -ano | findstr :8050
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8050 | xargs kill
```

---

### 5. No Data Available (Dashboard)

**Error:**
Dashboard shows "No data available"

**Solution:**
```bash
# Train a model first
python -m src.models.train_ensemble --mode demo

# Then start dashboard
python -m src.dashboard.app
```

---

### 6. Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
# or
MemoryError
```

**Solution:**
```bash
# Use quick-run mode (smaller dataset)
python -m src.models.train_ensemble --mode demo --quick-run

# Or use CPU instead of GPU
# Edit configs/default.yaml:
training:
  device: cpu
```

---

### 7. Data Download Failed (Real Mode)

**Error:**
```
RuntimeError: Failed to download...
```

**Solution:**
1. Check internet connection
2. Verify NOAA websites are accessible
3. Check firewall settings
4. Try again later (servers may be busy)
5. Data is cached after first download, so retry should work

---

### 8. Import Errors

**Error:**
```
ImportError: cannot import name 'X' from 'Y'
```

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or reinstall specific package
pip install --upgrade <package-name>
```

---

### 9. File Not Found

**Error:**
```
FileNotFoundError: models/latest/metrics.json
```

**Solution:**
```bash
# Train a model first to generate files
python -m src.models.train_ensemble --mode demo

# Check if files exist
ls models/latest/
```

---

### 10. Python Version Issues

**Error:**
```
SyntaxError: invalid syntax
# or version-related errors
```

**Solution:**
- Ensure Python 3.8+ is installed
- Check version: `python --version`
- Use virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## Getting Help

### Check Logs
- Look at terminal output for detailed error messages
- Check log files if any are generated

### Verify Installation
```bash
python -c "import torch, pandas, numpy, xgboost, dash; print('✅ All packages OK')"
```

### Test Individual Components
```bash
# Test data loading
python -c "from src.data.download_and_preprocess import load_dataset; df = load_dataset('demo'); print(f'Loaded {len(df)} rows')"

# Test model
python -c "from src.models.lstm_model import LSTMOutagePredictor; print('Model OK')"
```

### Run Tests
```bash
pytest
```

---

## Quick Fixes Checklist

- [ ] Installed all dependencies: `pip install -r requirements.txt`
- [ ] Python version 3.8+: `python --version`
- [ ] Trained a model: `python -m src.models.train_ensemble --mode demo`
- [ ] Checked file paths are correct
- [ ] Verified data files exist (for real mode)
- [ ] Checked terminal output for specific errors
- [ ] Tried quick-run mode for faster testing

---

## Still Having Issues?

1. **Check the error message** - It usually tells you what's wrong
2. **Review terminal output** - Look for warnings or error details
3. **Verify file paths** - Make sure you're in the project root
4. **Check documentation** - See `docs/` folder for detailed guides
5. **Try quick-run** - Use `--quick-run` flag for faster testing

---

Most issues can be resolved by:
1. Installing dependencies: `pip install -r requirements.txt`
2. Training a model first: `python -m src.models.train_ensemble --mode demo`
3. Using correct file paths and commands

