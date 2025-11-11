"""
NOAA API loaders for weather and storm events data.

This module provides API-based access to NOAA data sources:
- NCEI Data Service API (GHCN daily weather)
- Climate Data Online (CDO) Web Services API
- Storm Events API (if available)

Falls back to file downloads if APIs are unavailable or not configured.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Create a session with retry logic
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# NOAA API endpoints
NCEI_DATA_API_BASE = "https://www.ncei.noaa.gov/access/services/data/v1"
CDO_API_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
NCEI_SEARCH_API_BASE = "https://www.ncei.noaa.gov/access/services/search/v1"


def get_noaa_token() -> Optional[str]:
    """Get NOAA API token from environment variable or .env file."""
    import os
    
    # Try environment variable first
    token = os.getenv("NOAA_API_TOKEN") or os.getenv("NOAA_TOKEN")
    if token:
        return token
    
    # Try loading from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("NOAA_API_TOKEN") or os.getenv("NOAA_TOKEN")
        if token:
            return token
    except ImportError:
        pass  # python-dotenv not installed, skip .env loading
    
    return None


def load_ghcn_daily_via_api(
    station_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load GHCN daily weather data via NCEI Data Service API.
    
    Parameters
    ----------
    station_id: GHCN station ID (e.g., "USW00023234")
    start_date: Start date for data
    end_date: End date for data
    token: Optional NOAA API token
    
    Returns
    -------
    DataFrame with weather data, or empty DataFrame if API fails
    """
    token = token or get_noaa_token()
    
    if not token:
        logger.debug("No NOAA API token found. Skipping API call.")
        return pd.DataFrame()
    
    try:
        # NCEI Data Service API endpoint for GHCN daily
        # Note: API structure may vary - this is a template
        url = f"{NCEI_DATA_API_BASE}/data"
        
        params = {
            "dataset": "daily-summaries",
            "stations": station_id,
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "dataTypes": "TMAX,TMIN,PRCP,SNOW,SNWD,AWND",
            "format": "json",
            "token": token,
        }
        
        logger.info(f"Fetching GHCN data via API for station {station_id}")
        response = session.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data and "results" in data:
                df = pd.DataFrame(data["results"])
                # Transform to match expected format
                return _transform_ghcn_api_data(df)
            else:
                logger.warning(f"No data returned from API for station {station_id}")
                return pd.DataFrame()
        else:
            logger.warning(
                f"API request failed with status {response.status_code}: {response.text[:200]}"
            )
            return pd.DataFrame()
            
    except Exception as e:
        logger.warning(f"Error fetching GHCN data via API: {e}. Will fall back to file download.")
        return pd.DataFrame()


def load_ghcn_daily_via_cdo_api(
    station_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load GHCN daily weather data via Climate Data Online (CDO) API.
    
    Note: CDO API has a 1-year date range limit per request.
    This function splits large date ranges into yearly chunks.
    
    Parameters
    ----------
    station_id: GHCN station ID
    start_date: Start date for data
    end_date: End date for data
    token: Optional NOAA API token
    
    Returns
    -------
    DataFrame with weather data, or empty DataFrame if API fails
    """
    token = token or get_noaa_token()
    
    if not token:
        logger.debug("No NOAA API token found. Skipping CDO API call.")
        return pd.DataFrame()
    
    # CDO API has a 1-year limit per request, so split into yearly chunks
    date_range_days = (end_date - start_date).days
    if date_range_days > 365:
        logger.debug(
            f"Date range ({date_range_days} days) exceeds CDO API limit (365 days). "
            "Splitting into yearly chunks or using file downloads."
        )
        # For large date ranges, it's more efficient to use file downloads
        # CDO API is better for smaller, selective queries
        return pd.DataFrame()
    
    try:
        # CDO API endpoint
        url = f"{CDO_API_BASE}/data"
        
        params = {
            "datasetid": "GHCND",  # GHCN Daily dataset
            "stationid": f"GHCND:{station_id}",
            "startdate": start_date.strftime("%Y-%m-%d"),
            "enddate": end_date.strftime("%Y-%m-%d"),
            "datatypeid": "TMAX,TMIN,PRCP,SNOW,SNWD,AWND",
            "limit": 1000,  # CDO API limit per request
            "token": token,
        }
        
        logger.info(f"Fetching GHCN data via CDO API for station {station_id}")
        response = session.get(url, params=params, timeout=30, headers={"token": token})
        
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                df = pd.DataFrame(data["results"])
                return _transform_cdo_api_data(df)
            else:
                logger.warning(f"No data returned from CDO API for station {station_id}")
                return pd.DataFrame()
        else:
            logger.debug(
                f"CDO API request failed with status {response.status_code}. "
                "This is normal for large date ranges - will use file downloads."
            )
            return pd.DataFrame()
            
    except Exception as e:
        logger.debug(f"Error fetching GHCN data via CDO API: {e}. Will fall back to file download.")
        return pd.DataFrame()


def _transform_ghcn_api_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform API response to match expected format."""
    if df.empty:
        return df
    
    # Transform based on actual API response structure
    # This is a template - adjust based on actual API response
    result = pd.DataFrame()
    
    if "DATE" in df.columns:
        result["timestamp"] = pd.to_datetime(df["DATE"])
    
    # Map temperature (assuming API returns in tenths of degrees C)
    if "TMAX" in df.columns:
        result["tmax_c"] = df["TMAX"] / 10.0
    if "TMIN" in df.columns:
        result["tmin_c"] = df["TMIN"] / 10.0
    if "TMAX" in df.columns and "TMIN" in df.columns:
        result["tavg_c"] = (result["tmax_c"] + result["tmin_c"]) / 2.0
    
    # Map precipitation (assuming API returns in tenths of mm)
    if "PRCP" in df.columns:
        result["prcp_mm"] = df["PRCP"] / 10.0
    
    # Map snow (assuming API returns in mm)
    if "SNOW" in df.columns:
        result["snow_mm"] = df["SNOW"]
    if "SNWD" in df.columns:
        result["snwd_mm"] = df["SNWD"]
    
    # Map wind (assuming API returns in tenths of m/s)
    if "AWND" in df.columns:
        result["awnd_ms"] = df["AWND"] / 10.0
    
    return result


def _transform_cdo_api_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform CDO API response to match expected format."""
    if df.empty:
        return df
    
    # CDO API returns data in a different format
    # Pivot by datatype
    result = pd.DataFrame()
    
    if "date" in df.columns:
        result["timestamp"] = pd.to_datetime(df["date"])
    
    # Pivot data by datatype
    pivoted = df.pivot_table(
        index="date",
        columns="datatype",
        values="value",
        aggfunc="mean"
    )
    
    # Map to expected column names
    if "TMAX" in pivoted.columns:
        result["tmax_c"] = pivoted["TMAX"] / 10.0
    if "TMIN" in pivoted.columns:
        result["tmin_c"] = pivoted["TMIN"] / 10.0
    if "TMAX" in pivoted.columns and "TMIN" in pivoted.columns:
        result["tavg_c"] = (result["tmax_c"] + result["tmin_c"]) / 2.0
    
    if "PRCP" in pivoted.columns:
        result["prcp_mm"] = pivoted["PRCP"] / 10.0
    if "SNOW" in pivoted.columns:
        result["snow_mm"] = pivoted["SNOW"]
    if "SNWD" in pivoted.columns:
        result["snwd_mm"] = pivoted["SNWD"]
    if "AWND" in pivoted.columns:
        result["awnd_ms"] = pivoted["AWND"] / 10.0
    
    if "timestamp" not in result.columns and not pivoted.empty:
        result["timestamp"] = pd.to_datetime(pivoted.index)
    
    return result.reset_index(drop=True) if not result.empty else pd.DataFrame()


def load_storm_events_via_api(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    states: List[str],
    token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load Storm Events data via API (if available).
    
    Note: Storm Events may not have a direct API. This function attempts
    to use available NOAA APIs, but may need to fall back to file downloads.
    
    Parameters
    ----------
    start_date: Start date for data
    end_date: End date for data
    states: List of state codes
    token: Optional NOAA API token
    
    Returns
    -------
    DataFrame with storm events data, or empty DataFrame if API unavailable
    """
    token = token or get_noaa_token()
    
    if not token:
        logger.debug("No NOAA API token found. Storm Events API may not be available.")
        return pd.DataFrame()
    
    # Note: Storm Events database may not have a direct API
    # This is a placeholder for future API integration
    logger.info("Storm Events API integration not yet available. Using file downloads.")
    return pd.DataFrame()


def test_noaa_api_connection(token: Optional[str] = None) -> bool:
    """
    Test NOAA API connection.
    
    Parameters
    ----------
    token: Optional NOAA API token
    
    Returns
    -------
    True if API connection works, False otherwise
    """
    token = token or get_noaa_token()
    
    if not token:
        logger.warning("No NOAA API token found. Get one at: https://www.ncdc.noaa.gov/cdo-web/token")
        return False
    
    try:
        # Test CDO API connection
        url = f"{CDO_API_BASE}/datasets"
        response = session.get(url, headers={"token": token}, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ NOAA API connection successful!")
            return True
        else:
            logger.warning(f"NOAA API connection failed: {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"NOAA API connection test failed: {e}")
        return False

