"""
Additional data sources for comprehensive infrastructure failure prediction.

This module provides loaders for:
- Power outage records
- Infrastructure age and condition
- Population density
- Economic indicators
- Additional weather variables
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Infrastructure age estimates based on ASCE report cards and general knowledge
# These are placeholder values - in production, load from actual data sources
INFRASTRUCTURE_AGE_DATA = {
    "CA": {"age": 30, "maintenance_score": 6.5, "condition_rating": 6.5},
    "TX": {"age": 25, "maintenance_score": 7.0, "condition_rating": 7.0},
    "FL": {"age": 28, "maintenance_score": 6.8, "condition_rating": 6.8},
    "NY": {"age": 35, "maintenance_score": 6.0, "condition_rating": 6.0},
    "IL": {"age": 32, "maintenance_score": 6.2, "condition_rating": 6.2},
    "GA": {"age": 27, "maintenance_score": 7.2, "condition_rating": 7.2},
    "WA": {"age": 29, "maintenance_score": 6.7, "condition_rating": 6.7},
    "AZ": {"age": 26, "maintenance_score": 7.1, "condition_rating": 7.1},
    "NC": {"age": 28, "maintenance_score": 6.9, "condition_rating": 6.9},
    "CO": {"age": 24, "maintenance_score": 7.3, "condition_rating": 7.3},
    "PA": {"age": 33, "maintenance_score": 6.1, "condition_rating": 6.1},
    "OH": {"age": 31, "maintenance_score": 6.4, "condition_rating": 6.4},
    "MI": {"age": 29, "maintenance_score": 6.6, "condition_rating": 6.6},
    "MA": {"age": 34, "maintenance_score": 6.3, "condition_rating": 6.3},
    "OR": {"age": 27, "maintenance_score": 7.0, "condition_rating": 7.0},
    "TN": {"age": 26, "maintenance_score": 7.1, "condition_rating": 7.1},
    "IN": {"age": 28, "maintenance_score": 6.8, "condition_rating": 6.8},
    "MO": {"age": 30, "maintenance_score": 6.5, "condition_rating": 6.5},
    "MD": {"age": 32, "maintenance_score": 6.2, "condition_rating": 6.2},
    "WI": {"age": 28, "maintenance_score": 6.9, "condition_rating": 6.9},
}

# Population density estimates (people per square mile)
# Source: US Census Bureau estimates
POPULATION_DATA = {
    "CA": {"density": 253.6, "urban_pct": 95.0, "total_pop": 39538223},
    "TX": {"density": 114.4, "urban_pct": 83.7, "total_pop": 29145505},
    "FL": {"density": 416.9, "urban_pct": 91.2, "total_pop": 21538187},
    "NY": {"density": 421.2, "urban_pct": 87.9, "total_pop": 20201249},
    "IL": {"density": 231.1, "urban_pct": 88.1, "total_pop": 12812508},
    "GA": {"density": 185.2, "urban_pct": 75.1, "total_pop": 10711908},
    "WA": {"density": 115.0, "urban_pct": 84.1, "total_pop": 7705281},
    "AZ": {"density": 65.3, "urban_pct": 89.8, "total_pop": 7151502},
    "NC": {"density": 218.0, "urban_pct": 66.1, "total_pop": 10439388},
    "CO": {"density": 56.9, "urban_pct": 86.2, "total_pop": 5773714},
    "PA": {"density": 290.0, "urban_pct": 78.7, "total_pop": 13002700},
    "OH": {"density": 282.3, "urban_pct": 77.9, "total_pop": 11799448},
    "MI": {"density": 174.8, "urban_pct": 74.4, "total_pop": 10037261},
    "MA": {"density": 890.3, "urban_pct": 92.0, "total_pop": 7029917},
    "OR": {"density": 45.2, "urban_pct": 81.2, "total_pop": 4237256},
    "TN": {"density": 166.0, "urban_pct": 66.4, "total_pop": 6910840},
    "IN": {"density": 188.1, "urban_pct": 72.4, "total_pop": 6785528},
    "MO": {"density": 89.3, "urban_pct": 70.4, "total_pop": 6177957},
    "MD": {"density": 636.1, "urban_pct": 87.2, "total_pop": 6177224},
    "WI": {"density": 108.2, "urban_pct": 70.2, "total_pop": 5893718},
}

# GDP per capita estimates (thousands USD)
# Source: Bureau of Economic Analysis
ECONOMIC_DATA = {
    "CA": {"gdp_per_capita": 79.4, "infrastructure_investment": 12.5, "poverty_rate": 11.2},
    "TX": {"gdp_per_capita": 65.2, "infrastructure_investment": 10.8, "poverty_rate": 13.6},
    "FL": {"gdp_per_capita": 58.3, "infrastructure_investment": 9.2, "poverty_rate": 12.7},
    "NY": {"gdp_per_capita": 85.7, "infrastructure_investment": 14.1, "poverty_rate": 13.0},
    "IL": {"gdp_per_capita": 70.1, "infrastructure_investment": 11.3, "poverty_rate": 11.5},
    "GA": {"gdp_per_capita": 62.4, "infrastructure_investment": 9.8, "poverty_rate": 13.3},
    "WA": {"gdp_per_capita": 82.3, "infrastructure_investment": 13.2, "poverty_rate": 9.8},
    "AZ": {"gdp_per_capita": 55.8, "infrastructure_investment": 8.5, "poverty_rate": 12.8},
    "NC": {"gdp_per_capita": 59.7, "infrastructure_investment": 9.1, "poverty_rate": 12.9},
    "CO": {"gdp_per_capita": 73.5, "infrastructure_investment": 11.9, "poverty_rate": 9.3},
    "PA": {"gdp_per_capita": 68.2, "infrastructure_investment": 11.5, "poverty_rate": 11.8},
    "OH": {"gdp_per_capita": 64.8, "infrastructure_investment": 10.2, "poverty_rate": 12.6},
    "MI": {"gdp_per_capita": 61.5, "infrastructure_investment": 9.5, "poverty_rate": 13.1},
    "MA": {"gdp_per_capita": 83.1, "infrastructure_investment": 13.8, "poverty_rate": 9.4},
    "OR": {"gdp_per_capita": 66.7, "infrastructure_investment": 10.5, "poverty_rate": 11.2},
    "TN": {"gdp_per_capita": 60.3, "infrastructure_investment": 9.0, "poverty_rate": 13.1},
    "IN": {"gdp_per_capita": 63.9, "infrastructure_investment": 9.7, "poverty_rate": 12.2},
    "MO": {"gdp_per_capita": 59.2, "infrastructure_investment": 8.8, "poverty_rate": 12.7},
    "MD": {"gdp_per_capita": 78.6, "infrastructure_investment": 12.8, "poverty_rate": 9.0},
    "WI": {"gdp_per_capita": 65.4, "infrastructure_investment": 10.3, "poverty_rate": 10.4},
}


def load_infrastructure_age_data(regions: List[str]) -> pd.DataFrame:
    """
    Load infrastructure age and condition data.
    
    Currently uses placeholder data. In production, this would:
    - Query ASCE Infrastructure Report Card API
    - Scrape state DOT websites
    - Use FEMA infrastructure assessments
    
    Parameters
    ----------
    regions: List of region IDs (state codes)
    
    Returns
    -------
    DataFrame with columns: region_id, avg_infrastructure_age, maintenance_score, condition_rating
    """
    data = []
    for region in regions:
        info = INFRASTRUCTURE_AGE_DATA.get(region, {"age": 27, "maintenance_score": 6.5, "condition_rating": 6.5})
        data.append(
            {
                "region_id": region,
                "avg_infrastructure_age": info["age"],
                "maintenance_score": info["maintenance_score"],
                "condition_rating": info["condition_rating"],
            }
        )
    return pd.DataFrame(data)


def load_population_data(regions: List[str], use_api: bool = True) -> pd.DataFrame:
    """
    Load population density and demographic data from Census API.
    
    Requires CENSUS_API_KEY environment variable. See docs/API_INTEGRATION_GUIDE.md for setup.
    
    Parameters
    ----------
    regions: List of region IDs (state codes)
    use_api: Must be True (API is required)
    
    Returns
    -------
    DataFrame with columns: region_id, population_density, urban_percentage, total_population
    
    Raises
    ------
    RuntimeError: If API data cannot be loaded
    """
    if not use_api:
        raise ValueError("API data is required. Set use_api=True and configure CENSUS_API_KEY.")
    
    try:
        from .api_loaders import load_population_data_census
        api_df = load_population_data_census(regions)
        if not api_df.empty:
            logger.info("Using real Census API data for population")
            return api_df
        else:
            raise RuntimeError(
                "Census API returned empty data. Please check your CENSUS_API_KEY and ensure it's activated. "
                "See docs/API_INTEGRATION_GUIDE.md for setup instructions."
            )
    except ImportError:
        raise RuntimeError("api_loaders module not available. Cannot load Census data.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Census API data: {e}. "
            "Please configure CENSUS_API_KEY environment variable. "
            "See docs/API_INTEGRATION_GUIDE.md for setup instructions."
        ) from e


def load_economic_data(regions: List[str], use_api: bool = True) -> pd.DataFrame:
    """
    Load economic indicators from BEA or FRED API.
    
    Requires BEA_API_KEY or FRED_API_KEY environment variable. See docs/API_INTEGRATION_GUIDE.md for setup.
    
    Parameters
    ----------
    regions: List of region IDs (state codes)
    use_api: Must be True (API is required)
    
    Returns
    -------
    DataFrame with columns: region_id, gdp_per_capita, infrastructure_investment, poverty_rate
    
    Raises
    ------
    RuntimeError: If API data cannot be loaded
    """
    if not use_api:
        raise ValueError("API data is required. Set use_api=True and configure BEA_API_KEY or FRED_API_KEY.")
    
    try:
        from .api_loaders import load_economic_data_bea, load_economic_data_fred
        # Try BEA first
        api_df = load_economic_data_bea(regions)
        if not api_df.empty:
            logger.info("Using real BEA API data for economic indicators")
            return api_df
        # Try FRED as fallback
        api_df = load_economic_data_fred(regions)
        if not api_df.empty:
            logger.info("Using real FRED API data for economic indicators")
            return api_df
        else:
            raise RuntimeError(
                "Both BEA and FRED APIs returned empty data. Please check your API keys (BEA_API_KEY or FRED_API_KEY) "
                "and ensure they're activated. See docs/API_INTEGRATION_GUIDE.md for setup instructions."
            )
    except ImportError:
        raise RuntimeError("api_loaders module not available. Cannot load economic data.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load economic API data: {e}. "
            "Please configure BEA_API_KEY or FRED_API_KEY environment variable. "
            "See docs/API_INTEGRATION_GUIDE.md for setup instructions."
        ) from e


def load_power_outage_data(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    regions: List[str],
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load power outage data from EIA or utility APIs.
    
    TODO: Implement actual data loading from:
    - EIA Form 861 (Annual utility statistics)
    - DOE Grid Modernization data
    - Utility company APIs
    
    Parameters
    ----------
    start_date: Start date for outage records
    end_date: End date for outage records
    regions: List of region IDs
    cache_dir: Optional cache directory
    
    Returns
    -------
    DataFrame with columns: region_id, timestamp, customers_affected, outage_duration_hours, cause
    """
    logger.warning(
        "Power outage data loader not yet implemented. "
        "Returning empty DataFrame. See ADDING_MORE_DATA.md for implementation guide."
    )
    # Placeholder - would return actual outage data
    return pd.DataFrame(
        columns=["region_id", "timestamp", "customers_affected", "outage_duration_hours", "cause"]
    )


def load_additional_weather_variables(
    stations: Dict[str, str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
) -> pd.DataFrame:
    """
    Load additional weather variables (humidity, pressure, etc.).
    
    TODO: Extend GHCN parser to include more variables or use additional APIs.
    
    Parameters
    ----------
    stations: Dict mapping region_id to station_id
    start_date: Start date
    end_date: End date
    cache_dir: Cache directory
    
    Returns
    -------
    DataFrame with additional weather variables
    """
    logger.warning(
        "Additional weather variables loader not yet implemented. "
        "Returning empty DataFrame."
    )
    return pd.DataFrame(columns=["region_id", "timestamp"])

