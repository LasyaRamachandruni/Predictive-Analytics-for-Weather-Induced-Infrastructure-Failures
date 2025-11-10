"""
Data adapters for weather and infrastructure datasets.

Two modes are supported:

1. `demo` – synthetic data generator so the repository runs offline.
2. `real` – downloads NOAA Storm Events (infrastructure impact proxy) and
   NOAA GHCN daily weather observations to build an aligned, real-world dataset.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from src.utils.io import ensure_directory

logger = logging.getLogger(__name__)

DEFAULT_RANDOM_SEED = 42
STORM_EVENTS_BASE_URL = "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
GHCN_STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
GHCN_DAILY_BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/"
STORM_EVENTS_DATETIME_FORMAT = "%d-%b-%y %H:%M:%S"

STATE_NAME_TO_CODE = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
    "PUERTO RICO": "PR",
}

DEFAULT_REAL_CONFIG = {
    "start_date": "2021-01-01",
    "end_date": "2024-12-31",
    "cache_dir": "data/raw/real",
    "storm_events": {"years": [2021, 2022, 2023, 2024]},
    "ghcn": {
        "stations": {
            "CA": "USW00023234",  # Los Angeles, CA
            "TX": "USW00013904",  # Dallas/Fort Worth, TX
            "FL": "USW00012839",  # Miami, FL
            "NY": "USW00094728",  # New York, NY
            "IL": "USW00094846",  # Chicago, IL
            "GA": "USW00053863",  # Atlanta, GA
            "WA": "USW00024233",  # Seattle, WA
            "AZ": "USW00023183",  # Phoenix, AZ
            "NC": "USW00013723",  # Charlotte, NC
            "CO": "USW00023062",  # Denver, CO
        }
    },
}


@dataclass
class DemoDataConfig:
    """Configuration used for generating demo data."""

    num_regions: int = 6
    start_date: str = "2024-01-01"
    periods: int = 180
    freq: str = "6H"
    base_failure_rate: float = 0.4
    temperature_mean: float = 55.0
    precipitation_scale: float = 3.0
    wind_scale: float = 15.0
    random_state: int = DEFAULT_RANDOM_SEED


def load_dataset(mode: str = "demo", config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load the aligned dataset with columns `[region_id, timestamp, features..., failures]`.
    """
    config = config or {}

    if mode == "demo":
        demo_cfg = DemoDataConfig(**config.get("demo", {}))
        return _load_demo_dataset(demo_cfg)

    if mode == "real":
        merged = _load_real_dataset(config.get("real", {}))
        logger.info(
            "Loaded real dataset with %s rows across %s regions.",
            len(merged),
            merged["region_id"].nunique(),
        )
        return merged

    raise ValueError(f"Unsupported data loading mode: {mode}")


# ---------------------------------------------------------------------------
# Demo dataset (synthetic)
# ---------------------------------------------------------------------------


def _load_demo_dataset(config: DemoDataConfig) -> pd.DataFrame:
    """
    Generate an aligned synthetic dataset covering multiple regions.

    The generated features include temperature, humidity, precipitation, wind, and
    social vulnerability indicators. The target variable `failures` is simulated to
    correlate with extreme weather conditions and vulnerability.
    """
    rng = np.random.default_rng(config.random_state)

    timestamps = pd.date_range(
        start=config.start_date,
        periods=config.periods,
        freq=config.freq,
        inclusive="left",
    )

    region_ids = [f"Region_{i:02d}" for i in range(config.num_regions)]
    latitudes = rng.uniform(30, 47, size=config.num_regions)
    longitudes = rng.uniform(-105, -70, size=config.num_regions)
    elevation = rng.normal(loc=250, scale=120, size=config.num_regions).clip(min=5)
    svi_scores = rng.uniform(0.1, 0.9, size=config.num_regions)

    records = []
    for region_idx, region_id in enumerate(region_ids):
        phase_shift = rng.uniform(0, 2 * np.pi)
        baseline_temp = config.temperature_mean + rng.normal(0, 5)
        region_variability = rng.uniform(0.8, 1.2)
        svi = svi_scores[region_idx]
        elev = elevation[region_idx]
        lon = longitudes[region_idx]
        lat = latitudes[region_idx]

        for step, ts in enumerate(timestamps):
            seasonal_component = 15 * np.sin((2 * np.pi * step / 24) + phase_shift)
            temp = baseline_temp + seasonal_component + rng.normal(0, 2)
            humidity = np.clip(55 + rng.normal(0, 10) + 0.3 * (70 - temp), 10, 100)
            precipitation = np.clip(rng.gamma(shape=2.0, scale=config.precipitation_scale) - 1.5, 0, None)
            wind_speed = np.clip(rng.normal(loc=12, scale=config.wind_scale / 3), 0, None)
            extreme_event = rng.random() < 0.05
            snow_depth = max(0.0, rng.normal(0.5, 1.0) if temp < 32 else 0)

            failure_intensity = (
                config.base_failure_rate * region_variability
                + 0.08 * precipitation
                + 0.05 * max(0, wind_speed - 20)
                + 0.03 * (humidity / 100)
                + 1.5 * extreme_event
                + 0.4 * (1 - np.tanh((temp - 40) / 15) ** 2)
                + 0.6 * svi
                + 0.001 * elev
            )

            failures = float(rng.poisson(lam=max(failure_intensity, 0.05)))

            records.append(
                {
                    "region_id": region_id,
                    "timestamp": ts,
                    "temperature": temp,
                    "humidity": humidity,
                    "precipitation": precipitation,
                    "wind_speed": wind_speed,
                    "extreme_event": int(extreme_event),
                    "snow_depth": snow_depth,
                    "svi": svi,
                    "elevation": elev,
                    "longitude": lon,
                    "latitude": lat,
                    "failures": failures,
                }
            )

    df = pd.DataFrame.from_records(records)
    df.sort_values(by=["region_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Real dataset (NOAA Storm Events + GHCN Daily)
# ---------------------------------------------------------------------------


def _download_file(url: str, destination: Path) -> Path:
    ensure_directory(destination.parent)
    if destination.exists():
        return destination
    logger.info("Downloading %s -> %s", url, destination)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request) as response, destination.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    return destination


def _fetch_storm_events_catalog() -> Dict[str, str]:
    """Return mapping from year to the latest Storm Events detail filename."""
    request = Request(STORM_EVENTS_BASE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as resp:
        html = resp.read().decode("utf-8")

    matches = re.findall(
        r"StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d+)\.csv\.gz", html, flags=re.IGNORECASE
    )
    catalog: Dict[str, Tuple[str, str]] = {}
    for year, stamp in matches:
        filename = f"StormEvents_details-ftp_v1.0_d{year}_c{stamp}.csv.gz"
        if year not in catalog or stamp > catalog[year][0]:
            catalog[year] = (stamp, filename)
    return {year: entry[1] for year, entry in catalog.items()}


def _parse_damage_value(value: Any) -> float:
    """Convert NOAA damage property strings to absolute USD."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    text = str(value).strip()
    if not text or text in {"0", "0.00", "0.00K"}:
        return 0.0
    multiplier = 1.0
    suffix = text[-1].upper()
    if suffix in {"K", "M", "B"}:
        text = text[:-1]
        multiplier = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
    try:
        return float(text) * multiplier
    except ValueError:
        return 0.0


def _download_storm_events_files(years: Sequence[int], cache_dir: Path) -> List[Path]:
    catalog = _fetch_storm_events_catalog()
    downloaded: List[Path] = []
    for year in years:
        year_str = str(year)
        if year_str not in catalog:
            logger.warning("Storm Events catalog missing year %s", year)
            continue
        filename = catalog[year_str]
        url = f"{STORM_EVENTS_BASE_URL}{filename}"
        destination = cache_dir / filename
        _download_file(url, destination)
        downloaded.append(destination)
    if not downloaded:
        raise RuntimeError("No Storm Events files were downloaded; check year configuration.")
    return downloaded


def _load_storm_events_dataset(
    files: Sequence[Path],
    states: Sequence[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    usecols = [
        "EVENT_ID",
        "STATE",
        "EVENT_TYPE",
        "BEGIN_DATE_TIME",
        "BEGIN_LAT",
        "BEGIN_LON",
        "DAMAGE_PROPERTY",
        "DEATHS_DIRECT",
        "INJURIES_DIRECT",
        "MAGNITUDE",
    ]

    frames = []
    for file_path in files:
        logger.info("Loading Storm Events file %s", file_path.name)
        frame = pd.read_csv(
            file_path,
            usecols=usecols,
            compression="gzip",
            dtype={"EVENT_ID": "int64", "STATE": "string", "EVENT_TYPE": "string"},
            low_memory=False,
        )

        parsed_dates = pd.to_datetime(
            frame["BEGIN_DATE_TIME"],
            format=STORM_EVENTS_DATETIME_FORMAT,
            errors="coerce",
        )
        missing_mask = parsed_dates.isna() & frame["BEGIN_DATE_TIME"].notna()
        if missing_mask.any():
            parsed_dates.loc[missing_mask] = pd.to_datetime(
                frame.loc[missing_mask, "BEGIN_DATE_TIME"],
                errors="coerce",
            )
        frame["BEGIN_DATE_TIME"] = parsed_dates
        frames.append(frame)

    events = pd.concat(frames, axis=0, ignore_index=True)
    events.dropna(subset=["BEGIN_DATE_TIME", "STATE"], inplace=True)

    events["state_code"] = (
        events["STATE"].str.upper().map(STATE_NAME_TO_CODE).fillna(events["STATE"].str.upper())
    )
    events = events[events["state_code"].isin(states)]

    events["timestamp"] = events["BEGIN_DATE_TIME"].dt.floor("D")
    mask = (events["timestamp"] >= start_date) & (events["timestamp"] <= end_date)
    events = events.loc[mask].copy()

    events["damage_usd"] = events["DAMAGE_PROPERTY"].apply(_parse_damage_value)
    events["event_type_upper"] = events["EVENT_TYPE"].str.upper()
    events["is_wind"] = events["event_type_upper"].str.contains("WIND", na=False)
    events["is_flood"] = events["event_type_upper"].str.contains("FLOOD", na=False)
    events["is_winter"] = events["event_type_upper"].str.contains("WINTER|BLIZZARD|SNOW", na=False)
    events["is_hail"] = events["event_type_upper"].str.contains("HAIL", na=False)
    events["is_tornado"] = events["event_type_upper"].str.contains("TORNADO", na=False)
    events["is_lightning"] = events["event_type_upper"].str.contains("LIGHTNING", na=False)

    aggregated = (
        events.groupby(["state_code", "timestamp"])
        .agg(
            failures=("damage_usd", "sum"),
            event_count=("EVENT_ID", "count"),
            wind_event_count=("is_wind", "sum"),
            flood_event_count=("is_flood", "sum"),
            winter_event_count=("is_winter", "sum"),
            hail_event_count=("is_hail", "sum"),
            tornado_event_count=("is_tornado", "sum"),
            lightning_event_count=("is_lightning", "sum"),
            deaths=("DEATHS_DIRECT", "sum"),
            injuries=("INJURIES_DIRECT", "sum"),
            avg_magnitude=("MAGNITUDE", "mean"),
            avg_latitude=("BEGIN_LAT", "mean"),
            avg_longitude=("BEGIN_LON", "mean"),
        )
        .reset_index()
    )

    aggregated.rename(columns={"state_code": "region_id"}, inplace=True)
    aggregated["failures"] = aggregated["failures"].fillna(0.0) / 1_000.0  # thousands USD
    return aggregated


def _download_station_metadata(cache_dir: Path) -> Path:
    path = cache_dir / "ghcnd-stations.txt"
    return _download_file(GHCN_STATIONS_URL, path)


def _parse_station_metadata(path: Path, station_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    station_ids = set(station_ids)
    metadata: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            station_id = line[0:11].strip()
            if station_id not in station_ids:
                continue
            lat = float(line[12:20].strip())
            lon = float(line[21:30].strip())
            elev = float(line[31:37].strip())
            state = line[38:40].strip()
            name = line[41:71].strip()
            metadata[station_id] = {
                "latitude": lat,
                "longitude": lon,
                "elevation": elev,
                "state": state,
                "name": name,
            }
    missing = station_ids.difference(metadata.keys())
    if missing:
        logger.warning("Missing station metadata for %s", ", ".join(sorted(missing)))
    return metadata


def _download_ghcn_station(station_id: str, cache_dir: Path) -> Path:
    filename = f"{station_id}.dly"
    url = f"{GHCN_DAILY_BASE_URL}{filename}"
    return _download_file(url, cache_dir / filename)


def _parse_ghcn_daily(path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    records: List[Tuple[datetime, str, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            station = line[0:11]
            year = int(line[11:15])
            month = int(line[15:17])
            element = line[17:21]
            if element not in {"TMAX", "TMIN", "PRCP", "SNOW", "SNWD", "AWND"}:
                continue

            for day in range(31):
                value = line[21 + day * 8 : 26 + day * 8]
                qflag = line[27 + day * 8]
                if qflag.strip() or value.strip() == "-9999":
                    continue
                try:
                    value_num = float(value)
                except ValueError:
                    continue
                try:
                    current_date = datetime(year, month, day + 1)
                except ValueError:
                    continue
                if not (start_date <= pd.Timestamp(current_date) <= end_date):
                    continue
                records.append((current_date, element, value_num))

    if not records:
        return pd.DataFrame(columns=["timestamp"])

    df = pd.DataFrame(records, columns=["timestamp", "element", "raw"])
    df = df.pivot_table(index="timestamp", columns="element", values="raw", aggfunc="mean")
    df.reset_index(inplace=True)

    if "TMAX" in df.columns:
        df["tmax_c"] = df["TMAX"] / 10.0
    if "TMIN" in df.columns:
        df["tmin_c"] = df["TMIN"] / 10.0
    if "PRCP" in df.columns:
        df["prcp_mm"] = df["PRCP"] / 10.0
    if "SNOW" in df.columns:
        df["snow_mm"] = df["SNOW"]
    if "SNWD" in df.columns:
        df["snwd_mm"] = df["SNWD"]
    if "AWND" in df.columns:
        df["awnd_ms"] = df["AWND"] / 10.0

    df["tavg_c"] = df.filter(items=["tmax_c", "tmin_c"]).mean(axis=1)
    desired_columns = ["timestamp", "tavg_c", "tmax_c", "tmin_c", "prcp_mm", "snow_mm", "snwd_mm", "awnd_ms"]
    available_columns = [col for col in desired_columns if col in df.columns]
    missing_columns = set(desired_columns) - set(available_columns)
    for col in missing_columns:
        df[col] = np.nan
        available_columns.append(col)
    return df[available_columns]


def _load_ghcn_weather(
    stations: Dict[str, Any],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    station_map: Dict[str, str] = {}
    for region, data in stations.items():
        if isinstance(data, dict):
            station_map[region] = data["id"]
        else:
            station_map[region] = str(data)

    metadata_path = _download_station_metadata(cache_dir)
    station_metadata = _parse_station_metadata(metadata_path, station_map.values())

    frames: List[pd.DataFrame] = []
    coords: Dict[str, Dict[str, Any]] = {}

    for region_id, station_id in station_map.items():
        station_path = _download_ghcn_station(station_id, cache_dir)
        df = _parse_ghcn_daily(station_path, start_date, end_date)
        if df.empty:
            logger.warning("No GHCN data for station %s (%s)", station_id, region_id)
            continue

        meta = station_metadata.get(station_id, {})
        coords[region_id] = {
            "latitude": meta.get("latitude"),
            "longitude": meta.get("longitude"),
            "elevation": meta.get("elevation"),
        }

        df["region_id"] = region_id
        frames.append(df)

    if not frames:
        raise RuntimeError("No GHCN weather data available for configured stations.")

    weather = pd.concat(frames, axis=0, ignore_index=True)
    weather.sort_values(by=["region_id", "timestamp"], inplace=True)
    return weather, coords


def _combine_real_dataset(
    weather: pd.DataFrame,
    coords: Dict[str, Dict[str, Any]],
    failures: pd.DataFrame,
    regions: Sequence[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [sorted(set(regions)), pd.date_range(start_date, end_date, freq="D")],
        names=["region_id", "timestamp"],
    )
    base = pd.DataFrame(index=idx).reset_index()

    combined = base.merge(weather, how="left", on=["region_id", "timestamp"])
    combined = combined.merge(failures, how="left", on=["region_id", "timestamp"])

    weather_cols = ["tavg_c", "tmax_c", "tmin_c", "prcp_mm", "snow_mm", "snwd_mm", "awnd_ms"]
    combined[weather_cols] = (
        combined.groupby("region_id")[weather_cols]
        .apply(lambda group: group.fillna(method="ffill").fillna(method="bfill"))
        .reset_index(level=0, drop=True)
    )

    count_cols = [
        "event_count",
        "wind_event_count",
        "flood_event_count",
        "winter_event_count",
        "hail_event_count",
        "tornado_event_count",
        "lightning_event_count",
        "deaths",
        "injuries",
    ]
    for col in count_cols:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0.0)

    combined["failures"] = combined["failures"].fillna(0.0)
    combined["avg_magnitude"] = combined["avg_magnitude"].fillna(0.0)
    combined["latitude"] = combined["region_id"].map(lambda r: coords.get(r, {}).get("latitude"))
    combined["longitude"] = combined["region_id"].map(lambda r: coords.get(r, {}).get("longitude"))

    combined.sort_values(by=["region_id", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _load_real_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    cfg = DEFAULT_REAL_CONFIG.copy()
    cfg.update(config or {})

    start_date = pd.to_datetime(cfg.get("start_date")).normalize()
    end_date = pd.to_datetime(cfg.get("end_date")).normalize()
    if end_date < start_date:
        raise ValueError("`end_date` must be on or after `start_date`.")

    cache_dir = ensure_directory(Path(cfg.get("cache_dir", "data/raw/real")))
    storm_cache = ensure_directory(cache_dir / "storm_events")
    ghcn_cache = ensure_directory(cache_dir / "ghcn")

    storm_years = cfg.get("storm_events", {}).get("years")
    if not storm_years:
        storm_years = list(range(start_date.year, end_date.year + 1))

    stations_cfg = cfg.get("ghcn", {}).get("stations") or DEFAULT_REAL_CONFIG["ghcn"]["stations"]
    region_ids = list(stations_cfg.keys())

    storm_files = _download_storm_events_files(storm_years, storm_cache)
    storm_df = _load_storm_events_dataset(storm_files, region_ids, start_date, end_date)

    weather_df, coords = _load_ghcn_weather(stations_cfg, start_date, end_date, ghcn_cache)

    merged = _combine_real_dataset(weather_df, coords, storm_df, region_ids, start_date, end_date)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])

    return merged