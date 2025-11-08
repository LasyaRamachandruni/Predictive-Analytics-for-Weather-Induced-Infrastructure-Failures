"""
map_failures.py
---------------
Visualizes simulated (or real) infrastructure failure intensity on a US map.
Works even if the Natural Earth site blocks direct downloads (uses GeoPandas mirror).
"""

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import pandas as pd


def plot_failure_map():
    """Simulate and visualize infrastructure failures on a geographic map."""
    print("🗺️ Building geospatial failure visualization...")

    # Simulated data — replace later with actual outage/failure dataset
    np.random.seed(42)
    data = pd.DataFrame({
        "longitude": np.random.uniform(-100, -70, 30),
        "latitude": np.random.uniform(30, 45, 30),
        "failure_intensity": np.random.randint(1, 10, 30)
    })

    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(data["longitude"], data["latitude"])]
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

    # ✅ Load Natural Earth low-res world map from GitHub mirror
    world_url = "https://github.com/nvkelso/natural-earth-vector/raw/master/geojson/ne_110m_admin_0_countries.geojson"
    world = gpd.read_file(world_url)

    # Filter for USA only
    usa = world[world["ADMIN"] == "United States of America"]

    # Plot map
    fig, ax = plt.subplots(figsize=(10, 6))
    usa.plot(ax=ax, color="whitesmoke", edgecolor="gray")

    gdf.plot(
        ax=ax,
        column="failure_intensity",
        cmap="Reds",
        markersize=gdf["failure_intensity"] * 25,
        legend=True,
        alpha=0.7
    )

    plt.title("Infrastructure Failure Intensity Map (Simulated Data)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    plot_failure_map()
