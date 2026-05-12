"""
Trajectory analysis module for public transport accessibility.

This module provides functions to:
- Query OSRM routing service for different transport modes
- Extract street view images along trajectories
- Calculate density features for route segments
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import requests
from shapely.geometry import LineString, Point


# OSRM service endpoints
OSRM = {
    "simple": "http://localhost:5000", # do not use
    "foot":   "http://localhost:5001",
    "bike":   "http://localhost:5002",
    "car":    "http://localhost:5003",
}


def get_osrm_route(
    lon1, lat1,
    lon2, lat2,
    mode,
    overview="simplified",
    geometries="geojson",
):
    """
    Query OSRM and return route geometry + metadata.

    Parameters
    ----------
    lon1 : float
        Longitude of origin
    lat1 : float
        Latitude of origin
    lon2 : float
        Longitude of destination
    lat2 : float
        Latitude of destination
    mode : str
        Transport mode: one of {"foot", "bike", "car"}
    overview : str
        Route geometry detail level (default: "simplified")
    geometries : str
        Geometry format (default: "geojson")

    Returns
    -------
    dict
        Dictionary containing:
        - mode: transport mode
        - geometry: LineString of route
        - distance_m: route distance in meters
        - duration_s: route duration in seconds
        - raw: full OSRM response

    Raises
    ------
    ValueError
        If mode is not recognized
    requests.HTTPError
        If OSRM request fails
    """
    if mode not in OSRM:
        raise ValueError(f"Unknown mode '{mode}'. Available: {list(OSRM)}")

    url = (
        f"{OSRM[mode]}/route/v1/{mode}/"
        f"{lon1},{lat1};{lon2},{lat2}"
    )

    params = {
        "overview": overview,
        "geometries": geometries,
    }

    r = requests.get(url, params=params)
    r.raise_for_status()

    data = r.json()
    route = data["routes"][0]
    geom = route["geometry"]

    return {
        "mode": mode,
        "geometry": LineString(geom["coordinates"]),
        "distance_m": route["distance"],
        "duration_s": route["duration"],
        "raw": data,
    }


def get_trajectory_images(p_home, home_station_geometry, gdf_img, mode="foot", buffer_m=20):
    """
    Create a route from home to station and find all images within buffer distance.

    Parameters
    ----------
    p_home : Point
        Home location (EPSG:4326)
    home_station_geometry : Point
        Station location (EPSG:4326)
    gdf_img : GeoDataFrame
        GeoDataFrame with street view images (should be in EPSG:3857)
    mode : str
        Transport mode: "foot", "bike", or "car" (default: "foot")
    buffer_m : float
        Buffer distance in meters (default: 20)

    Returns
    -------
    dict
        Dictionary containing:
        - images: GeoDataFrame of images within buffer
        - trajectory: GeoSeries of buffered trajectory (EPSG:3857)
        - route_line: LineString of route (EPSG:4326)
        - distance_m: route distance in meters
        - duration_s: route duration in seconds
        - mode: transport mode used
        - n_images: number of images found

    Examples
    --------
    >>> result = get_trajectory_images(
    ...     p_home=Point(12.5, 55.7),
    ...     home_station_geometry=Point(12.51, 55.71),
    ...     gdf_img=gdf_img,
    ...     mode="foot",
    ...     buffer_m=20
    ... )
    >>> print(f"Found {result['n_images']} images")
    >>> result['images'].head()
    """
    # Get route from OSRM
    output = get_osrm_route(
        lon1=p_home.x,
        lat1=p_home.y,
        lon2=home_station_geometry.x,
        lat2=home_station_geometry.y,
        mode=mode,
    )

    # Extract geometry
    L = output['raw']['routes'][0]['geometry']
    geo = LineString(L['coordinates'])

    # Create buffered trajectory in EPSG:3857 to match gdf_img
    traj = gpd.GeoSeries([geo], crs='EPSG:4326').to_crs('EPSG:32632').buffer(buffer_m).to_crs('EPSG:3857')

    # Find images within buffer
    imgs_within = gdf_img[gdf_img.within(traj.iloc[0])]

    return {
        'images': imgs_within,
        'trajectory': traj,
        'route_line': geo,
        'distance_m': output['distance_m'],
        'duration_s': output['duration_s'],
        'mode': mode,
        'n_images': len(imgs_within)
    }


def calculate_trajectory_features(result, density_cols=None):
    """
    Calculate aggregated density features from images along a trajectory.

    Parameters
    ----------
    result : dict
        Output from get_trajectory_images()
    density_cols : list, optional
        List of density column names to aggregate. If None, will use all columns
        ending with '_density'

    Returns
    -------
    dict
        Dictionary with mean, median, std, min, max for each density feature
    """
    imgs = result['images']

    if len(imgs) == 0:
        return {}

    if density_cols is None:
        density_cols = [col for col in imgs.columns if col.endswith('_density')]

    features = {}
    for col in density_cols:
        features[f'{col}_mean'] = imgs[col].mean()
        features[f'{col}_median'] = imgs[col].median()
        features[f'{col}_std'] = imgs[col].std()
        features[f'{col}_min'] = imgs[col].min()
        features[f'{col}_max'] = imgs[col].max()

    for col in density_cols: # L2 norm
        col_l2 = f'{col}_l2_norm'
        features[f'{col_l2}_mean'] = imgs[col_l2].mean()
        features[f'{col_l2}_median'] = imgs[col_l2].median()
        features[f'{col_l2}_std'] = imgs[col_l2].std()
        features[f'{col_l2}_min'] = imgs[col_l2].min()
        features[f'{col_l2}_max'] = imgs[col_l2].max()

    for col in density_cols: # min-max norm
        col_mm = col.replace('_density', '_density_minmax')
        features[f'{col_mm}_mean'] = imgs[col_mm].mean()
        features[f'{col_mm}_median'] = imgs[col_mm].median()
        features[f'{col_mm}_std'] = imgs[col_mm].std()
        features[f'{col_mm}_min'] = imgs[col_mm].min()
        features[f'{col_mm}_max'] = imgs[col_mm].max()

    for col in density_cols: # quantile norm
        col_qt = col.replace('_density', '_density_quantile')
        features[f'{col_qt}_mean'] = imgs[col_qt].mean()
        features[f'{col_qt}_median'] = imgs[col_qt].median()
        features[f'{col_qt}_std'] = imgs[col_qt].std()
        features[f'{col_qt}_min'] = imgs[col_qt].min()
        features[f'{col_qt}_max'] = imgs[col_qt].max()

    features['n_images'] = len(imgs)
    features['distance_m'] = result['distance_m']
    features['duration_s'] = result['duration_s']

    return features
