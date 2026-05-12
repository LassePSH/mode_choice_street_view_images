#!/usr/bin/env python3
"""
Compute street view density features for each public transport station
used in trajectory_for_mode_choice, using circular buffers of 250, 500, 750, 1000m.

For each station and each buffer radius, street view images within the buffer
are aggregated with mean, median, std, min, max.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyproj import Transformer
from shapely.geometry import Point
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer


# ── paths ─────────────────────────────────────────────────────────────────────
GTFS_PATH = '/home/lpsha/s154446/design_standard_public_transport/data/choice_model/GTFS_TU_StationList_v19_grouped.csv'
TU_PATH   = '/work/lpsha/data/design_standard_for_public_transport/tu/df_basis_with_home_activity_coords_26_2_2.csv'
META_PATH = '/data/google_street_view/denmark_big/meta.csv'
DINO_PATH = '/data/google_street_view/denmark_big/DINOSAM_26c.csv'
OUTPUT    = Path(__file__).parent / 'station_buffer_features.csv'

BUFFERS   = [250, 500, 750, 1000]

# density columns (same as process_trajectories.py)
BASE_COLS = [
    'NaN', 'sidewalk', 'sky', 'buildings', 'road', 'grass', 'trees',
    'plants', 'bushes', 'hedge', 'water', 'door', 'fencing', 'window',
    'graffiti', 'bench', 'street sign', 'pole', 'traffic light',
    'trash can', 'bicycles', 'scooter', 'motorcycle', 'car', 'bus',
    'truck', 'person',
]
DERIVED_COLS = [
    'greens', 'building_window_ratio', 'building_all',
    'road_life_g1', 'urban_life_g1', 'road_life_g2', 'vehicle_life_g2',
]
ALL_COLS = BASE_COLS + DERIVED_COLS

DENSITY_VARIANTS = ['_density', '_density_l2_norm', '_density_minmax', '_density_quantile']
AGG_FUNCS = ['mean', 'median', 'std', 'min', 'max']


def load_stations():
    """Load GTFS stations and keep only those used in the TU survey data."""
    df_gtfs = pd.read_csv(GTFS_PATH)
    gdf_st = gpd.GeoDataFrame(
        df_gtfs,
        geometry=gpd.points_from_xy(df_gtfs['stop_lon'], df_gtfs['stop_lat']),
        crs='EPSG:4326',
    )

    df_tu = pd.read_csv(TU_PATH, sep=',')
    used_stations = set(df_tu['HOME_NearestStation'].dropna()) | set(df_tu['ACT_NearestStation'].dropna())
    gdf_st = gdf_st[gdf_st['TU'].isin(used_stations)].copy()
    gdf_st = gdf_st.drop_duplicates(subset='TU').reset_index(drop=True)

    print(f"Stations used in TU data: {len(gdf_st)}")
    return gdf_st


def load_images():
    """Load and preprocess street view images (same pipeline as process_trajectories.py)."""
    print("Loading street view images...")
    meta = pd.read_csv(META_PATH)
    dino = pd.read_csv(DINO_PATH)
    dino['id'] = dino['id'].apply(lambda x: x[:-4])
    dino.set_index('id', inplace=True)
    meta.set_index('id', inplace=True)
    merged = meta.join(dino, how='inner')

    merged['geometry'] = gpd.points_from_xy(merged['lon'], merged['lat'])
    gdf_img = gpd.GeoDataFrame(merged, geometry='geometry', crs='EPSG:4326')

    # ── density features ──────────────────────────────────────────────────────
    for col in BASE_COLS:
        gdf_img[col + '_density'] = gdf_img[col] / gdf_img[BASE_COLS].sum(axis=1)

    gdf_img = gdf_img[gdf_img['NaN_density'] < gdf_img['NaN_density'].quantile(0.90)]

    gdf_img['greens_density'] = gdf_img[['grass_density', 'trees_density', 'plants_density',
                                          'bushes_density', 'hedge_density']].sum(axis=1)
    gdf_img['building_window_ratio_density'] = gdf_img['window_density'] / (gdf_img['buildings_density'] + 1e-6)
    gdf_img['building_all_density'] = (gdf_img['buildings_density'] + gdf_img['door_density']
                                        + gdf_img['window_density'])
    gdf_img['road_life_g1_density'] = gdf_img[['road_density', 'street sign_density', 'pole_density',
                                                 'traffic light_density', 'trash can_density',
                                                 'scooter_density', 'motorcycle_density', 'car_density',
                                                 'bus_density', 'truck_density', 'bicycles_density']].sum(axis=1)
    
    gdf_img['urban_life_g1_density'] = gdf_img[['person_density', 'bench_density', 'sidewalk_density']].sum(axis=1)
    gdf_img['road_life_g2_density'] = gdf_img[['road_density', 'street sign_density', 'pole_density',
                                                 'traffic light_density', 'trash can_density']].sum(axis=1)
    gdf_img['vehicle_life_g2_density'] = gdf_img[['scooter_density', 'motorcycle_density', 'car_density',
                                                    'bus_density', 'truck_density', 'bicycles_density']].sum(axis=1)

    # L2 norm
    for col in ALL_COLS:
        x = gdf_img[[col + '_density']]
        gdf_img[col + '_density_l2_norm'] = preprocessing.normalize(x).flatten()

    # Min-max norm
    scaler = MinMaxScaler()
    for col in ALL_COLS:
        gdf_img[col + '_density_minmax'] = scaler.fit_transform(gdf_img[[col + '_density']])

    # Quantile norm
    density_cols = [col + '_density' for col in ALL_COLS]
    qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
    qt_data = qt.fit_transform(gdf_img[density_cols])
    qt_df = pd.DataFrame(qt_data,
                         columns=[col + '_density_quantile' for col in ALL_COLS],
                         index=gdf_img.index)
    gdf_img = pd.concat([gdf_img, qt_df], axis=1)

    print(f"Loaded {len(gdf_img)} street view images")
    return gdf_img


def compute_buffer_features(gdf_st, gdf_img):
    """
    For each station and each buffer radius, find images within the buffer
    and aggregate density features.
    """
    feature_cols = [col + variant for col in ALL_COLS for variant in DENSITY_VARIANTS]

    # project images once to EPSG:32632 for fast spatial ops
    gdf_img_32632 = gdf_img[['geometry'] + feature_cols].to_crs('EPSG:32632')
    gdf_st_32632  = gdf_st.to_crs('EPSG:32632')

    all_rows = []

    for b in BUFFERS:
        print(f"\nProcessing buffer {b}m...")
        buffers = gdf_st_32632.geometry.buffer(b)
        gdf_buf = gdf_st[['TU']].copy()
        gdf_buf['geometry'] = buffers.values
        gdf_buf = gpd.GeoDataFrame(gdf_buf, geometry='geometry', crs='EPSG:32632')

        # spatial join: images within station buffers
        joined = gpd.sjoin(gdf_img_32632, gdf_buf, how='inner', predicate='within')
        # joined has index = image idx, 'index_right' = station row index in gdf_buf

        for i, row in tqdm(gdf_st.iterrows(), total=len(gdf_st), desc=f"  buffer {b}m"):
            tu = row['TU']
            imgs = joined[joined['index_right'] == i]

            station_row = {'TU': tu, 'buffer_m': b, 'n_images': len(imgs)}

            if len(imgs) == 0:
                for col in feature_cols:
                    for agg in AGG_FUNCS:
                        station_row[f'{col}_{agg}'] = np.nan
            else:
                for col in feature_cols:
                    vals = imgs[col]
                    station_row[f'{col}_mean']   = vals.mean()
                    station_row[f'{col}_median'] = vals.median()
                    station_row[f'{col}_std']    = vals.std()
                    station_row[f'{col}_min']    = vals.min()
                    station_row[f'{col}_max']    = vals.max()

            all_rows.append(station_row)

    return pd.DataFrame(all_rows)


def main():
    gdf_st  = load_stations()
    gdf_img = load_images()
    df_out  = compute_buffer_features(gdf_st, gdf_img)

    df_out.to_csv(OUTPUT, index=False)
    print(f"\nSaved to: {OUTPUT.resolve()}")
    print(f"Shape: {df_out.shape}")
    print(f"Buffers: {sorted(df_out['buffer_m'].unique())}")
    print(f"Stations per buffer: {df_out.groupby('buffer_m')['TU'].nunique().to_dict()}")


if __name__ == '__main__':
    main()
