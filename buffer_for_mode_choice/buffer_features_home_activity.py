#!/usr/bin/env python3
"""
Compute street view density features for each unique home and activity location
in the TU survey data, using circular buffers of 250, 500, 750, 1000m.

For each location and each buffer radius, street view images within the buffer
are aggregated with mean, median, std, min, max.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer


# ── paths ─────────────────────────────────────────────────────────────────────
TU_PATH   = '/work/lpsha/data/design_standard_for_public_transport/tu/df_basis_with_home_activity_coords_26_2_2.csv'
META_PATH = '/data/google_street_view/denmark_big/meta.csv'
DINO_PATH = '/data/google_street_view/denmark_big/DINOSAM_26c.csv'
OUTPUT_HOME     = Path(__file__).parent / 'home_buffer_features.csv'
OUTPUT_ACTIVITY = Path(__file__).parent / 'activity_buffer_features.csv'

BUFFERS   = [250, 500, 750, 1000]

# TU coordinates are ETRS89 / UTM zone 32N
TU_CRS = 'EPSG:25832'

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


def load_home_activity_locations():
    """Load unique home and activity locations from TU survey data."""
    df_tu = pd.read_csv(TU_PATH, sep=',')

    # ── home locations ────────────────────────────────────────────────────────
    df_home = df_tu[['HOME_e', 'HOME_n']].dropna().drop_duplicates().reset_index(drop=True)
    df_home['loc_id'] = df_home.index
    gdf_home = gpd.GeoDataFrame(
        df_home,
        geometry=gpd.points_from_xy(df_home['HOME_e'], df_home['HOME_n']),
        crs=TU_CRS,
    )

    # ── activity locations ────────────────────────────────────────────────────
    df_act = df_tu[['ACT_e', 'ACT_n']].dropna().drop_duplicates().reset_index(drop=True)
    df_act['loc_id'] = df_act.index
    gdf_act = gpd.GeoDataFrame(
        df_act,
        geometry=gpd.points_from_xy(df_act['ACT_e'], df_act['ACT_n']),
        crs=TU_CRS,
    )

    print(f"Unique home locations: {len(gdf_home)}")
    print(f"Unique activity locations: {len(gdf_act)}")
    return gdf_home, gdf_act


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


def compute_buffer_features(gdf_locs, gdf_img, coord_e_col, coord_n_col, label):
    """
    For each location and each buffer radius, find images within the buffer
    and aggregate density features.

    Parameters
    ----------
    gdf_locs : GeoDataFrame
        Locations with geometry in EPSG:25832.
    gdf_img : GeoDataFrame
        Street view images with geometry in EPSG:4326.
    coord_e_col : str
        Name of the easting column (for output).
    coord_n_col : str
        Name of the northing column (for output).
    label : str
        Label for progress bars.
    """
    feature_cols = [col + variant for col in ALL_COLS for variant in DENSITY_VARIANTS]

    # project images to EPSG:25832 to match TU coordinates
    gdf_img_25832 = gdf_img[['geometry'] + feature_cols].to_crs(TU_CRS)

    all_rows = []

    for b in BUFFERS:
        print(f"\nProcessing {label} buffer {b}m...")
        buffers = gdf_locs.geometry.buffer(b)
        gdf_buf = gdf_locs[[coord_e_col, coord_n_col, 'loc_id']].copy()
        gdf_buf['geometry'] = buffers.values
        gdf_buf = gpd.GeoDataFrame(gdf_buf, geometry='geometry', crs=TU_CRS)

        # spatial join: images within location buffers
        joined = gpd.sjoin(gdf_img_25832, gdf_buf, how='inner', predicate='within')

        for i, row in tqdm(gdf_locs.iterrows(), total=len(gdf_locs), desc=f"  {label} buffer {b}m"):
            loc_row = {
                coord_e_col: row[coord_e_col],
                coord_n_col: row[coord_n_col],
                'buffer_m': b,
                'n_images': 0,
            }

            imgs = joined[joined['index_right'] == i]
            loc_row['n_images'] = len(imgs)

            if len(imgs) == 0:
                for col in feature_cols:
                    for agg in AGG_FUNCS:
                        loc_row[f'{col}_{agg}'] = np.nan
            else:
                for col in feature_cols:
                    vals = imgs[col]
                    loc_row[f'{col}_mean']   = vals.mean()
                    loc_row[f'{col}_median'] = vals.median()
                    loc_row[f'{col}_std']    = vals.std()
                    loc_row[f'{col}_min']    = vals.min()
                    loc_row[f'{col}_max']    = vals.max()

            all_rows.append(loc_row)

    return pd.DataFrame(all_rows)


def main():
    gdf_home, gdf_act = load_home_activity_locations()
    gdf_img = load_images()

    print("\n=== Computing HOME buffer features ===")
    df_home = compute_buffer_features(gdf_home, gdf_img, 'HOME_e', 'HOME_n', 'HOME')
    df_home.to_csv(OUTPUT_HOME, index=False)
    print(f"\nSaved HOME features to: {OUTPUT_HOME.resolve()}")
    print(f"Shape: {df_home.shape}")
    print(f"Buffers: {sorted(df_home['buffer_m'].unique())}")
    print(f"Locations per buffer: {df_home.groupby('buffer_m')['HOME_e'].count().to_dict()}")

    print("\n=== Computing ACTIVITY buffer features ===")
    df_act = compute_buffer_features(gdf_act, gdf_img, 'ACT_e', 'ACT_n', 'ACTIVITY')
    df_act.to_csv(OUTPUT_ACTIVITY, index=False)
    print(f"\nSaved ACTIVITY features to: {OUTPUT_ACTIVITY.resolve()}")
    print(f"Shape: {df_act.shape}")
    print(f"Buffers: {sorted(df_act['buffer_m'].unique())}")
    print(f"Locations per buffer: {df_act.groupby('buffer_m')['ACT_e'].count().to_dict()}")


if __name__ == '__main__':
    main()
