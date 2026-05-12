#!/usr/bin/env python3
"""
Process trajectories from home to station and activity to station, extracting street view features.

This script:
1. Loads GTFS station data, street view images, and TU travel survey data
2. Routes from home to nearest station for each trip
3. Routes from activity location to nearest station for each trip
4. Finds street view images along each route
5. Calculates aggregated density features (mean, median, std, min, max) for both legs
6. Saves results to CSV

Usage:
    python process_trajectories.py [--buffer BUFFER] [--output OUTPUT] [--limit LIMIT]

Example:
    # Test with 10 trips
    python process_trajectories.py --limit 10

    # Process all trips
    python process_trajectories.py --output results.csv
"""

import argparse
import sys
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyproj import Transformer
from sklearn import preprocessing
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

# Add src to path
sys.path.append(str(Path(__file__).parent))

from trajectory_analysis import get_trajectory_images, calculate_trajectory_features


def load_data():
    """Load all necessary data files."""
    print("Loading data...")

    # Load GTFS stations
    gtfs_path = '/home/lpsha/s154446/design_standard_public_transport/data/choice_model/GTFS_TU_StationList_v19_grouped.csv'
    df_gtfs = pd.read_csv(gtfs_path)
    gdf_st = gpd.GeoDataFrame(
        df_gtfs,
        geometry=gpd.points_from_xy(df_gtfs['stop_lon'], df_gtfs['stop_lat']),
        crs="EPSG:4326")
    print(f"  Loaded {len(gdf_st)} stations")

    # Load street view images with DINOSAM features
    meta = pd.read_csv('/data/google_street_view/denmark_big/meta.csv')
    dino = pd.read_csv('/data/google_street_view/denmark_big/DINOSAM_26c.csv')
    dino['id'] = dino['id'].apply(lambda x: x[:-4])
    dino.set_index('id', inplace=True)
    meta.set_index('id', inplace=True)
    merged = meta.join(dino, how='inner')

    # Create GeoDataFrame
    merged['point_geometry'] = gpd.points_from_xy(merged['lon'], merged['lat'])
    gdf_img = gpd.GeoDataFrame(merged, geometry='point_geometry', crs="EPSG:4326")
    gdf_img = gdf_img.to_crs(epsg=3857)

    # Calculate density features
    cols = ['NaN', 'sidewalk', 'sky', 'buildings', 'road', 'grass', 'trees',
            'plants', 'bushes', 'hedge', 'water', 'door', 'fencing', 'window',
            'graffiti', 'bench', 'street sign', 'pole', 'traffic light',
            'trash can', 'bicycles', 'scooter', 'motorcycle', 'car', 'bus',
            'truck', 'person']

    for col in cols:
        gdf_img[col+'_density'] = gdf_img[col] / (gdf_img[cols].sum(axis=1))

    # Filter out images with too much NaN
    gdf_img = gdf_img[gdf_img['NaN_density'] < gdf_img['NaN_density'].quantile(0.90)]


    gdf_img['greens_density'] = gdf_img[['grass_density', 'trees_density', 'plants_density', 'bushes_density', 'hedge_density']].sum(axis=1)
    gdf_img['building_window_ratio_density'] = gdf_img['window_density'] / (gdf_img['buildings_density'] + 1e-6)
    gdf_img['building_all_density'] = gdf_img['buildings_density'] + gdf_img['door_density'] + gdf_img['window_density'] 

    gdf_img['road_life_g1_density'] = gdf_img[['road_density','street sign_density','pole_density','traffic light_density','trash can_density',
                                                'scooter_density','motorcycle_density', 'car_density', 'bus_density', 'truck_density','bicycles_density']].sum(axis=1)
    gdf_img['urban_life_g1_density'] = gdf_img[['person_density', 'bench_density','sidewalk_density']].sum(axis=1)

    gdf_img['road_life_g2_density'] = gdf_img[['road_density','street sign_density','pole_density','traffic light_density','trash can_density',]].sum(axis=1)
    gdf_img['vehicle_life_g2_density'] = gdf_img[['scooter_density','motorcycle_density', 'car_density', 'bus_density', 'truck_density','bicycles_density']].sum(axis=1)
  
    cols = cols + ['greens', 'building_window_ratio', 'building_all', 'road_life_g1', 'urban_life_g1', 'road_life_g2', 'vehicle_life_g2']

    for col in cols:
        x = gdf_img[[col+'_density']]
        x_norm = preprocessing.normalize(x,)
        gdf_img[col+'_density_l2_norm'] = x_norm.flatten()

    scaler = MinMaxScaler()
    for col in cols:
        gdf_img[col+'_density_minmax'] = scaler.fit_transform(gdf_img[[col+'_density']])

    qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
    density_cols = [col + '_density' for col in cols]
    qt_data = qt.fit_transform(gdf_img[density_cols])
    qt_df = pd.DataFrame(qt_data,columns=[col + '_density_quantile' for col in cols],index=gdf_img.index)
    gdf_img = pd.concat([gdf_img, qt_df], axis=1)
    
    print(f"  Loaded {len(gdf_img)} street view images")

    # Load TU travel survey data
    tu_path = '/work/lpsha/data/design_standard_for_public_transport/tu/df_basis_with_home_activity_coords_26_2_2.csv'
    df = pd.read_csv(tu_path, sep=',')
    df = df.rename(columns={'HOME_e': 'home_lon', 'HOME_n': 'home_lat'})
    df = df.rename(columns={'ACT_e': 'act_lon', 'ACT_n': 'act_lat'})
    df['p_home'] = gpd.points_from_xy(df['home_lon'], df['home_lat'])
    df['p_act'] = gpd.points_from_xy(df['act_lon'], df['act_lat'])

    # Transform coordinates from EPSG:25832 to EPSG:4326
    transformer = Transformer.from_crs("epsg:25832", "epsg:4326", always_xy=True)
    df['p_act'] = df['p_act'].apply(lambda point: transformer.transform(point.x, point.y))
    df['p_home'] = df['p_home'].apply(lambda point: transformer.transform(point.x, point.y))

    # Convert tuples back to Points
    df['p_act'] = df['p_act'].apply(lambda xy: Point(xy))
    df['p_home'] = df['p_home'].apply(lambda xy: Point(xy))

    # Get station geometries
    df['home_station_geometry'] = df.HOME_NearestStation.apply(
        lambda x: gdf_st[gdf_st['TU']==x].geometry.values[0]
        if not gdf_st[gdf_st['TU']==x].empty else None
    )
    df['act_station_geometry'] = df.ACT_NearestStation.apply(
        lambda x: gdf_st[gdf_st['TU']==x].geometry.values[0]
        if not gdf_st[gdf_st['TU']==x].empty else None
    )

    # Filter out rows with missing station geometries
    org_size = len(df)
    df = df.dropna(subset=['home_station_geometry', 'act_station_geometry'])
    print(f"  Loaded {len(df)} valid trips")
    print(f"  non-valid rows: {org_size-len(df)}")

    return gdf_st, gdf_img, df


def process_trajectories(df, gdf_img, mode="foot", buffer_m=20, limit=None):
    """
    Process all trajectories and extract features.

    Parameters
    ----------
    df : DataFrame
        Trip data with home/station locations
    gdf_img : GeoDataFrame
        Street view images
    mode : str
        Transport mode
    buffer_m : float
        Buffer distance in meters
    limit : int, optional
        Maximum number of trips to process

    Returns
    -------
    DataFrame
        Features for each trajectory
    """
    if limit is not None:
        df = df.head(limit)
        print(f"Processing first {limit} trips...")
    else:
        print(f"Processing all {len(df)} trips...")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing routes"):
        try:
            # Process HOME to station
            result_home = get_trajectory_images(
                p_home=row['p_home'],
                home_station_geometry=row['home_station_geometry'],
                gdf_img=gdf_img,
                mode=mode,
                buffer_m=buffer_m
            )

            # Calculate features for home leg
            features_home = calculate_trajectory_features(result_home)

            # Prefix home features
            features = {f'home_{k}': v for k, v in features_home.items()}

            # Process ACTIVITY to station
            result_act = get_trajectory_images(
                p_home=row['p_act'],
                home_station_geometry=row['act_station_geometry'],
                gdf_img=gdf_img,
                mode=mode,
                buffer_m=buffer_m
            )

            # Calculate features for activity leg
            features_act = calculate_trajectory_features(result_act)
            # Prefix activity features
            features.update({f'act_{k}': v for k, v in features_act.items()})

            # Process HOME to ACTIVITY for each alternative mode (foot, bike, car)
            for alt_mode in ['foot', 'bike', 'car']:
                try:
                    result_h2a = get_trajectory_images(
                        p_home=row['p_home'],
                        home_station_geometry=row['p_act'],
                        gdf_img=gdf_img,
                        mode=alt_mode,
                        buffer_m=buffer_m
                    )
                    features_h2a = calculate_trajectory_features(result_h2a)
                    features.update({f'h2a_{alt_mode}_{k}': v for k, v in features_h2a.items()})
                except Exception as e:
                    print(f"\nError processing h2a_{alt_mode} for TurId {row['TurId']}: {e}")

            # Add trip identifiers
            features['SessionId'] = row['SessionId']
            features['TurId'] = row['TurId']
            features['HOME_NearestStation'] = row['HOME_NearestStation']
            features['ACT_NearestStation'] = row['ACT_NearestStation']
            features['mode'] = mode

            results.append(features)

        except Exception as e:
            print(f"\nError processing TurId {row['TurId']}: {e}")
            continue

    df_results = pd.DataFrame(results)
    print(f"\nSuccessfully processed {len(df_results)} trajectories")

    return df_results


def main():
    parser = argparse.ArgumentParser(description='Process home-to-station trajectories (foot mode only)')
    parser.add_argument('--buffer', type=float, default=20,
                        help='Buffer distance in meters (default: 20)')
    parser.add_argument('--output', type=str,
                        default='trajectory_features_full.csv',
                        help='Output CSV file (default: trajectory_features_full.csv)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of trips to process (default: all)')

    args = parser.parse_args()

    # Load data
    gdf_st, gdf_img, df = load_data()

    # Process trajectories (foot mode only)
    df_results = process_trajectories(
        df=df,
        gdf_img=gdf_img,
        mode='foot',
        buffer_m=args.buffer,
        limit=args.limit
    )

    # Save results
    output_path = Path(args.output)
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path.absolute()}")
    print(f"Shape: {df_results.shape}")
    print(f"Orginal shape: {df.shape}")
    # print(f"\nColumns: {list(df_results.columns)}")

    # Print summary statistics
    print("\nSummary (Foot mode):")
    print("  Home leg:")
    print(f"    Mean distance: {df_results['home_distance_m'].mean():.0f} m")
    print(f"    Mean duration: {df_results['home_duration_s'].mean():.0f} s")
    print(f"    Mean images per route: {df_results['home_n_images'].mean():.1f}")

    print("  Activity leg:")
    print(f"    Mean distance: {df_results['act_distance_m'].mean():.0f} m")
    print(f"    Mean duration: {df_results['act_duration_s'].mean():.0f} s")
    print(f"    Mean images per route: {df_results['act_n_images'].mean():.1f}")

    for alt_mode in ['foot', 'bike', 'car']:
        prefix = f'h2a_{alt_mode}'
        dist_col = f'{prefix}_distance_m'
        dur_col = f'{prefix}_duration_s'
        img_col = f'{prefix}_n_images'
        if dist_col in df_results.columns:
            print(f"  Home-to-Activity ({alt_mode}):")
            print(f"    Mean distance: {df_results[dist_col].mean():.0f} m")
            print(f"    Mean duration: {df_results[dur_col].mean():.0f} s")
            print(f"    Mean images per route: {df_results[img_col].mean():.1f}")


if __name__ == "__main__":
    main()
