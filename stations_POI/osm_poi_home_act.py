from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import osmium
from shapely.geometry import Point, LineString
import numpy as np

keys_to_keep_amenity = {'school',
    'university',
    'library',
    'college',
    'research_institute',
    'bank',
    'hospital',
    'clinic',
    'doctors',
    'pharmacy',
    'bar',
    'cafe',
    'food_court',
    'fast_food',
    'pub',
    'restaurant',
    'community_centre',
    'arts_centre',
    'cinema',
    'music_venue',
    'nightclub',
    'social_centre',
    'theatre',
    'exhibition_centre',
    'post_office',
    'fire_station',
    'police',
    'townhall',}

keys_to_keep_leisure =  {'park',
                        'garden',
                        'playground',
                        'dance',
                        'sports_hall',
                        'sports_centre',
                        'fitness_centre',
                        'fitness_station'}

keys_to_keep_building = {'office',
                        'commercial',
                        'retail',
                        'kiosk',
                        'supermarket'}


OSM_FILE = "/work/lpsha/data/OSM_denmark/denmark-latest_new.osm.pbf"
TU_PATH = '/work/lpsha/data/design_standard_for_public_transport/tu/df_basis_with_home_activity_coords_26_2_2.csv'
TU_CRS = 'EPSG:25832'
BUFFERS = [250, 500, 750, 1000]
TAGS = ['amenity', 'office', 'shop', 'building', 'leisure']
N_WORKERS = 10

OUTPUT_HOME = Path(__file__).parent / 'home_with_poi_counts.csv'
OUTPUT_ACTIVITY = Path(__file__).parent / 'activity_with_poi_counts.csv'


def extract_osm_tag(tag):
    """Scan the OSM file once and collect all features carrying `tag`."""
    records = []
    for o in osmium.FileProcessor(OSM_FILE).with_filter(osmium.filter.KeyFilter(tag)).with_locations():
        val = o.tags.get(tag)
        if val is None:
            continue
        if o.is_node():
            if o.location.valid():
                records.append((val, Point(o.location.lon, o.location.lat)))
        elif o.is_way():
            coords = [(n.lon, n.lat) for n in o.nodes if n.location.valid()]
            if len(coords) >= 2:
                records.append((val, LineString(coords)))
            elif len(coords) == 1:
                records.append((val, Point(coords[0])))
    if not records:
        return tag, gpd.GeoDataFrame({'value': [], 'geometry': []}, crs='EPSG:4326')
    vals, geoms = zip(*records)
    gdf = gpd.GeoDataFrame({'value': list(vals)}, geometry=list(geoms), crs='EPSG:4326')
    return tag, gdf


def compute_entropies(d, M):
    total = sum(d.values())
    if total == 0 or M <= 1:
        return 0.0
    probs = [count / total for count in d.values() if count > 0]
    e = -np.sum([p_i * np.log(p_i) for p_i in probs]) / np.log(M)
    return e


def load_home_activity_locations():
    df_tu = pd.read_csv(TU_PATH, sep=',')

    df_home = df_tu[['HOME_e', 'HOME_n']].dropna().drop_duplicates().reset_index(drop=True)
    df_home['loc_id'] = df_home.index
    gdf_home = gpd.GeoDataFrame(
        df_home,
        geometry=gpd.points_from_xy(df_home['HOME_e'], df_home['HOME_n']),
        crs=TU_CRS,
    )

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


def counts_per_location(gdf_buf, gdf_tag_proj, n_locs):
    """Return a list of Counters, one per loc_id (0..n_locs-1)."""
    joined = gpd.sjoin(
        gdf_tag_proj[['value', 'geometry']],
        gdf_buf[['loc_id', 'geometry']],
        predicate='intersects',
        how='inner',
    )
    counts = [Counter() for _ in range(n_locs)]
    if len(joined) == 0:
        return counts
    grouped = joined.groupby(['loc_id', 'value']).size()
    for (loc_id, val), n in grouped.items():
        counts[int(loc_id)][val] = int(n)
    return counts


def compute_poi_features(gdf_locs, tag_gdfs, label):
    """Compute POI counts + entropy per buffer for each location."""
    n = len(gdf_locs)
    gdf = gdf_locs.drop(columns='geometry').copy()

    # pre-project tag GeoDataFrames to TU_CRS once
    tag_gdfs_proj = {tag: g.to_crs(TU_CRS) for tag, g in tag_gdfs.items()}

    for b in tqdm(BUFFERS, desc=f'{label} buffers'):
        gdf_buf = gpd.GeoDataFrame(
            {'loc_id': gdf_locs['loc_id'].values},
            geometry=gdf_locs.geometry.buffer(b).values,
            crs=TU_CRS,
        )

        for tag in TAGS:
            gdf[f'{tag}_count_{b}'] = counts_per_location(gdf_buf, tag_gdfs_proj[tag], n)

        gdf[f'all_count_{b}'] = (gdf[f'amenity_count_{b}'] + gdf[f'office_count_{b}'] +
                                 gdf[f'shop_count_{b}'] + gdf[f'building_count_{b}'] +
                                 gdf[f'leisure_count_{b}'])
        M = 279 + 96 + 274 + 219 + 219
        gdf[f'entropy_all_{b}'] = gdf[f'all_count_{b}'].apply(lambda c: compute_entropies(c, M))

        gdf[f'amenity_count_filtered_{b}'] = gdf[f'amenity_count_{b}'].apply(
            lambda c: Counter({k: c[k] for k in keys_to_keep_amenity if k in c}))
        gdf[f'leisure_count_filtered_{b}'] = gdf[f'leisure_count_{b}'].apply(
            lambda c: Counter({k: c[k] for k in keys_to_keep_leisure if k in c}))
        gdf[f'building_count_filtered_{b}'] = gdf[f'building_count_{b}'].apply(
            lambda c: Counter({k: c[k] for k in keys_to_keep_building if k in c}))
        gdf[f'all_count_filtered_{b}'] = (gdf[f'amenity_count_filtered_{b}'] + gdf[f'office_count_{b}'] +
                                          gdf[f'shop_count_{b}'] + gdf[f'building_count_filtered_{b}'] +
                                          gdf[f'leisure_count_filtered_{b}'])
        M_filtered = 96 + 274 + len(keys_to_keep_building) + len(keys_to_keep_amenity) + len(keys_to_keep_leisure)
        gdf[f'entropy_filtered_{b}'] = gdf[f'all_count_filtered_{b}'].apply(
            lambda c: compute_entropies(c, M_filtered))

        # replace Counter columns with their sums
        counter_cols = [f'{t}_count_{b}' for t in TAGS] + [f'all_count_{b}',
            f'amenity_count_filtered_{b}', f'leisure_count_filtered_{b}',
            f'building_count_filtered_{b}', f'all_count_filtered_{b}']
        for c in counter_cols:
            gdf[c] = gdf[c].apply(lambda x: sum(x.values()))

    return gdf


def main():
    gdf_home, gdf_act = load_home_activity_locations()

    print(f"\nExtracting OSM features for tags {TAGS} using {min(N_WORKERS, len(TAGS))} workers...")
    with Pool(processes=min(N_WORKERS, len(TAGS))) as pool:
        tag_gdfs = dict(pool.map(extract_osm_tag, TAGS))
    for t, g in tag_gdfs.items():
        print(f"  {t}: {len(g)} features")

    df_tu = pd.read_csv(TU_PATH, sep=',')

    print("\n=== Computing HOME POI features ===")
    home_feat = compute_poi_features(gdf_home, tag_gdfs, 'HOME')
    home_feat = home_feat.drop(columns=['loc_id'])
    out_h = df_tu[['SessionId', 'TurId', 'HOME_e', 'HOME_n']].merge(
        home_feat, on=['HOME_e', 'HOME_n'], how='right')
    out_h = out_h.drop(columns=['HOME_e', 'HOME_n'])
    out_h.to_csv(OUTPUT_HOME, index=False)
    print(f"Saved HOME features to: {OUTPUT_HOME.resolve()}  shape={out_h.shape}")

    print("\n=== Computing ACTIVITY POI features ===")
    act_feat = compute_poi_features(gdf_act, tag_gdfs, 'ACTIVITY')
    act_feat = act_feat.drop(columns=['loc_id'])
    out_a = df_tu[['SessionId', 'TurId', 'ACT_e', 'ACT_n']].merge(
        act_feat, on=['ACT_e', 'ACT_n'], how='right')
    out_a = out_a.drop(columns=['ACT_e', 'ACT_n'])
    out_a.to_csv(OUTPUT_ACTIVITY, index=False)
    print(f"Saved ACTIVITY features to: {OUTPUT_ACTIVITY.resolve()}  shape={out_a.shape}")


if __name__ == '__main__':
    main()
