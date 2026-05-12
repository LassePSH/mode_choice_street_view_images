from collections import Counter
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import osmium
from shapely.geometry import Point, Polygon, LineString
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

def get_poi(A,tag):
    osm_file = "/work/lpsha/data/OSM_denmark/denmark-latest_new.osm.pbf"
    values = []
    # for tag in ['amenity']:
    for o in osmium.FileProcessor(osm_file).with_filter(osmium.filter.KeyFilter(tag)).with_locations():
        if o.is_way():
            coords = [(n.lon, n.lat) for n in o.nodes if n.location.valid()]
            if coords:
                line = LineString(coords)
                if line.intersects(A):
                    values.append(o.tags.get(tag))

        if o.is_node() and o.location.valid():
            p = Point(o.location.lon, o.location.lat)
            if p.intersects(A):
                values.append(o.tags.get(tag))
    return values

def get_count(c,tag):
    vals = get_poi(c,tag)
    counts = Counter(vals)
    return counts

def compute_entropies(d, M):
    total = sum(d.values())
    if total == 0 or M <= 1:
        return 0.0
    probs = [count / total for count in d.values() if count > 0]
    e = -np.sum([p_i * np.log(p_i) for p_i in probs]) / np.log(M)
    return e

gtfs_path = '/home/lpsha/s154446/design_standard_public_transport/data/choice_model/GTFS_TU_StationList_v19_grouped.csv'
gdf = gpd.read_file('/home/lpsha/s154446/design_standard_public_transport/data/choice_model/GTFS_TU_StationList_v19_grouped.csv')
gdf['geometry'] = gdf.apply(lambda row: Point(row['stop_lon'], row['stop_lat']), axis=1)
gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
gdf.set_crs(epsg=4326, inplace=True)


for b in [250, 500, 1000]:
    gdf[f'area_{b}'] = gdf.geometry.to_crs('EPSG:32632').buffer(b).to_crs(epsg=4326)

    tqdm.pandas(desc=f'buffer {b}m')
    for tag in ['amenity', 'office', 'shop', 'building', 'leisure']:
        gdf[f'{tag}_count_{b}'] = gdf[f'area_{b}'].progress_apply(lambda a: get_count(a, tag))

    gdf[f'all_count_{b}'] = (gdf[f'amenity_count_{b}'] + gdf[f'office_count_{b}'] +
                             gdf[f'shop_count_{b}'] + gdf[f'building_count_{b}'] +
                             gdf[f'leisure_count_{b}'])
    M = 279 + 96 + 274 + 219 + 219  # {'amenity': 279, 'office': 96, 'shop': 274, 'building': 219, 'leisure': 219}
    gdf[f'entropy_all_{b}'] = gdf[f'all_count_{b}'].apply(lambda c: compute_entropies(c, M))

    gdf[f'amenity_count_filtered_{b}'] = gdf[f'amenity_count_{b}'].apply(lambda c: Counter({k: c[k] for k in keys_to_keep_amenity if k in c}))
    gdf[f'leisure_count_filtered_{b}'] = gdf[f'leisure_count_{b}'].apply(lambda c: Counter({k: c[k] for k in keys_to_keep_leisure if k in c}))
    gdf[f'building_count_filtered_{b}'] = gdf[f'building_count_{b}'].apply(lambda c: Counter({k: c[k] for k in keys_to_keep_building if k in c}))
    gdf[f'all_count_filtered_{b}'] = (gdf[f'amenity_count_filtered_{b}'] + gdf[f'office_count_{b}'] +
                                      gdf[f'shop_count_{b}'] + gdf[f'building_count_filtered_{b}'] +
                                      gdf[f'leisure_count_filtered_{b}'])
    M_filtered = 96 + 274 + len(keys_to_keep_building) + len(keys_to_keep_amenity) + len(keys_to_keep_leisure)
    gdf[f'entropy_filtered_{b}'] = gdf[f'all_count_filtered_{b}'].apply(lambda c: compute_entropies(c, M_filtered))

# save
path = '/home/lpsha/s154446/design_standard_public_transport/stations_POI/stations_with_poi_counts.csv'
gdf.drop(columns=[c for c in gdf.columns if c.startswith('area_')]).to_csv(path, index=False)