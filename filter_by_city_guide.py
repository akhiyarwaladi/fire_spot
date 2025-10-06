"""
FILTER DATA FIRE SPOTS BERDASARKAN WILAYAH
===========================================

DUA METODE FILTERING:
1. Bounding Box - Cepat, menggunakan kotak lat/lon
2. GeoJSON Polygon - Akurat, menggunakan batas wilayah sebenarnya

Author: Untuk keperluan pengajaran Data Mining
"""

import pandas as pd
import os
import json
import folium
from folium.plugins import HeatMap
from shapely.geometry import shape, Point

# =============================================================================
# WILAYAH INDONESIA DENGAN BOUNDING BOX
# =============================================================================

INDONESIA_REGIONS = {
    # SUMATERA
    'Aceh': {'lat': (2.5, 6.0), 'lon': (95.0, 98.5)},
    'Sumatera_Utara': {'lat': (1.0, 4.5), 'lon': (98.0, 100.5)},
    'Riau': {'lat': (-1.5, 2.5), 'lon': (100.0, 105.0)},
    'Kepulauan_Riau': {'lat': (-1.5, 2.0), 'lon': (103.0, 108.0)},
    'Sumatera_Barat': {'lat': (-3.5, 0.5), 'lon': (98.5, 102.0)},
    'Jambi': {'lat': (-3.0, 0.5), 'lon': (101.0, 105.0)},
    'Sumatera_Selatan': {'lat': (-5.0, -1.5), 'lon': (102.0, 106.0)},
    'Bengkulu': {'lat': (-5.5, -2.0), 'lon': (101.0, 104.0)},
    'Lampung': {'lat': (-6.0, -3.5), 'lon': (103.5, 106.0)},
    'Bangka_Belitung': {'lat': (-3.5, -1.5), 'lon': (105.0, 108.5)},

    # JAWA
    'Banten': {'lat': (-7.0, -5.5), 'lon': (105.0, 107.0)},
    'DKI_Jakarta': {'lat': (-6.5, -5.8), 'lon': (106.5, 107.2)},
    'Jawa_Barat': {'lat': (-8.0, -5.5), 'lon': (106.0, 109.0)},
    'Jawa_Tengah': {'lat': (-8.5, -6.0), 'lon': (108.5, 111.5)},
    'DI_Yogyakarta': {'lat': (-8.5, -7.5), 'lon': (110.0, 111.0)},
    'Jawa_Timur': {'lat': (-9.0, -6.5), 'lon': (111.0, 115.0)},

    # KALIMANTAN
    'Kalimantan_Barat': {'lat': (-3.5, 2.5), 'lon': (108.5, 112.5)},
    'Kalimantan_Tengah': {'lat': (-4.0, 0.5), 'lon': (111.0, 115.5)},
    'Kalimantan_Selatan': {'lat': (-4.5, -1.5), 'lon': (114.0, 117.0)},
    'Kalimantan_Timur': {'lat': (-2.5, 3.0), 'lon': (115.0, 119.5)},
    'Kalimantan_Utara': {'lat': (1.5, 4.5), 'lon': (115.5, 118.5)},

    # SULAWESI
    'Sulawesi_Utara': {'lat': (0.0, 2.5), 'lon': (123.5, 127.0)},
    'Gorontalo': {'lat': (0.0, 1.5), 'lon': (121.5, 123.5)},
    'Sulawesi_Tengah': {'lat': (-3.5, 1.5), 'lon': (119.5, 124.0)},
    'Sulawesi_Barat': {'lat': (-3.5, -1.5), 'lon': (118.5, 120.0)},
    'Sulawesi_Selatan': {'lat': (-7.0, -2.5), 'lon': (118.5, 121.5)},
    'Sulawesi_Tenggara': {'lat': (-6.0, -2.5), 'lon': (120.5, 124.0)},

    # BALI & NUSA TENGGARA
    'Bali': {'lat': (-8.8, -8.0), 'lon': (114.5, 115.8)},
    'NTB': {'lat': (-9.5, -8.0), 'lon': (115.5, 119.5)},
    'NTT': {'lat': (-11.0, -8.0), 'lon': (118.5, 125.0)},

    # MALUKU & PAPUA
    'Maluku': {'lat': (-9.0, -2.0), 'lon': (124.0, 132.0)},
    'Maluku_Utara': {'lat': (-2.5, 3.5), 'lon': (124.0, 129.5)},
    'Papua_Barat': {'lat': (-4.5, 1.5), 'lon': (130.0, 135.0)},
    'Papua': {'lat': (-9.5, 0.5), 'lon': (135.0, 141.5)},
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_name(name):
    """Normalize province name for matching"""
    return name.upper().replace(' ', '_')

def find_province_feature(province_name, geojson_data):
    """Find matching province feature in GeoJSON"""
    search_name = normalize_name(province_name)
    for feature in geojson_data['features']:
        geojson_name = normalize_name(feature['properties']['Propinsi'])
        if geojson_name == search_name or search_name in geojson_name or geojson_name in search_name:
            return feature
    return None

def add_boundary(map_obj, region_name, num_points, geojson_data, style_params=None):
    """Add boundary (polygon or rectangle) to map"""
    region_display = region_name.replace('_', ' ')

    if geojson_data:
        # Add GeoJSON polygon
        feature = find_province_feature(region_name, geojson_data)
        if feature:
            default_style = {
                'fillColor': 'transparent',
                'color': 'blue',
                'weight': 2,
                'fillOpacity': 0,
                'opacity': 0.7
            }
            style = {**default_style, **(style_params or {})}

            folium.GeoJson(
                feature,
                style_function=lambda x, s=style: s,
                tooltip=f"<b>{region_display}</b><br>{num_points:,} fire spots"
            ).add_to(map_obj)

    elif region_name in INDONESIA_REGIONS:
        # Add bounding box rectangle
        bounds = INDONESIA_REGIONS[region_name]
        min_lat, max_lat = bounds['lat']
        min_lon, max_lon = bounds['lon']

        default_color = 'black' if not style_params else 'red'
        color = style_params.get('color', default_color) if style_params else default_color
        weight = style_params.get('weight', 2) if style_params else 2

        folium.Rectangle(
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            popup=f"<b>{region_display}</b><br>{num_points:,} fire spots",
            color=color,
            fill=False,
            weight=weight,
            opacity=0.5
        ).add_to(map_obj)

def create_info_box(region_display, num_points, avg_frp, is_geojson):
    """Create info box HTML for map"""
    boundary_desc = "Blue line: Province polygon" if is_geojson else "Red box: Bounding box"
    method_name = "GeoJSON Polygon" if is_geojson else "Bounding Box"

    return f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 270px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px; border-radius: 5px">
        <h4 style="margin: 0 0 10px 0;">🔥 {region_display}</h4>
        <p style="margin: 5px 0;"><b>Method:</b> {method_name}</p>
        <p style="margin: 5px 0;"><b>Total:</b> {num_points:,} fire spots</p>
        <p style="margin: 5px 0;"><b>Avg FRP:</b> {avg_frp:.2f}</p>
        <p style="margin: 5px 0; font-size: 11px; color: #666;">
            Red dots: Fire spots<br>
            {boundary_desc}
        </p>
    </div>
    '''

# =============================================================================
# FILTER FUNCTIONS
# =============================================================================

def filter_by_region(df, region_name):
    """Filter dataframe berdasarkan bounding box wilayah"""
    if region_name not in INDONESIA_REGIONS:
        return None

    bounds = INDONESIA_REGIONS[region_name]
    min_lat, max_lat = bounds['lat']
    min_lon, max_lon = bounds['lon']

    filtered = df[
        (df['latitude'] >= min_lat) &
        (df['latitude'] <= max_lat) &
        (df['longitude'] >= min_lon) &
        (df['longitude'] <= max_lon)
    ].copy()

    return filtered

def filter_by_geojson(df, province_name, geojson_data):
    """Filter dataframe menggunakan polygon GeoJSON yang lebih akurat"""
    feature = find_province_feature(province_name, geojson_data)

    if not feature:
        print(f"   ⚠️  Provinsi '{province_name}' tidak ditemukan di GeoJSON")
        return None

    polygon = shape(feature['geometry'])

    print(f"   🔍 Filtering dengan polygon (ini bisa lebih lama)...")
    filtered = df[df.apply(lambda row: polygon.contains(Point(row['longitude'], row['latitude'])), axis=1)].copy()

    return filtered

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_overview_map(successful_regions, output_dir, geojson_data):
    """Create overview map with all regions"""
    print("\n📍 Creating overview map...")

    # Calculate center from all points
    all_lats, all_lons = [], []
    for r in successful_regions:
        df_temp = pd.read_csv(os.path.join(output_dir, f"{r['region']}.csv"))
        all_lats.extend(df_temp['latitude'].tolist())
        all_lons.extend(df_temp['longitude'].tolist())

    center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)] if all_lats else [-2.5, 118]

    # Create map
    map_overview = folium.Map(location=center, zoom_start=5, tiles='OpenStreetMap')

    # Add regions with different colors
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'darkred',
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white']

    for idx, r in enumerate(successful_regions[:15]):
        df_region = pd.read_csv(os.path.join(output_dir, f"{r['region']}.csv"))

        # Sample for performance
        df_sample = df_region.sample(n=min(500, len(df_region)), random_state=42)

        color = colors[idx % len(colors)]
        region_display = r['region'].replace('_', ' ')
        fg = folium.FeatureGroup(name=f"{region_display} ({r['num_points']:,} points)")

        # Add markers
        for _, row in df_sample.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                popup=f"<b>{region_display}</b><br>FRP: {row['frp']:.2f}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(fg)

        fg.add_to(map_overview)

    # Add boundaries
    for r in successful_regions[:15]:
        add_boundary(map_overview, r['region'], r['num_points'], geojson_data)

    folium.LayerControl().add_to(map_overview)

    # Save
    maps_dir = os.path.join(output_dir, 'maps')
    os.makedirs(maps_dir, exist_ok=True)
    overview_file = os.path.join(maps_dir, 'overview_all_regions.html')
    map_overview.save(overview_file)
    print(f"   ✓ Saved: {overview_file}")

    return maps_dir

def create_individual_maps(top_regions, output_dir, geojson_data):
    """Create individual maps for top regions"""
    print("\n📍 Creating individual region maps (top 5)...")

    maps_dir = os.path.join(output_dir, 'maps')

    for r in top_regions[:5]:
        df_region = pd.read_csv(os.path.join(output_dir, f"{r['region']}.csv"))
        region_display = r['region'].replace('_', ' ')

        # Create map centered on region
        center = [df_region['latitude'].mean(), df_region['longitude'].mean()]
        map_region = folium.Map(location=center, zoom_start=8, tiles='OpenStreetMap')

        # Add boundary
        style = {'color': 'blue', 'weight': 3} if geojson_data else {'color': 'red', 'weight': 3}
        add_boundary(map_region, r['region'], r['num_points'], geojson_data, style)

        # Add fire spots
        df_sample = df_region.sample(n=min(1000, len(df_region)), random_state=42)
        for _, row in df_sample.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                popup=f"<b>Fire Spot</b><br>Lat: {row['latitude']:.4f}<br>"
                      f"Lon: {row['longitude']:.4f}<br>FRP: {row['frp']:.2f}<br>Date: {row['acq_date']}",
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.7
            ).add_to(map_region)

        # Add heatmap
        heat_data = [[row['latitude'], row['longitude'], row['frp']] for _, row in df_region.iterrows()]
        HeatMap(heat_data, radius=15, blur=20).add_to(folium.FeatureGroup(name='Heatmap').add_to(map_region))

        folium.LayerControl().add_to(map_region)

        # Add info box
        info_html = create_info_box(region_display, r['num_points'], r['avg_frp'], bool(geojson_data))
        map_region.get_root().html.add_child(folium.Element(info_html))

        # Save
        map_file = os.path.join(maps_dir, f"{r['region']}.html")
        map_region.save(map_file)
        print(f"   ✓ {region_display}: {map_file}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def generate_filtered_datasets(regions_to_filter=None, method='bounding_box'):
    """
    Generate filtered datasets untuk wilayah yang dipilih

    Parameters:
    - regions_to_filter: List wilayah yang mau di-filter (None = semua)
    - method: 'bounding_box' atau 'geojson'
    """

    print("="*80)
    print(f"GENERATE FILTERED DATA PER WILAYAH - METHOD: {method.upper()}")
    print("="*80)
    print()

    # Load raw data
    print("📂 Loading raw data...")
    raw_data_path = 'data/raw/fire_nrt_J2V-C2_669817.csv'

    if not os.path.exists(raw_data_path):
        print(f"   ❌ File tidak ditemukan: {raw_data_path}")
        return

    df = pd.read_csv(raw_data_path)
    print(f"   ✓ Total data: {len(df):,} fire spots")
    print(f"   📅 Periode: {df['acq_date'].min()} - {df['acq_date'].max()}")
    print()

    # Load GeoJSON if needed
    geojson_data = None
    if method == 'geojson':
        geojson_path = 'data/geojson/indonesia-prov-38.json'
        if not os.path.exists(geojson_path):
            print(f"   ❌ GeoJSON tidak ditemukan: {geojson_path}")
            return

        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        print(f"   ✓ GeoJSON loaded: {len(geojson_data['features'])} provinsi")
        print()

    # Determine regions to process
    if regions_to_filter is None:
        if method == 'bounding_box':
            regions_to_filter = list(INDONESIA_REGIONS.keys())
        else:
            regions_to_filter = [f['properties']['Propinsi'].replace(' ', '_')
                               for f in geojson_data['features']]

    # Create output directory
    output_dir = f"data/filtered/method_{'1_bounding_box' if method == 'bounding_box' else '2_geojson'}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 Output: {output_dir}/")
    print(f"🗺️  Wilayah: {len(regions_to_filter)} wilayah")
    print("\n" + "="*80)
    print("PROCESSING...")
    print("="*80)

    # Filter and save each region
    results = []
    for i, region in enumerate(regions_to_filter, 1):
        print(f"\n[{i}/{len(regions_to_filter)}] {region.replace('_', ' ')}")

        # Filter
        df_region = filter_by_region(df, region) if method == 'bounding_box' else filter_by_geojson(df, region, geojson_data)

        if df_region is None or len(df_region) == 0:
            print(f"   ⚠️  {'Wilayah tidak ditemukan' if df_region is None else 'Tidak ada fire spots'}")
            if df_region is not None:
                results.append({'region': region, 'status': 'no_data', 'num_points': 0})
            continue

        # Save
        file_path = os.path.join(output_dir, f"{region}.csv")
        df_region.to_csv(file_path, index=False)

        # Stats
        num_points = len(df_region)
        avg_frp = df_region['frp'].mean()

        print(f"   ✓ {num_points:,} fire spots")
        print(f"   📊 Avg FRP: {avg_frp:.2f}")
        print(f"   📅 {df_region['acq_date'].min()} - {df_region['acq_date'].max()}")
        print(f"   💾 {region}.csv")

        results.append({
            'region': region,
            'status': 'success',
            'num_points': num_points,
            'avg_frp': avg_frp
        })

    # Summary
    successful = [r for r in results if r['status'] == 'success']
    no_data = [r for r in results if r['status'] == 'no_data']

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✅ Berhasil: {len(successful)} wilayah")
    print(f"⚠️  Tidak ada data: {len(no_data)} wilayah")

    if successful:
        print("\nTop 15 wilayah dengan fire spots terbanyak:")
        print("-" * 70)
        sorted_results = sorted(successful, key=lambda x: x['num_points'], reverse=True)[:15]
        for i, r in enumerate(sorted_results, 1):
            print(f"{i:2d}. {r['region'].replace('_', ' '):25s} : {r['num_points']:6,} fire spots (FRP: {r['avg_frp']:6.2f})")

    print(f"\n📁 Files saved to: {output_dir}/")

    # Generate visualizations
    if successful:
        print("\n" + "="*80)
        print("🗺️  GENERATING VISUALIZATION MAPS...")
        print("="*80)

        maps_dir = create_overview_map(successful, output_dir, geojson_data)
        create_individual_maps(sorted_results, output_dir, geojson_data)

        print(f"\n📁 Maps saved to: {maps_dir}/")
        print("\n📌 VISUALIZATION FILES:")
        print(f"   • overview_all_regions.html  - Lihat semua wilayah")
        print(f"   • [Region].html              - Lihat per wilayah detail")

    print("\n" + "="*80)
    print("✅ SELESAI!")
    print("="*80)
    print("\nNext:")
    print("  1. Buka maps di browser untuk verifikasi visual")
    print(f"     firefox {os.path.join(output_dir, 'maps')}/overview_all_regions.html")
    print("\n  2. Jalankan clustering untuk wilayah tertentu:")
    print("     Edit clustering_fire_spots.py")
    print("     DATA_SOURCE = 'data/filtered/method_1_bounding_box/Riau.csv'")
    print("     python clustering_fire_spots.py")
    print()

    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys

    method = 'geojson' if '--geojson' in sys.argv else 'bounding_box'
    if '--geojson' in sys.argv:
        sys.argv.remove('--geojson')

    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            generate_filtered_datasets(regions_to_filter=None, method=method)
        elif sys.argv[1] == '--sumatera':
            sumatera = ['Aceh', 'Sumatera_Utara', 'Riau', 'Kepulauan_Riau',
                       'Sumatera_Barat', 'Jambi', 'Sumatera_Selatan',
                       'Bengkulu', 'Lampung', 'Bangka_Belitung']
            generate_filtered_datasets(regions_to_filter=sumatera, method=method)
        elif sys.argv[1] == '--kalimantan':
            kalimantan = ['Kalimantan_Barat', 'Kalimantan_Tengah', 'Kalimantan_Selatan',
                         'Kalimantan_Timur', 'Kalimantan_Utara']
            generate_filtered_datasets(regions_to_filter=kalimantan, method=method)
        elif sys.argv[1] == '--help':
            print("="*80)
            print("FILTER FIRE SPOTS BY REGION")
            print("="*80)
            print("\nUsage:")
            print("  python filter_by_city_guide.py [options] [--geojson]")
            print("\nOptions:")
            print("  --all                   # All provinces")
            print("  --sumatera              # Sumatera only (10 provinces)")
            print("  --kalimantan            # Kalimantan only (5 provinces)")
            print("  --help                  # Show this help message")
            print("\nMethods:")
            print("  --geojson               # Use GeoJSON polygon (more accurate)")
            print("  (default)               # Use bounding box (faster)")
            print("\nExamples:")
            print("  python filter_by_city_guide.py --sumatera")
            print("  python filter_by_city_guide.py --sumatera --geojson")
            print("  python filter_by_city_guide.py --all --geojson")
            print()
        else:
            print("❌ Invalid option. Use --help for usage information")
    else:
        print("="*80)
        print("FILTER FIRE SPOTS BY REGION")
        print("="*80)
        print("\n⚠️  No options specified. Use one of:")
        print("  python filter_by_city_guide.py --sumatera")
        print("  python filter_by_city_guide.py --kalimantan")
        print("  python filter_by_city_guide.py --all")
        print("  python filter_by_city_guide.py --help")
        print("\nAdd --geojson for more accurate polygon-based filtering")
