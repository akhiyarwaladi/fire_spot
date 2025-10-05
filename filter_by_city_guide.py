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
# FUNGSI FILTER
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
    # Cari provinsi di GeoJSON
    province_feature = None
    for feature in geojson_data['features']:
        # Normalize nama provinsi untuk matching
        geojson_name = feature['properties']['Propinsi'].upper().replace(' ', '_')
        search_name = province_name.upper().replace(' ', '_')

        # Coba berbagai variasi nama
        if geojson_name == search_name or \
           search_name in geojson_name or \
           geojson_name in search_name:
            province_feature = feature
            break

    if province_feature is None:
        print(f"   ⚠️  Provinsi '{province_name}' tidak ditemukan di GeoJSON")
        return None

    # Buat polygon dari GeoJSON
    polygon = shape(province_feature['geometry'])

    # Filter fire spots yang ada di dalam polygon
    def is_inside(row):
        point = Point(row['longitude'], row['latitude'])
        return polygon.contains(point)

    # Apply filter
    print(f"   🔍 Filtering dengan polygon (ini bisa lebih lama)...")
    filtered = df[df.apply(is_inside, axis=1)].copy()

    return filtered

# =============================================================================
# FUNGSI UTAMA
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

    # Load GeoJSON if using geojson method
    geojson_data = None
    if method == 'geojson':
        geojson_path = 'data/geojson/indonesia-prov-38.json'
        if not os.path.exists(geojson_path):
            print(f"   ❌ GeoJSON tidak ditemukan: {geojson_path}")
            print(f"   💡 Download dulu dengan:")
            print(f"      curl -L https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson -o {geojson_path}")
            return

        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        print(f"   ✓ GeoJSON loaded: {len(geojson_data['features'])} provinsi (Updated 2023)")
        print()

    # Determine regions
    if regions_to_filter is None:
        if method == 'bounding_box':
            regions_to_filter = list(INDONESIA_REGIONS.keys())
        else:
            # Untuk GeoJSON, ambil dari file GeoJSON
            regions_to_filter = [f['properties']['Propinsi'].replace(' ', '_')
                                for f in geojson_data['features']]

    # Create output directory
    if method == 'bounding_box':
        output_dir = 'data/filtered/method_1_bounding_box'
    else:
        output_dir = 'data/filtered/method_2_geojson'

    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 Output: {output_dir}/")
    print(f"🗺️  Wilayah: {len(regions_to_filter)} wilayah")
    print()

    # Filter and save
    print("="*80)
    print("PROCESSING...")
    print("="*80)

    results = []
    for i, region in enumerate(regions_to_filter, 1):
        print(f"\n[{i}/{len(regions_to_filter)}] {region.replace('_', ' ')}")

        # Filter based on method
        if method == 'bounding_box':
            df_region = filter_by_region(df, region)
        else:
            df_region = filter_by_geojson(df, region, geojson_data)

        if df_region is None:
            print(f"   ⚠️  Wilayah tidak ditemukan")
            continue

        num_points = len(df_region)

        if num_points == 0:
            print(f"   ⚠️  Tidak ada fire spots")
            results.append({'region': region, 'status': 'no_data', 'num_points': 0})
            continue

        # Save
        filename = f"{region}.csv"
        file_path = os.path.join(output_dir, filename)
        df_region.to_csv(file_path, index=False)

        # Stats
        avg_frp = df_region['frp'].mean()
        date_range = f"{df_region['acq_date'].min()} - {df_region['acq_date'].max()}"

        print(f"   ✓ {num_points:,} fire spots")
        print(f"   📊 Avg FRP: {avg_frp:.2f}")
        print(f"   📅 {date_range}")
        print(f"   💾 {filename}")

        results.append({
            'region': region,
            'status': 'success',
            'num_points': num_points,
            'avg_frp': avg_frp
        })

    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)

    successful = [r for r in results if r['status'] == 'success']
    no_data = [r for r in results if r['status'] == 'no_data']

    print(f"\n✅ Berhasil: {len(successful)} wilayah")
    print(f"⚠️  Tidak ada data: {len(no_data)} wilayah")
    print()

    if successful:
        print("Top 15 wilayah dengan fire spots terbanyak:")
        print("-" * 70)
        sorted_results = sorted(successful, key=lambda x: x['num_points'], reverse=True)[:15]
        for i, r in enumerate(sorted_results, 1):
            region_display = r['region'].replace('_', ' ')
            print(f"{i:2d}. {region_display:25s} : {r['num_points']:6,} fire spots (FRP: {r['avg_frp']:6.2f})")

    print()
    print(f"📁 Files saved to: {output_dir}/")

    # ==========================================================================
    # VISUALISASI FOLIUM
    # ==========================================================================

    if successful:
        print()
        print("="*80)
        print("🗺️  GENERATING VISUALIZATION MAPS...")
        print("="*80)

        # Create maps folder inside the method folder
        maps_dir = os.path.join(output_dir, 'maps')
        os.makedirs(maps_dir, exist_ok=True)
        print(f"\n📁 Maps directory: {maps_dir}/")

        # Load GeoJSON if using geojson method (for polygon visualization)
        geojson_for_viz = None
        if method == 'geojson' and geojson_data:
            geojson_for_viz = geojson_data
            print(f"   ✓ Using GeoJSON polygons for boundary visualization")

        # Map 1: Overview Map - All regions in one map
        print("\n📍 Creating overview map...")

        # Calculate center
        all_lats = []
        all_lons = []
        for r in successful:
            region_file = os.path.join(output_dir, f"{r['region']}.csv")
            if os.path.exists(region_file):
                df_temp = pd.read_csv(region_file)
                all_lats.extend(df_temp['latitude'].tolist())
                all_lons.extend(df_temp['longitude'].tolist())

        if all_lats and all_lons:
            center_lat = sum(all_lats) / len(all_lats)
            center_lon = sum(all_lons) / len(all_lons)
        else:
            center_lat, center_lon = -2.5, 118  # Center of Indonesia

        # Create overview map
        map_overview = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )

        # Colors for regions
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
                  'pink', 'darkred', 'lightred', 'beige', 'darkblue',
                  'darkgreen', 'cadetblue', 'darkpurple', 'white']

        # Add each region to overview map
        for idx, r in enumerate(successful[:15]):  # Limit to top 15 for performance
            region_file = os.path.join(output_dir, f"{r['region']}.csv")
            if not os.path.exists(region_file):
                continue

            df_region = pd.read_csv(region_file)

            # Sample for performance (max 500 points per region)
            if len(df_region) > 500:
                df_region = df_region.sample(n=500, random_state=42)

            color = colors[idx % len(colors)]
            region_display = r['region'].replace('_', ' ')

            # Create feature group
            fg = folium.FeatureGroup(name=f"{region_display} ({r['num_points']:,} points)")

            # Add markers
            for _, row in df_region.iterrows():
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

        # Add boundaries (polygons for GeoJSON, rectangles for bounding box)
        for r in successful[:15]:
            region_display = r['region'].replace('_', ' ')

            if geojson_for_viz:
                # Draw actual province polygons from GeoJSON
                for feature in geojson_for_viz['features']:
                    geojson_name = feature['properties']['Propinsi'].upper().replace(' ', '_')
                    search_name = r['region'].upper().replace(' ', '_')

                    if geojson_name == search_name or search_name in geojson_name or geojson_name in search_name:
                        folium.GeoJson(
                            feature,
                            name=f"Boundary: {region_display}",
                            style_function=lambda x: {
                                'fillColor': 'transparent',
                                'color': 'blue',
                                'weight': 2,
                                'fillOpacity': 0,
                                'opacity': 0.7
                            },
                            tooltip=f"<b>{region_display}</b><br>{r['num_points']:,} fire spots"
                        ).add_to(map_overview)
                        break
            elif r['region'] in INDONESIA_REGIONS:
                # Draw bounding box rectangles
                bounds = INDONESIA_REGIONS[r['region']]
                min_lat, max_lat = bounds['lat']
                min_lon, max_lon = bounds['lon']

                folium.Rectangle(
                    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                    popup=f"<b>{region_display}</b><br>{r['num_points']:,} fire spots",
                    color='black',
                    fill=False,
                    weight=2,
                    opacity=0.5
                ).add_to(map_overview)

        # Add layer control
        folium.LayerControl().add_to(map_overview)

        # Save overview map
        overview_file = os.path.join(maps_dir, 'overview_all_regions.html')
        map_overview.save(overview_file)
        print(f"   ✓ Saved: {overview_file}")

        # Map 2: Individual maps for top 5 regions
        print("\n📍 Creating individual region maps (top 5)...")

        for r in sorted_results[:5]:
            region_file = os.path.join(output_dir, f"{r['region']}.csv")
            if not os.path.exists(region_file):
                continue

            df_region = pd.read_csv(region_file)
            region_display = r['region'].replace('_', ' ')

            # Calculate center for this region
            reg_center_lat = df_region['latitude'].mean()
            reg_center_lon = df_region['longitude'].mean()

            # Create map
            map_region = folium.Map(
                location=[reg_center_lat, reg_center_lon],
                zoom_start=8,
                tiles='OpenStreetMap'
            )

            # Add boundary (polygon for GeoJSON, rectangle for bounding box)
            if geojson_for_viz:
                # Draw actual province polygon from GeoJSON
                for feature in geojson_for_viz['features']:
                    geojson_name = feature['properties']['Propinsi'].upper().replace(' ', '_')
                    search_name = r['region'].upper().replace(' ', '_')

                    if geojson_name == search_name or search_name in geojson_name or geojson_name in search_name:
                        folium.GeoJson(
                            feature,
                            name=f"Province Boundary: {region_display}",
                            style_function=lambda x: {
                                'fillColor': 'transparent',
                                'color': 'blue',
                                'weight': 3,
                                'fillOpacity': 0
                            },
                            tooltip=f"<b>Province Boundary</b><br>{region_display}"
                        ).add_to(map_region)
                        break
            elif r['region'] in INDONESIA_REGIONS:
                # Draw bounding box rectangle
                bounds = INDONESIA_REGIONS[r['region']]
                min_lat, max_lat = bounds['lat']
                min_lon, max_lon = bounds['lon']

                folium.Rectangle(
                    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                    popup=f"<b>Bounding Box</b><br>{region_display}",
                    color='red',
                    fill=False,
                    weight=3
                ).add_to(map_region)

            # Sample points for performance
            df_sample = df_region.sample(n=min(1000, len(df_region)), random_state=42)

            # Add fire spots
            for _, row in df_sample.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=4,
                    popup=f"<b>Fire Spot</b><br>"
                          f"Lat: {row['latitude']:.4f}<br>"
                          f"Lon: {row['longitude']:.4f}<br>"
                          f"FRP: {row['frp']:.2f}<br>"
                          f"Date: {row['acq_date']}",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7
                ).add_to(map_region)

            # Add heatmap layer
            heat_data = [[row['latitude'], row['longitude'], row['frp']]
                        for _, row in df_region.iterrows()]
            HeatMap(heat_data, radius=15, blur=20).add_to(folium.FeatureGroup(name='Heatmap').add_to(map_region))

            # Add layer control
            folium.LayerControl().add_to(map_region)

            # Add info box
            boundary_desc = "Blue line: Province polygon" if geojson_for_viz else "Red box: Bounding box"
            method_name = "GeoJSON Polygon" if geojson_for_viz else "Bounding Box"

            info_html = f'''
            <div style="position: fixed; top: 10px; right: 10px; width: 270px;
                        background-color: white; border:2px solid grey; z-index:9999;
                        font-size:14px; padding: 10px; border-radius: 5px">
                <h4 style="margin: 0 0 10px 0;">🔥 {region_display}</h4>
                <p style="margin: 5px 0;"><b>Method:</b> {method_name}</p>
                <p style="margin: 5px 0;"><b>Total:</b> {r['num_points']:,} fire spots</p>
                <p style="margin: 5px 0;"><b>Avg FRP:</b> {r['avg_frp']:.2f}</p>
                <p style="margin: 5px 0; font-size: 11px; color: #666;">
                    Red dots: Fire spots<br>
                    {boundary_desc}
                </p>
            </div>
            '''
            map_region.get_root().html.add_child(folium.Element(info_html))

            # Save
            region_map_file = os.path.join(maps_dir, f"{r['region']}.html")
            map_region.save(region_map_file)
            print(f"   ✓ {region_display}: {region_map_file}")

        print()
        print(f"📁 Maps saved to: {maps_dir}/")
        print()
        print("📌 VISUALIZATION FILES:")
        print(f"   • overview_all_regions.html  - Lihat semua wilayah")
        print(f"   • [Region].html              - Lihat per wilayah detail")

    print()
    print("="*80)
    print("✅ SELESAI!")
    print("="*80)
    print()
    print("Next:")
    print("  1. Buka maps di browser untuk verifikasi visual")
    print(f"     firefox {maps_dir}/overview_all_regions.html")
    print()
    print("  2. Jalankan clustering untuk wilayah tertentu:")
    print("     Edit clustering_fire_spots.py")
    print("     DATA_SOURCE = 'data/filtered/method_1_bounding_box/Riau.csv'")
    print("     python clustering_fire_spots.py")
    print()

    return results

# Interactive mode removed for simplicity - use command line flags instead

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys

    # Default method
    method = 'bounding_box'

    # Check for method flag
    if '--geojson' in sys.argv:
        method = 'geojson'
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
            print()
            print("Usage:")
            print("  python filter_by_city_guide.py [options] [--geojson]")
            print()
            print("Options:")
            print("  --all                   # All provinces")
            print("  --sumatera              # Sumatera only (10 provinces)")
            print("  --kalimantan            # Kalimantan only (5 provinces)")
            print("  --help                  # Show this help message")
            print()
            print("Methods:")
            print("  --geojson               # Use GeoJSON polygon (more accurate)")
            print("  (default)               # Use bounding box (faster)")
            print()
            print("Examples:")
            print("  python filter_by_city_guide.py --sumatera")
            print("  python filter_by_city_guide.py --sumatera --geojson")
            print("  python filter_by_city_guide.py --all --geojson")
            print()
        else:
            print("❌ Invalid option. Use --help for usage information")
    else:
        # Default: show help
        print("="*80)
        print("FILTER FIRE SPOTS BY REGION")
        print("="*80)
        print("\n⚠️  No options specified. Use one of:")
        print("  python filter_by_city_guide.py --sumatera")
        print("  python filter_by_city_guide.py --kalimantan")
        print("  python filter_by_city_guide.py --all")
        print("  python filter_by_city_guide.py --help")
        print("\nAdd --geojson for more accurate polygon-based filtering")
