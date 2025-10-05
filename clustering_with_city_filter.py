"""
CLUSTERING FIRE SPOTS DENGAN FILTER PER KOTA
=============================================

Program ini menggabungkan:
1. Filter data berdasarkan kota tertentu
2. Clustering K-Means
3. Visualisasi dengan Folium

CARA PAKAI:
-----------
1. Ganti CITY_NAME di bawah dengan kota yang Anda inginkan
2. Jalankan script ini
3. Lihat hasil clustering khusus untuk kota tersebut

Author: Untuk keperluan pengajaran Data Mining
"""

# =============================================================================
# KONFIGURASI - UBAH DI SINI
# =============================================================================

# Pilih kota yang ingin dianalisis
CITY_NAME = 'Jakarta'  # Ubah sesuai keinginan

# Jumlah cluster (K)
NUM_CLUSTERS = 3

# Sampling (None untuk semua data)
SAMPLE_SIZE = None  # Set None untuk menggunakan semua data kota

# =============================================================================
# IMPORT LIBRARY
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CITY BOUNDARIES DATABASE
# =============================================================================

INDONESIA_CITIES = {
    # Pulau Jawa
    'Jakarta': {'lat': (-6.35, -6.08), 'lon': (106.65, 106.98), 'zoom': 11},
    'Bogor': {'lat': (-6.65, -6.50), 'lon': (106.75, 106.85), 'zoom': 12},
    'Depok': {'lat': (-6.45, -6.35), 'lon': (106.75, 106.90), 'zoom': 12},
    'Tangerang': {'lat': (-6.25, -6.10), 'lon': (106.55, 106.75), 'zoom': 12},
    'Bekasi': {'lat': (-6.30, -6.15), 'lon': (106.90, 107.10), 'zoom': 12},
    'Bandung': {'lat': (-7.05, -6.80), 'lon': (107.50, 107.75), 'zoom': 12},
    'Semarang': {'lat': (-7.05, -6.90), 'lon': (110.35, 110.50), 'zoom': 12},
    'Yogyakarta': {'lat': (-7.90, -7.75), 'lon': (110.30, 110.45), 'zoom': 12},
    'Surabaya': {'lat': (-7.35, -7.15), 'lon': (112.65, 112.85), 'zoom': 12},
    'Malang': {'lat': (-8.00, -7.90), 'lon': (112.60, 112.70), 'zoom': 12},

    # Pulau Sumatera
    'Medan': {'lat': (3.45, 3.65), 'lon': (98.60, 98.75), 'zoom': 12},
    'Pekanbaru': {'lat': (0.45, 0.60), 'lon': (101.35, 101.50), 'zoom': 12},
    'Padang': {'lat': (-1.05, -0.85), 'lon': (100.30, 100.45), 'zoom': 12},
    'Palembang': {'lat': (-3.05, -2.85), 'lon': (104.65, 104.85), 'zoom': 12},
    'Jambi': {'lat': (-1.65, -1.50), 'lon': (103.55, 103.70), 'zoom': 12},
    'Bengkulu': {'lat': (-3.85, -3.75), 'lon': (102.25, 102.35), 'zoom': 12},
    'Bandar Lampung': {'lat': (-5.50, -5.35), 'lon': (105.20, 105.35), 'zoom': 12},
    'Banda Aceh': {'lat': (5.50, 5.60), 'lon': (95.30, 95.40), 'zoom': 12},

    # Pulau Kalimantan
    'Pontianak': {'lat': (-0.10, 0.10), 'lon': (109.25, 109.40), 'zoom': 12},
    'Palangkaraya': {'lat': (-2.30, -2.15), 'lon': (113.85, 114.00), 'zoom': 12},
    'Banjarmasin': {'lat': (-3.35, -3.25), 'lon': (114.55, 114.65), 'zoom': 12},
    'Balikpapan': {'lat': (-1.35, -1.15), 'lon': (116.75, 116.95), 'zoom': 12},
    'Samarinda': {'lat': (-0.60, -0.40), 'lon': (117.10, 117.20), 'zoom': 12},

    # Pulau Sulawesi
    'Makassar': {'lat': (-5.20, -5.05), 'lon': (119.35, 119.50), 'zoom': 12},
    'Manado': {'lat': (1.45, 1.50), 'lon': (124.80, 124.90), 'zoom': 12},
    'Palu': {'lat': (-0.95, -0.85), 'lon': (119.85, 119.95), 'zoom': 12},
    'Kendari': {'lat': (-4.00, -3.95), 'lon': (122.50, 122.60), 'zoom': 12},

    # Bali & Nusa Tenggara
    'Denpasar': {'lat': (-8.75, -8.60), 'lon': (115.15, 115.30), 'zoom': 12},
    'Mataram': {'lat': (-8.65, -8.55), 'lon': (116.05, 116.15), 'zoom': 12},
    'Kupang': {'lat': (-10.25, -10.15), 'lon': (123.55, 123.65), 'zoom': 12},

    # Maluku & Papua
    'Ambon': {'lat': (-3.75, -3.65), 'lon': (128.15, 128.25), 'zoom': 12},
    'Jayapura': {'lat': (-2.65, -2.50), 'lon': (140.65, 140.75), 'zoom': 12},
}

# =============================================================================
# FUNGSI FILTER
# =============================================================================

def filter_by_city(df, city_name):
    """Filter dataframe berdasarkan city boundary"""
    if city_name not in INDONESIA_CITIES:
        print(f"❌ Error: Kota '{city_name}' tidak ditemukan!")
        print(f"\n📍 Kota yang tersedia ({len(INDONESIA_CITIES)}):")
        for city in sorted(INDONESIA_CITIES.keys()):
            print(f"   - {city}")
        return None

    city_bounds = INDONESIA_CITIES[city_name]
    min_lat, max_lat = city_bounds['lat']
    min_lon, max_lon = city_bounds['lon']

    filtered = df[
        (df['latitude'] >= min_lat) &
        (df['latitude'] <= max_lat) &
        (df['longitude'] >= min_lon) &
        (df['longitude'] <= max_lon)
    ].copy()

    return filtered

# =============================================================================
# MAIN PROGRAM
# =============================================================================

print("="*80)
print(f"CLUSTERING FIRE SPOTS - {CITY_NAME.upper()}")
print("="*80)
print()

# 1. Load data
print("📂 Loading data...")
df = pd.read_csv('fire_nrt_J2V-C2_669817.csv')
print(f"   Total data: {len(df):,} fire spots")
print()

# 2. Filter by city
print(f"🔍 Filtering data untuk kota: {CITY_NAME}")
df_city = filter_by_city(df, CITY_NAME)

if df_city is None:
    print("\n⚠️  Program dihentikan karena kota tidak ditemukan.")
    print("    Silakan ubah CITY_NAME di bagian KONFIGURASI.")
    exit()

if len(df_city) == 0:
    print(f"\n⚠️  Tidak ada fire spots di {CITY_NAME}")
    print("    Coba kota lain atau cek data Anda.")
    exit()

print(f"   ✓ Ditemukan {len(df_city):,} fire spots di {CITY_NAME}")
print(f"   📅 Periode: {df_city['acq_date'].min()} - {df_city['acq_date'].max()}")
print()

# 3. Sampling jika perlu
if SAMPLE_SIZE and len(df_city) > SAMPLE_SIZE:
    df_city = df_city.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"   📊 Data di-sampling: {len(df_city):,} samples")
    print()

# 4. Preprocessing
print("🔧 Preprocessing...")
df_clean = df_city[['latitude', 'longitude', 'brightness', 'frp']].dropna()
print(f"   ✓ Data clean: {len(df_clean):,} rows")

# Normalisasi
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_clean)
print(f"   ✓ Data normalized")
print()

# 5. Clustering
print(f"🤖 Running K-Means clustering (K={NUM_CLUSTERS})...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)
df_clean['cluster'] = clusters
print(f"   ✓ Clustering complete!")
print()

# 6. Evaluasi
print("📊 Evaluation Metrics:")
silhouette = silhouette_score(features_scaled, clusters)
davies_bouldin = davies_bouldin_score(features_scaled, clusters)
print(f"   • Silhouette Score: {silhouette:.4f}")
print(f"   • Davies-Bouldin Index: {davies_bouldin:.4f}")
print()

# 7. Distribusi cluster
print("📈 Cluster Distribution:")
for i in range(NUM_CLUSTERS):
    count = len(df_clean[df_clean['cluster'] == i])
    pct = (count / len(df_clean)) * 100
    avg_frp = df_clean[df_clean['cluster'] == i]['frp'].mean()
    print(f"   Cluster {i}: {count:4d} points ({pct:5.1f}%) | Avg FRP: {avg_frp:6.2f}")
print()

# 8. Visualisasi Scatter Plot
print("📊 Creating scatter plot...")
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

for i in range(NUM_CLUSTERS):
    cluster_data = df_clean[df_clean['cluster'] == i]
    plt.scatter(cluster_data['longitude'], cluster_data['latitude'],
                c=colors[i], label=f'Cluster {i}', alpha=0.6, s=30)

plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title(f'Fire Spots Clustering - {CITY_NAME}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

filename = f'clustering_{CITY_NAME.lower().replace(" ", "_")}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {filename}")
print()

# 9. Visualisasi Folium
print("🗺️  Creating interactive map...")

# Center point
center_lat = df_clean['latitude'].mean()
center_lon = df_clean['longitude'].mean()
zoom_level = INDONESIA_CITIES[CITY_NAME].get('zoom', 12)

# Buat peta
map_city = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=zoom_level,
    tiles='OpenStreetMap'
)

# Warna untuk cluster
folium_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Tambahkan markers per cluster
for i in range(NUM_CLUSTERS):
    cluster_data = df_clean[df_clean['cluster'] == i]

    feature_group = folium.FeatureGroup(name=f'Cluster {i} ({len(cluster_data)} points)')

    # Sampling untuk performa
    max_points = min(300, len(cluster_data))
    cluster_sample = cluster_data.sample(n=max_points, random_state=42)

    for idx, row in cluster_sample.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            popup=f"<b>Cluster {i}</b><br>"
                  f"Lat: {row['latitude']:.5f}<br>"
                  f"Lon: {row['longitude']:.5f}<br>"
                  f"Brightness: {row['brightness']:.1f}<br>"
                  f"FRP: {row['frp']:.2f}",
            color=folium_colors[i],
            fill=True,
            fillColor=folium_colors[i],
            fillOpacity=0.7
        ).add_to(feature_group)

    feature_group.add_to(map_city)

# Layer control
folium.LayerControl().add_to(map_city)

# Legend
legend_html = f'''
<div style="position: fixed;
            bottom: 50px; right: 50px; width: 220px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px; border-radius: 5px">
<p style="margin: 0 0 8px 0;"><b>🔥 {CITY_NAME}</b></p>
<p style="margin: 0 0 5px 0; font-size:11px; color:#666;">
   Total: {len(df_clean):,} fire spots
</p>
'''

for i in range(NUM_CLUSTERS):
    count = len(df_clean[df_clean['cluster'] == i])
    legend_html += f'<p style="margin: 2px 0;"><span style="color: {folium_colors[i]};">●</span> Cluster {i} ({count})</p>'

legend_html += '</div>'
map_city.get_root().html.add_child(folium.Element(legend_html))

# Simpan
map_filename = f'map_{CITY_NAME.lower().replace(" ", "_")}_clusters.html'
map_city.save(map_filename)
print(f"   ✓ Saved: {map_filename}")
print()

# 10. Heatmap
print("🔥 Creating heatmap...")
map_heat = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=zoom_level,
    tiles='OpenStreetMap'
)

heat_data = [[row['latitude'], row['longitude'], row['frp']]
             for idx, row in df_clean.iterrows()]

HeatMap(heat_data, radius=15, blur=20, max_zoom=13).add_to(map_heat)

heat_filename = f'heatmap_{CITY_NAME.lower().replace(" ", "_")}.html'
map_heat.save(heat_filename)
print(f"   ✓ Saved: {heat_filename}")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("="*80)
print("✅ SUMMARY")
print("="*80)
print(f"""
📍 Lokasi: {CITY_NAME}
📊 Data Points: {len(df_clean):,} fire spots
📅 Periode: {df_city['acq_date'].min()} - {df_city['acq_date'].max()}

🤖 Clustering:
   • Algoritma: K-Means
   • Jumlah Cluster: {NUM_CLUSTERS}
   • Silhouette Score: {silhouette:.4f}
   • Davies-Bouldin: {davies_bouldin:.4f}

📁 Output Files:
   1. {filename}
   2. {map_filename}
   3. {heat_filename}

💡 Interpretasi:
   - Cluster mengelompokkan titik api berdasarkan lokasi & intensitas
   - Buka file HTML untuk melihat peta interaktif
   - Silhouette score {silhouette:.2f} menunjukkan clustering {'baik' if silhouette > 0.5 else 'cukup baik' if silhouette > 0.3 else 'kurang optimal'}

🔄 Untuk analyze kota lain:
   → Ubah CITY_NAME di bagian KONFIGURASI (baris 18)
   → Kota tersedia: {', '.join(list(INDONESIA_CITIES.keys())[:5])}...
""")

print("="*80)
print("🚀 SELESAI!")
print("="*80)
