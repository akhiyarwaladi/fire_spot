"""
CLUSTERING TITIK API (FIRE SPOTS) MENGGUNAKAN K-MEANS
======================================================

Studi Kasus: Analisis Clustering Titik Api di Indonesia
Data: VIIRS Fire Detection Data

CARA PAKAI:
-----------
1. Jalankan filter_by_city_guide.py dulu untuk generate filtered data per kota
2. Edit KONFIGURASI di bawah (pilih kota dan metode)
3. Jalankan script ini: python clustering_fire_spots.py
4. Lihat hasil di folder output/

Author: Untuk keperluan pengajaran Data Mining
"""

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
import os
warnings.filterwarnings('ignore')

# =============================================================================
# ⚙️ KONFIGURASI - EDIT DI SINI
# =============================================================================

# Pilih sumber data
DATA_SOURCE = 'data/filtered/method_1_bounding_box/Jakarta.csv'  # Ubah sesuai kota yang diinginkan

# Nama kota (untuk penamaan output)
CITY_NAME = 'Jakarta'  # Sesuaikan dengan file data

# Metode filter yang digunakan
METHOD = 'method_1_bounding_box'  # method_1_bounding_box atau method_4_dictionary

# Clustering settings
NUM_CLUSTERS = 3  # Jumlah cluster yang diinginkan
RANDOM_STATE = 42

# Sampling (None untuk semua data)
SAMPLE_SIZE = None  # Set angka (misal 1000) jika ingin sampling

# Visualization settings
PLOT_DPI = 300
MAP_ZOOM = 12
MAX_MARKERS = 300  # Max markers per cluster di peta (untuk performa)

# =============================================================================
# 📁 SETUP OUTPUT FOLDERS
# =============================================================================

# Buat folder output terstruktur
OUTPUT_BASE = f'output/{METHOD}/{CITY_NAME.replace(" ", "_").lower()}'
OUTPUT_EDA = os.path.join(OUTPUT_BASE, 'eda')
OUTPUT_PLOTS = os.path.join(OUTPUT_BASE, 'plots')
OUTPUT_MAPS = os.path.join(OUTPUT_BASE, 'maps')

for folder in [OUTPUT_BASE, OUTPUT_EDA, OUTPUT_PLOTS, OUTPUT_MAPS]:
    os.makedirs(folder, exist_ok=True)

# =============================================================================
# LANGKAH 1: LOAD DATA
# =============================================================================

print("="*80)
print(f"CLUSTERING FIRE SPOTS - {CITY_NAME.upper()}")
print("="*80)
print()

print("📂 STEP 1: Load Data")
print("-" * 80)

# Check if data exists
if not os.path.exists(DATA_SOURCE):
    print(f"❌ File tidak ditemukan: {DATA_SOURCE}")
    print()
    print("💡 Jalankan dulu: python filter_by_city_guide.py")
    print("   untuk generate filtered data per kota")
    exit()

df = pd.read_csv(DATA_SOURCE)
print(f"✓ Data berhasil dimuat: {len(df):,} fire spots")
print(f"✓ Periode: {df['acq_date'].min()} - {df['acq_date'].max()}")

# Sampling jika diperlukan
if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(f"✓ Data di-sampling: {len(df):,} samples")

print()

# =============================================================================
# LANGKAH 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("📊 STEP 2: Exploratory Data Analysis")
print("-" * 80)

# Save info to text file
with open(os.path.join(OUTPUT_EDA, 'dataset_info.txt'), 'w') as f:
    f.write(f"DATASET INFORMATION - {CITY_NAME}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total records: {len(df):,}\n")
    f.write(f"Date range: {df['acq_date'].min()} - {df['acq_date'].max()}\n\n")
    f.write("Columns:\n")
    for col in df.columns:
        f.write(f"  - {col}\n")

print("✓ Dataset info")

# Descriptive statistics
desc_stats = df[['latitude', 'longitude', 'brightness', 'frp']].describe()
desc_stats.to_csv(os.path.join(OUTPUT_EDA, 'descriptive_statistics.csv'))
print("✓ Statistik deskriptif")

# Missing values
missing = df.isnull().sum()
missing.to_csv(os.path.join(OUTPUT_EDA, 'missing_values.csv'))
print("✓ Missing values check")

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Distribusi Data Fire Spots - {CITY_NAME}', fontsize=14, fontweight='bold')

df['latitude'].hist(ax=axes[0,0], bins=30, edgecolor='black')
axes[0,0].set_title('Distribusi Latitude')
axes[0,0].set_xlabel('Latitude')

df['longitude'].hist(ax=axes[0,1], bins=30, edgecolor='black')
axes[0,1].set_title('Distribusi Longitude')
axes[0,1].set_xlabel('Longitude')

df['brightness'].hist(ax=axes[1,0], bins=30, edgecolor='black')
axes[1,0].set_title('Distribusi Brightness')
axes[1,0].set_xlabel('Brightness')

df['frp'].hist(ax=axes[1,1], bins=30, edgecolor='black')
axes[1,1].set_title('Distribusi FRP')
axes[1,1].set_xlabel('FRP (Fire Radiative Power)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_EDA, 'data_distribution.png'), dpi=PLOT_DPI)
plt.close()
print("✓ Distribution plots")

print()

# =============================================================================
# LANGKAH 3: PREPROCESSING
# =============================================================================

print("🔧 STEP 3: Preprocessing Data")
print("-" * 80)

# Select features
features = ['latitude', 'longitude', 'brightness', 'frp']
df_clean = df[features].dropna().copy()
print(f"✓ Features selected: {', '.join(features)}")
print(f"✓ Data clean: {len(df_clean):,} rows (removed {len(df) - len(df_clean)} rows with missing values)")

# Standardization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_clean)
print(f"✓ Data normalized (StandardScaler)")

print()

# =============================================================================
# LANGKAH 4: ELBOW METHOD (Determine Optimal K)
# =============================================================================

print("📈 STEP 4: Elbow Method - Determine Optimal K")
print("-" * 80)

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(features_scaled)
    inertias.append(kmeans.inertia_)

    # Calculate silhouette score
    labels = kmeans.labels_
    sil_score = silhouette_score(features_scaled, labels)
    silhouette_scores.append(sil_score)

    print(f"  K={k:2d} | Inertia: {kmeans.inertia_:10.2f} | Silhouette: {sil_score:.4f}")

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
ax1.set_title('Elbow Method', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(K_range)

ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score by K', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(K_range)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOTS, 'elbow_method.png'), dpi=PLOT_DPI, bbox_inches='tight')
plt.close()

print(f"\n✓ Elbow curve saved")
print()

# =============================================================================
# LANGKAH 5: K-MEANS CLUSTERING
# =============================================================================

print(f"🤖 STEP 5: K-Means Clustering (K={NUM_CLUSTERS})")
print("-" * 80)

kmeans_final = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
clusters = kmeans_final.fit_predict(features_scaled)
df_clean['cluster'] = clusters

print(f"✓ Clustering complete with K={NUM_CLUSTERS}")
print()

# =============================================================================
# LANGKAH 6: EVALUATION
# =============================================================================

print("📊 STEP 6: Evaluation Metrics")
print("-" * 80)

silhouette = silhouette_score(features_scaled, clusters)
davies_bouldin = davies_bouldin_score(features_scaled, clusters)

print(f"Silhouette Score    : {silhouette:.4f} (closer to 1 is better)")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (closer to 0 is better)")
print()

# Cluster distribution
print("Cluster Distribution:")
for i in range(NUM_CLUSTERS):
    count = len(df_clean[df_clean['cluster'] == i])
    pct = (count / len(df_clean)) * 100
    avg_frp = df_clean[df_clean['cluster'] == i]['frp'].mean()
    print(f"  Cluster {i}: {count:5d} points ({pct:5.1f}%) | Avg FRP: {avg_frp:7.2f}")

print()

# Save cluster statistics
cluster_stats = df_clean.groupby('cluster')[features].mean()
cluster_stats['count'] = df_clean.groupby('cluster').size()
cluster_stats.to_csv(os.path.join(OUTPUT_EDA, 'cluster_statistics.csv'))
print("✓ Cluster statistics saved")
print()

# =============================================================================
# LANGKAH 7: VISUALIZATION - SCATTER PLOTS
# =============================================================================

print("🎨 STEP 7: Generating Visualizations")
print("-" * 80)

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Scatter plot: Location
plt.figure(figsize=(12, 8))
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
plt.savefig(os.path.join(OUTPUT_PLOTS, 'clustering_location.png'), dpi=PLOT_DPI)
plt.close()
print("✓ Location scatter plot")

# Scatter plot: Brightness vs FRP
plt.figure(figsize=(12, 8))
for i in range(NUM_CLUSTERS):
    cluster_data = df_clean[df_clean['cluster'] == i]
    plt.scatter(cluster_data['brightness'], cluster_data['frp'],
                c=colors[i], label=f'Cluster {i}', alpha=0.6, s=30)

plt.xlabel('Brightness', fontsize=12)
plt.ylabel('FRP (Fire Radiative Power)', fontsize=12)
plt.title(f'Clustering by Brightness & FRP - {CITY_NAME}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOTS, 'clustering_brightness_frp.png'), dpi=PLOT_DPI)
plt.close()
print("✓ Brightness vs FRP scatter plot")

print()

# =============================================================================
# LANGKAH 8: INTERACTIVE MAPS
# =============================================================================

print("🗺️  STEP 8: Creating Interactive Maps")
print("-" * 80)

# Center point
center_lat = df_clean['latitude'].mean()
center_lon = df_clean['longitude'].mean()

# Map 1: Clustered markers
map_clusters = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=MAP_ZOOM,
    tiles='OpenStreetMap'
)

folium_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

for i in range(NUM_CLUSTERS):
    cluster_data = df_clean[df_clean['cluster'] == i]
    feature_group = folium.FeatureGroup(name=f'Cluster {i} ({len(cluster_data)} points)')

    # Sample for performance
    sample_size = min(MAX_MARKERS, len(cluster_data))
    cluster_sample = cluster_data.sample(n=sample_size, random_state=RANDOM_STATE)

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

    feature_group.add_to(map_clusters)

folium.LayerControl().add_to(map_clusters)

# Add legend
legend_html = f'''
<div style="position: fixed; bottom: 50px; right: 50px; width: 220px;
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
map_clusters.get_root().html.add_child(folium.Element(legend_html))

map_clusters.save(os.path.join(OUTPUT_MAPS, 'interactive_clusters.html'))
print("✓ Cluster map (interactive)")

# Map 2: Heatmap
map_heat = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=MAP_ZOOM,
    tiles='OpenStreetMap'
)

heat_data = [[row['latitude'], row['longitude'], row['frp']]
             for idx, row in df_clean.iterrows()]

HeatMap(heat_data, radius=15, blur=20, max_zoom=13).add_to(map_heat)
map_heat.save(os.path.join(OUTPUT_MAPS, 'heatmap.html'))
print("✓ Heatmap (intensity)")

print()

# =============================================================================
# LANGKAH 9: SUMMARY REPORT
# =============================================================================

print("="*80)
print("📋 SUMMARY REPORT")
print("="*80)
print(f"""
📍 Location: {CITY_NAME}
📊 Dataset: {len(df_clean):,} fire spots
📅 Period: {df['acq_date'].min()} - {df['acq_date'].max()}

🤖 Clustering Results:
   • Algorithm: K-Means
   • Number of Clusters: {NUM_CLUSTERS}
   • Silhouette Score: {silhouette:.4f}
   • Davies-Bouldin Index: {davies_bouldin:.4f}

📁 Output Files Saved to: {OUTPUT_BASE}/

   eda/
   ├── dataset_info.txt
   ├── descriptive_statistics.csv
   ├── missing_values.csv
   ├── cluster_statistics.csv
   └── data_distribution.png

   plots/
   ├── elbow_method.png
   ├── clustering_location.png
   └── clustering_brightness_frp.png

   maps/
   ├── interactive_clusters.html  ← Buka ini di browser!
   └── heatmap.html               ← Buka ini di browser!

💡 Next Steps:
   • Buka file HTML untuk melihat peta interaktif
   • Analisis karakteristik setiap cluster
   • Coba kota lain dengan mengubah DATA_SOURCE dan CITY_NAME
""")

print("="*80)
print("✅ CLUSTERING SELESAI!")
print("="*80)
