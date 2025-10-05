"""
CLUSTERING TITIK API (FIRE SPOTS) MENGGUNAKAN K-MEANS
======================================================

Studi Kasus: Analisis Clustering Titik Api di Indonesia

CARA PAKAI:
-----------
1. Edit DATA_SOURCE di bawah (pilih data yang mau dianalisis)
2. Jalankan: python clustering_fire_spots.py
3. Lihat hasil di folder output/

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

# Pilih sumber data (pilih salah satu):
DATA_SOURCE = 'data/raw/fire_nrt_J2V-C2_669817.csv'                    # Semua Indonesia
DATA_SOURCE = 'data/raw/fire_nrt_J2V-C2_669817.csv'      # Jakarta only
# DATA_SOURCE = 'data/filtered/method_1_bounding_box/Surabaya.csv'    # Surabaya only
# DATA_SOURCE = 'data/filtered/method_1_bounding_box/Bandung.csv'     # Bandung only

# Jumlah cluster
NUM_CLUSTERS = 5

# Sampling (None = pakai semua data, atau set angka misal 5000)
SAMPLE_SIZE = 10000  # Ubah ke None untuk pakai semua data

# =============================================================================
# AUTO-DETECT OUTPUT FOLDER
# =============================================================================

# Otomatis detect nama dari file
dataset_name = os.path.basename(DATA_SOURCE).replace('.csv', '')
OUTPUT_DIR = f'output/{dataset_name}'

# Buat subfolder
OUTPUT_EDA = os.path.join(OUTPUT_DIR, 'eda')
OUTPUT_PLOTS = os.path.join(OUTPUT_DIR, 'plots')
OUTPUT_MAPS = os.path.join(OUTPUT_DIR, 'maps')

for folder in [OUTPUT_DIR, OUTPUT_EDA, OUTPUT_PLOTS, OUTPUT_MAPS]:
    os.makedirs(folder, exist_ok=True)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("="*80)
print(f"CLUSTERING FIRE SPOTS - {dataset_name.upper()}")
print("="*80)
print()

print("📂 STEP 1: Load Data")
print("-" * 80)

if not os.path.exists(DATA_SOURCE):
    print(f"❌ File tidak ditemukan: {DATA_SOURCE}")
    print()
    print("💡 Tips:")
    print("   - Untuk data raw: DATA_SOURCE = 'data/raw/fire_nrt_J2V-C2_669817.csv'")
    print("   - Untuk data kota: Jalankan dulu filter_by_city_guide.py")
    exit()

df = pd.read_csv(DATA_SOURCE)
print(f"✓ Data loaded: {len(df):,} fire spots")
print(f"✓ Periode: {df['acq_date'].min()} - {df['acq_date'].max()}")

# Sampling
if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"✓ Sampled to: {len(df):,} samples")

print()

# =============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("📊 STEP 2: Exploratory Data Analysis")
print("-" * 80)

# Basic stats
stats = df[['latitude', 'longitude', 'brightness', 'frp']].describe()
stats.to_csv(os.path.join(OUTPUT_EDA, 'statistics.csv'))
print("✓ Statistics saved")

# Distribution plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f'Data Distribution - {dataset_name}', fontsize=14, fontweight='bold')

df['latitude'].hist(ax=axes[0,0], bins=30, edgecolor='black')
axes[0,0].set_title('Latitude')

df['longitude'].hist(ax=axes[0,1], bins=30, edgecolor='black')
axes[0,1].set_title('Longitude')

df['brightness'].hist(ax=axes[1,0], bins=30, edgecolor='black')
axes[1,0].set_title('Brightness')

df['frp'].hist(ax=axes[1,1], bins=30, edgecolor='black')
axes[1,1].set_title('FRP')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_EDA, 'distribution.png'), dpi=300)
plt.close()
print("✓ Distribution plots saved")
print()

# =============================================================================
# STEP 3: PREPROCESSING
# =============================================================================

print("🔧 STEP 3: Preprocessing")
print("-" * 80)

# Select features
features = ['latitude', 'longitude', 'brightness', 'frp']
df_clean = df[features].dropna().copy()
print(f"✓ Features: {', '.join(features)}")
print(f"✓ Clean data: {len(df_clean):,} rows")

# Normalize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_clean)
print(f"✓ Data normalized")
print()

# =============================================================================
# STEP 4: ELBOW METHOD
# =============================================================================

print("📈 STEP 4: Elbow Method")
print("-" * 80)

inertias = []
silhouettes = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    inertias.append(kmeans.inertia_)
    sil = silhouette_score(features_scaled, kmeans.labels_)
    silhouettes.append(sil)
    print(f"K={k:2d} | Inertia: {kmeans.inertia_:10.2f} | Silhouette: {sil:.4f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(2, 11), inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('K')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.grid(True, alpha=0.3)

ax2.plot(range(2, 11), silhouettes, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('K')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score by K')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOTS, 'elbow.png'), dpi=300)
plt.close()
print("\n✓ Elbow curve saved")
print()

# =============================================================================
# STEP 5: K-MEANS CLUSTERING
# =============================================================================

print(f"🤖 STEP 5: K-Means Clustering (K={NUM_CLUSTERS})")
print("-" * 80)

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)
df_clean['cluster'] = clusters

print(f"✓ Clustering complete")
print()

# =============================================================================
# STEP 6: EVALUATION
# =============================================================================

print("📊 STEP 6: Evaluation")
print("-" * 80)

silhouette = silhouette_score(features_scaled, clusters)
davies_bouldin = davies_bouldin_score(features_scaled, clusters)

print(f"Silhouette Score: {silhouette:.4f} (closer to 1 is better)")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (closer to 0 is better)")
print()

print("Cluster Distribution:")
for i in range(NUM_CLUSTERS):
    count = len(df_clean[df_clean['cluster'] == i])
    pct = (count / len(df_clean)) * 100
    avg_frp = df_clean[df_clean['cluster'] == i]['frp'].mean()
    print(f"  Cluster {i}: {count:6,} points ({pct:5.1f}%) | Avg FRP: {avg_frp:7.2f}")

# Save cluster stats
cluster_stats = df_clean.groupby('cluster')[features].mean()
cluster_stats['count'] = df_clean.groupby('cluster').size()
cluster_stats.to_csv(os.path.join(OUTPUT_EDA, 'cluster_stats.csv'))
print("\n✓ Cluster stats saved")
print()

# =============================================================================
# STEP 7: VISUALIZATION
# =============================================================================

print("🎨 STEP 7: Visualization")
print("-" * 80)

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Plot 1: Location
plt.figure(figsize=(12, 8))
for i in range(NUM_CLUSTERS):
    cluster_data = df_clean[df_clean['cluster'] == i]
    plt.scatter(cluster_data['longitude'], cluster_data['latitude'],
                c=colors[i], label=f'Cluster {i}', alpha=0.6, s=20)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Clustering - {dataset_name}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOTS, 'clusters_location.png'), dpi=300)
plt.close()
print("✓ Location plot saved")

# Plot 2: Brightness vs FRP
plt.figure(figsize=(12, 8))
for i in range(NUM_CLUSTERS):
    cluster_data = df_clean[df_clean['cluster'] == i]
    plt.scatter(cluster_data['brightness'], cluster_data['frp'],
                c=colors[i], label=f'Cluster {i}', alpha=0.6, s=20)

plt.xlabel('Brightness')
plt.ylabel('FRP')
plt.title(f'Brightness vs FRP - {dataset_name}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOTS, 'clusters_brightness_frp.png'), dpi=300)
plt.close()
print("✓ Brightness vs FRP plot saved")
print()

# =============================================================================
# STEP 8: INTERACTIVE MAPS
# =============================================================================

print("🗺️  STEP 8: Interactive Maps")
print("-" * 80)

center_lat = df_clean['latitude'].mean()
center_lon = df_clean['longitude'].mean()

# Map 1: Clusters
map_clusters = folium.Map(location=[center_lat, center_lon], zoom_start=5)

for i in range(NUM_CLUSTERS):
    cluster_data = df_clean[df_clean['cluster'] == i]
    fg = folium.FeatureGroup(name=f'Cluster {i} ({len(cluster_data)} points)')

    # Sample untuk performa
    sample_size = min(300, len(cluster_data))
    cluster_sample = cluster_data.sample(n=sample_size, random_state=42)

    for idx, row in cluster_sample.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            popup=f"Cluster {i}<br>Lat: {row['latitude']:.4f}<br>Lon: {row['longitude']:.4f}<br>FRP: {row['frp']:.2f}",
            color=colors[i],
            fill=True,
            fillColor=colors[i],
            fillOpacity=0.7
        ).add_to(fg)

    fg.add_to(map_clusters)

folium.LayerControl().add_to(map_clusters)
map_clusters.save(os.path.join(OUTPUT_MAPS, 'clusters.html'))
print("✓ Cluster map saved")

# Map 2: Heatmap
map_heat = folium.Map(location=[center_lat, center_lon], zoom_start=5)
heat_data = [[row['latitude'], row['longitude'], row['frp']] for idx, row in df_clean.iterrows()]
HeatMap(heat_data, radius=15, blur=20).add_to(map_heat)
map_heat.save(os.path.join(OUTPUT_MAPS, 'heatmap.html'))
print("✓ Heatmap saved")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("="*80)
print("📋 SUMMARY")
print("="*80)
print(f"""
Dataset: {dataset_name}
Total: {len(df_clean):,} fire spots
Period: {df['acq_date'].min()} - {df['acq_date'].max()}

Clustering:
  • Algorithm: K-Means
  • K: {NUM_CLUSTERS}
  • Silhouette: {silhouette:.4f}
  • Davies-Bouldin: {davies_bouldin:.4f}

Output saved to: {OUTPUT_DIR}/
  ├── eda/
  │   ├── statistics.csv
  │   ├── cluster_stats.csv
  │   └── distribution.png
  ├── plots/
  │   ├── elbow.png
  │   ├── clusters_location.png
  │   └── clusters_brightness_frp.png
  └── maps/
      ├── clusters.html      ← Buka di browser!
      └── heatmap.html       ← Buka di browser!

Next:
  • Buka file HTML untuk lihat peta interaktif
  • Analisis karakteristik tiap cluster
  • Coba dataset lain dengan ubah DATA_SOURCE
""")

print("="*80)
print("✅ SELESAI!")
print("="*80)
