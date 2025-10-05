"""
CLUSTERING TITIK API (FIRE SPOTS) MENGGUNAKAN K-MEANS
======================================================

Studi Kasus: Analisis Clustering Titik Api di Indonesia
Data: VIIRS Fire Detection Data

Langkah-langkah:
1. Import Library dan Load Data
2. Eksplorasi Data Awal
3. Preprocessing Data
4. Implementasi Clustering (K-Means)
5. Evaluasi Hasil Clustering
6. Visualisasi dengan Folium

Author: Untuk keperluan pengajaran Data Mining
"""

# =============================================================================
# LANGKAH 1: IMPORT LIBRARY
# =============================================================================
print("="*70)
print("STEP 1: Import Library")
print("="*70)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import folium
from folium.plugins import MarkerCluster, HeatMap
import warnings
warnings.filterwarnings('ignore')

print("✓ Library berhasil di-import")
print()

# =============================================================================
# LANGKAH 2: LOAD DAN EKSPLORASI DATA
# =============================================================================
print("="*70)
print("STEP 2: Load dan Eksplorasi Data")
print("="*70)

# Load data
df = pd.read_csv('fire_nrt_J2V-C2_669817.csv')

print(f"Jumlah data total: {len(df)} baris")

# SAMPLING DATA (untuk mempercepat proses demo)
# Untuk demo/pembelajaran, kita ambil sample 10,000 data
# Untuk analisis penuh, set SAMPLE_SIZE = None
SAMPLE_SIZE = 10000

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Data di-sampling menjadi: {len(df)} baris (untuk demo)")
else:
    print(f"Menggunakan semua data: {len(df)} baris")

print(f"Jumlah kolom: {len(df.columns)} kolom")
print()

print("5 Data Pertama:")
print(df.head())
print()

print("Informasi Dataset:")
print(df.info())
print()

print("Statistik Deskriptif:")
print(df.describe())
print()

print("Cek Missing Values:")
print(df.isnull().sum())
print()

# =============================================================================
# LANGKAH 3: PREPROCESSING DATA
# =============================================================================
print("="*70)
print("STEP 3: Preprocessing Data")
print("="*70)

# Pilih fitur untuk clustering
# Kita gunakan: latitude, longitude, brightness, frp (Fire Radiative Power)
print("Fitur yang dipilih untuk clustering:")
print("- latitude: koordinat lintang")
print("- longitude: koordinat bujur")
print("- brightness: tingkat kecerahan")
print("- frp: Fire Radiative Power (daya radiasi api)")
print()

# Filter data yang valid (tidak ada missing values pada kolom penting)
df_clean = df[['latitude', 'longitude', 'brightness', 'frp']].dropna()
print(f"Jumlah data setelah cleaning: {len(df_clean)} baris")
print()

# Normalisasi data menggunakan StandardScaler
print("Melakukan normalisasi data...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_clean)

print("✓ Data berhasil dinormalisasi")
print()

# =============================================================================
# LANGKAH 4: MENENTUKAN JUMLAH CLUSTER OPTIMAL (ELBOW METHOD)
# =============================================================================
print("="*70)
print("STEP 4: Menentukan Jumlah Cluster Optimal")
print("="*70)

print("Menggunakan Elbow Method...")
print("Menguji K = 2 sampai K = 10...")

# Hitung inertia untuk berbagai nilai K
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    inertias.append(kmeans.inertia_)
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}")

print()

# Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Jumlah Cluster (K)', fontsize=12)
plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
plt.title('Elbow Method untuk Menentukan K Optimal', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.savefig('elbow_curve.png', dpi=300, bbox_inches='tight')
print("✓ Elbow curve disimpan sebagai 'elbow_curve.png'")
print()

# =============================================================================
# LANGKAH 5: IMPLEMENTASI K-MEANS CLUSTERING
# =============================================================================
print("="*70)
print("STEP 5: Implementasi K-Means Clustering")
print("="*70)

# Berdasarkan elbow method, kita pilih K=5 (bisa disesuaikan)
K_optimal = 5
print(f"Jumlah cluster yang dipilih: K = {K_optimal}")
print()

# Fit K-Means
kmeans = KMeans(n_clusters=K_optimal, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)

# Tambahkan hasil clustering ke dataframe
df_clean['cluster'] = clusters

print("✓ Clustering berhasil!")
print()

# =============================================================================
# LANGKAH 6: EVALUASI HASIL CLUSTERING
# =============================================================================
print("="*70)
print("STEP 6: Evaluasi Hasil Clustering")
print("="*70)

# Hitung metrik evaluasi
silhouette = silhouette_score(features_scaled, clusters)
davies_bouldin = davies_bouldin_score(features_scaled, clusters)

print(f"Silhouette Score: {silhouette:.4f}")
print("  → Semakin mendekati 1, semakin baik (range: -1 sampai 1)")
print()

print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print("  → Semakin kecil, semakin baik (minimum: 0)")
print()

# Distribusi data per cluster
print("Distribusi Data per Cluster:")
cluster_counts = df_clean['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(df_clean)) * 100
    print(f"  Cluster {cluster_id}: {count} data ({percentage:.1f}%)")
print()

# Karakteristik setiap cluster
print("Karakteristik Setiap Cluster:")
cluster_stats = df_clean.groupby('cluster')[['latitude', 'longitude', 'brightness', 'frp']].mean()
print(cluster_stats)
print()

# =============================================================================
# LANGKAH 7: VISUALISASI HASIL
# =============================================================================
print("="*70)
print("STEP 7: Visualisasi Hasil Clustering")
print("="*70)

# 7.1 Scatter Plot (Latitude vs Longitude)
print("Membuat scatter plot...")
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

for i in range(K_optimal):
    cluster_data = df_clean[df_clean['cluster'] == i]
    plt.scatter(cluster_data['longitude'], cluster_data['latitude'],
                c=colors[i], label=f'Cluster {i}', alpha=0.6, s=20)

plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('Clustering Titik Api Berdasarkan Lokasi', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('clustering_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Scatter plot disimpan sebagai 'clustering_scatter.png'")
print()

# 7.2 Visualisasi dengan Folium
print("Membuat peta interaktif dengan Folium...")

# Hitung center point (rata-rata latitude dan longitude)
center_lat = df_clean['latitude'].mean()
center_lon = df_clean['longitude'].mean()

# Buat peta dasar
map_clusters = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles='OpenStreetMap'
)

# Warna untuk setiap cluster
folium_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Tambahkan marker untuk setiap cluster
for i in range(K_optimal):
    cluster_data = df_clean[df_clean['cluster'] == i]

    # Buat feature group untuk setiap cluster (agar bisa di-toggle)
    feature_group = folium.FeatureGroup(name=f'Cluster {i} ({len(cluster_data)} points)')

    # Sampling data jika terlalu banyak (untuk performa)
    sample_size = min(200, len(cluster_data))  # Maksimal 200 points per cluster
    cluster_sample = cluster_data.sample(n=sample_size, random_state=42)

    # Tambahkan circle marker untuk setiap titik
    for idx, row in cluster_sample.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            popup=f"Cluster: {i}<br>"
                  f"Lat: {row['latitude']:.4f}<br>"
                  f"Lon: {row['longitude']:.4f}<br>"
                  f"Brightness: {row['brightness']:.2f}<br>"
                  f"FRP: {row['frp']:.2f}",
            color=folium_colors[i],
            fill=True,
            fillColor=folium_colors[i],
            fillOpacity=0.6
        ).add_to(feature_group)

    feature_group.add_to(map_clusters)

# Tambahkan layer control
folium.LayerControl().add_to(map_clusters)

# Tambahkan legend
legend_html = '''
<div style="position: fixed;
            bottom: 50px; right: 50px; width: 200px; height: auto;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px">
<p style="margin-bottom: 5px;"><b>Cluster Legend</b></p>
'''

for i in range(K_optimal):
    count = len(df_clean[df_clean['cluster'] == i])
    legend_html += f'<p style="margin: 3px;"><span style="color: {folium_colors[i]};">●</span> Cluster {i} ({count} points)</p>'

legend_html += '</div>'
map_clusters.get_root().html.add_child(folium.Element(legend_html))

# Simpan peta
map_clusters.save('fire_spots_clustering_map.html')
print("✓ Peta interaktif disimpan sebagai 'fire_spots_clustering_map.html'")
print()

# 7.3 Peta Heatmap
print("Membuat heatmap...")
map_heatmap = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles='OpenStreetMap'
)

# Data untuk heatmap (latitude, longitude, weight=frp)
heat_data = [[row['latitude'], row['longitude'], row['frp']]
             for idx, row in df_clean.iterrows()]

# Tambahkan heatmap layer
HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(map_heatmap)

# Simpan heatmap
map_heatmap.save('fire_spots_heatmap.html')
print("✓ Heatmap disimpan sebagai 'fire_spots_heatmap.html'")
print()

# =============================================================================
# LANGKAH 8: RINGKASAN HASIL
# =============================================================================
print("="*70)
print("STEP 8: RINGKASAN HASIL")
print("="*70)

print(f"""
HASIL ANALISIS CLUSTERING TITIK API
====================================

1. Dataset:
   - Total data: {len(df)} titik api
   - Data valid untuk clustering: {len(df_clean)} titik api
   - Periode: {df['acq_date'].min()} s/d {df['acq_date'].max()}

2. Metode Clustering:
   - Algoritma: K-Means
   - Jumlah cluster: {K_optimal}
   - Fitur: latitude, longitude, brightness, FRP

3. Evaluasi:
   - Silhouette Score: {silhouette:.4f}
   - Davies-Bouldin Index: {davies_bouldin:.4f}

4. Output yang dihasilkan:
   - elbow_curve.png: Grafik untuk menentukan K optimal
   - clustering_scatter.png: Scatter plot hasil clustering
   - fire_spots_clustering_map.html: Peta interaktif dengan marker per cluster
   - fire_spots_heatmap.html: Heatmap intensitas titik api

5. Interpretasi:
   Titik api telah dikelompokkan menjadi {K_optimal} cluster berdasarkan
   lokasi geografis dan intensitas api. Setiap cluster merepresentasikan
   area dengan karakteristik kebakaran yang serupa.

SELESAI! Silakan buka file HTML untuk melihat visualisasi interaktif.
""")

print("="*70)
print("PROGRAM SELESAI")
print("="*70)
