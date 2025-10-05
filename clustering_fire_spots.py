"""
CLUSTERING TITIK API (FIRE SPOTS) MENGGUNAKAN K-MEANS
======================================================

CARA PAKAI:
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
sns.set_style('whitegrid')

# =============================================================================
# ⚙️ KONFIGURASI - EDIT DI SINI
# =============================================================================

DATA_SOURCE = 'data/raw/fire_nrt_J2V-C2_669817.csv'  # Semua Indonesia
# DATA_SOURCE = 'data/filtered/method_1_bounding_box/Riau.csv'
# DATA_SOURCE = 'data/filtered/method_2_geojson/Riau.csv'

NUM_CLUSTERS = 5
SAMPLE_SIZE = 10000  # None untuk pakai semua data
FEATURES = ['latitude', 'longitude', 'brightness', 'frp']

# =============================================================================
# FUNCTIONS
# =============================================================================

def setup_output_dirs(dataset_name):
    """Create output directories"""
    output_dir = f'output/{dataset_name}'
    dirs = {
        'base': output_dir,
        'eda': os.path.join(output_dir, 'eda'),
        'plots': os.path.join(output_dir, 'plots'),
        'maps': os.path.join(output_dir, 'maps')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def load_and_sample_data(filepath, sample_size=None):
    """Load data and optionally sample"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    df = pd.read_csv(filepath)
    print(f"✓ Loaded: {len(df):,} fire spots")
    print(f"✓ Period: {df['acq_date'].min()} - {df['acq_date'].max()}")

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"✓ Sampled: {len(df):,} points")

    return df

def perform_eda(df, features, output_dir):
    """Exploratory Data Analysis"""
    print("\n📊 STEP 2: Exploratory Data Analysis")
    print("-" * 80)

    # Statistics
    stats = df[features].describe()
    stats.to_csv(os.path.join(output_dir, 'statistics.csv'))
    print("✓ Statistics saved")

    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Data Distribution', fontsize=14, fontweight='bold')

    for idx, feature in enumerate(features):
        ax = axes[idx // 2, idx % 2]
        df[feature].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(feature.capitalize())
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution.png'), dpi=300)
    plt.close()
    print("✓ Distribution plots saved")

def find_optimal_k(data, k_range=(2, 11)):
    """Elbow method to find optimal K"""
    print("\n📈 STEP 4: Elbow Method")
    print("-" * 80)

    inertias, silhouettes = [], []

    for k in range(*k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(data, kmeans.labels_)
        silhouettes.append(sil)
        print(f"K={k:2d} | Inertia: {kmeans.inertia_:10.2f} | Silhouette: {sil:.4f}")

    return inertias, silhouettes

def plot_elbow(inertias, silhouettes, k_range, output_file):
    """Plot elbow curve and silhouette scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ks = range(*k_range)
    ax1.plot(ks, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('K')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, silhouettes, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('K')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score by K')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print("✓ Elbow curve saved")

def perform_clustering(data, n_clusters):
    """Perform K-Means clustering"""
    print(f"\n🤖 STEP 5: K-Means Clustering (K={n_clusters})")
    print("-" * 80)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)

    print("✓ Clustering complete")
    return clusters, kmeans

def evaluate_clustering(data, clusters):
    """Evaluate clustering quality"""
    print("\n📊 STEP 6: Evaluation")
    print("-" * 80)

    silhouette = silhouette_score(data, clusters)
    davies_bouldin = davies_bouldin_score(data, clusters)

    print(f"Silhouette Score: {silhouette:.4f} (closer to 1 is better)")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (closer to 0 is better)")

    return silhouette, davies_bouldin

def save_cluster_stats(df, features, output_file):
    """Save cluster statistics"""
    cluster_stats = df.groupby('cluster')[features].mean()
    cluster_stats['count'] = df.groupby('cluster').size()
    cluster_stats.to_csv(output_file)

    print("\nCluster Distribution:")
    for i in range(df['cluster'].nunique()):
        count = len(df[df['cluster'] == i])
        pct = (count / len(df)) * 100
        avg_frp = df[df['cluster'] == i]['frp'].mean()
        print(f"  Cluster {i}: {count:6,} points ({pct:5.1f}%) | Avg FRP: {avg_frp:7.2f}")

    print("\n✓ Cluster stats saved")

def plot_clusters(df, output_dir):
    """Plot cluster visualizations"""
    print("\n🎨 STEP 7: Visualization")
    print("-" * 80)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    n_clusters = df['cluster'].nunique()

    # Location plot
    plt.figure(figsize=(12, 8))
    for i in range(n_clusters):
        cluster_data = df[df['cluster'] == i]
        plt.scatter(cluster_data['longitude'], cluster_data['latitude'],
                    c=colors[i], label=f'Cluster {i}', alpha=0.6, s=20)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Clustering by Location')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clusters_location.png'), dpi=300)
    plt.close()
    print("✓ Location plot saved")

    # Brightness vs FRP plot
    plt.figure(figsize=(12, 8))
    for i in range(n_clusters):
        cluster_data = df[df['cluster'] == i]
        plt.scatter(cluster_data['brightness'], cluster_data['frp'],
                    c=colors[i], label=f'Cluster {i}', alpha=0.6, s=20)

    plt.xlabel('Brightness')
    plt.ylabel('FRP')
    plt.title('Brightness vs FRP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clusters_brightness_frp.png'), dpi=300)
    plt.close()
    print("✓ Brightness vs FRP plot saved")

def create_maps(df, output_dir):
    """Create interactive Folium maps"""
    print("\n🗺️  STEP 8: Interactive Maps")
    print("-" * 80)

    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    n_clusters = df['cluster'].nunique()

    # Clusters map
    map_clusters = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    for i in range(n_clusters):
        cluster_data = df[df['cluster'] == i]
        fg = folium.FeatureGroup(name=f'Cluster {i} ({len(cluster_data)} points)')

        # Sample for performance
        sample_size = min(300, len(cluster_data))
        cluster_sample = cluster_data.sample(n=sample_size, random_state=42)

        for _, row in cluster_sample.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                popup=f"Cluster {i}<br>FRP: {row['frp']:.2f}",
                color=colors[i],
                fill=True,
                fillColor=colors[i],
                fillOpacity=0.7
            ).add_to(fg)

        fg.add_to(map_clusters)

    folium.LayerControl().add_to(map_clusters)
    map_clusters.save(os.path.join(output_dir, 'clusters.html'))
    print("✓ Cluster map saved")

    # Heatmap
    map_heat = folium.Map(location=[center_lat, center_lon], zoom_start=5)
    heat_data = [[row['latitude'], row['longitude'], row['frp']]
                 for _, row in df.iterrows()]
    HeatMap(heat_data, radius=15, blur=20).add_to(map_heat)
    map_heat.save(os.path.join(output_dir, 'heatmap.html'))
    print("✓ Heatmap saved")

def print_summary(dataset_name, df, n_clusters, silhouette, davies_bouldin, output_dir):
    """Print final summary"""
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    print(f"""
Dataset: {dataset_name}
Total: {len(df):,} fire spots

Clustering:
  • Algorithm: K-Means
  • K: {n_clusters}
  • Silhouette: {silhouette:.4f}
  • Davies-Bouldin: {davies_bouldin:.4f}

Output: {output_dir}/
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
      └── heatmap.html

Next:
  • Buka file HTML untuk lihat peta interaktif
  • Analisis karakteristik tiap cluster
  • Coba dataset lain dengan ubah DATA_SOURCE
""")
    print("="*80)
    print("✅ SELESAI!")
    print("="*80)

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main clustering pipeline"""
    # Setup
    dataset_name = os.path.basename(DATA_SOURCE).replace('.csv', '')
    dirs = setup_output_dirs(dataset_name)

    print("="*80)
    print(f"CLUSTERING FIRE SPOTS - {dataset_name.upper()}")
    print("="*80)

    # Load data
    print("\n📂 STEP 1: Load Data")
    print("-" * 80)
    df = load_and_sample_data(DATA_SOURCE, SAMPLE_SIZE)

    # EDA
    perform_eda(df, FEATURES, dirs['eda'])

    # Preprocessing
    print("\n🔧 STEP 3: Preprocessing")
    print("-" * 80)
    df_clean = df[FEATURES].dropna().copy()
    print(f"✓ Features: {', '.join(FEATURES)}")
    print(f"✓ Clean data: {len(df_clean):,} rows")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_clean)
    print("✓ Data normalized")

    # Elbow method
    inertias, silhouettes = find_optimal_k(features_scaled)
    plot_elbow(inertias, silhouettes, (2, 11),
               os.path.join(dirs['plots'], 'elbow.png'))

    # Clustering
    clusters, kmeans = perform_clustering(features_scaled, NUM_CLUSTERS)
    df_clean['cluster'] = clusters

    # Evaluation
    silhouette, davies_bouldin = evaluate_clustering(features_scaled, clusters)
    save_cluster_stats(df_clean, FEATURES,
                      os.path.join(dirs['eda'], 'cluster_stats.csv'))

    # Visualization
    plot_clusters(df_clean, dirs['plots'])
    create_maps(df_clean, dirs['maps'])

    # Summary
    print_summary(dataset_name, df_clean, NUM_CLUSTERS,
                 silhouette, davies_bouldin, dirs['base'])

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tips:")
        print("   - Untuk data raw: DATA_SOURCE = 'data/raw/fire_nrt_J2V-C2_669817.csv'")
        print("   - Untuk data filter: Jalankan dulu filter_by_city_guide.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
