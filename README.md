# Clustering Titik Api (Fire Spots) dengan K-Means

## Deskripsi
Program ini melakukan analisis clustering pada data titik api (fire hotspots) menggunakan algoritma K-Means. Data berasal dari VIIRS (Visible Infrared Imaging Radiometer Suite) Fire Detection.

## Fitur
1. **Eksplorasi Data**: Analisis statistik deskriptif dan distribusi data
2. **Preprocessing**: Cleaning dan normalisasi data
3. **Clustering**: Implementasi K-Means dengan penentuan K optimal menggunakan Elbow Method
4. **Evaluasi**: Silhouette Score dan Davies-Bouldin Index
5. **Visualisasi**:
   - Scatter plot hasil clustering
   - Peta interaktif dengan Folium
   - Heatmap intensitas titik api

## Cara Penggunaan

### Di Local Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan script
python clustering_fire_spots.py
```

### Di Google Colab
```python
# Upload file CSV dan script Python ke Colab

# Install dependencies
!pip install pandas numpy matplotlib seaborn scikit-learn folium

# Jalankan script
!python clustering_fire_spots.py

# Lihat hasil visualisasi
from IPython.display import IFrame
IFrame('fire_spots_clustering_map.html', width=800, height=600)
```

## Output
- `elbow_curve.png`: Grafik Elbow Method untuk menentukan K optimal
- `clustering_scatter.png`: Scatter plot hasil clustering
- `fire_spots_clustering_map.html`: Peta interaktif dengan marker per cluster
- `fire_spots_heatmap.html`: Heatmap intensitas titik api

## Konfigurasi
Untuk mengubah jumlah data yang diproses, edit variabel `SAMPLE_SIZE` di script:
```python
SAMPLE_SIZE = 10000  # Ubah sesuai kebutuhan, set None untuk semua data
```

Untuk mengubah jumlah cluster, edit variabel `K_optimal`:
```python
K_optimal = 5  # Sesuaikan berdasarkan Elbow Method
```

## Interpretasi Hasil
- **Cluster 0**: Intensitas tinggi (FRP ≈ 30.6)
- **Cluster 1**: Intensitas sedang, populasi terbanyak (47.5%)
- **Cluster 2**: Intensitas sangat tinggi (FRP ≈ 138.7), populasi sedikit
- **Cluster 3**: Intensitas rendah (FRP ≈ 2.6)
- **Cluster 4**: Area spesifik di Sulawesi/NTT

## Metrik Evaluasi
- **Silhouette Score**: 0.3807 (cukup baik)
- **Davies-Bouldin Index**: 1.0153 (semakin kecil semakin baik)

## Catatan untuk Pengajaran
Script ini dibuat sesimpel mungkin dengan penjelasan per langkah untuk memudahkan pembelajaran. Mahasiswa dapat:
1. Memahami alur clustering dari awal hingga akhir
2. Melihat visualisasi interaktif untuk interpretasi hasil
3. Mengeksplorasi parameter (K, sampling, fitur) untuk pemahaman lebih dalam
