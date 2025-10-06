# 🔥 Clustering Titik Api (Fire Spots) dengan K-Means

Analisis clustering data titik api (fire hotspots) di Indonesia menggunakan algoritma K-Means. Data berasal dari VIIRS (Visible Infrared Imaging Radiometer Suite) Fire Detection.

## 📋 Daftar Isi
- [Fitur](#fitur)
- [Setup](#setup)
- [Workflow](#workflow)
- [Metode Filtering](#metode-filtering)
- [Cara Penggunaan](#cara-penggunaan)
- [Struktur Folder](#struktur-folder)
- [Interpretasi Hasil](#interpretasi-hasil)
- [Untuk Mahasiswa](#untuk-mahasiswa)

## ✨ Fitur

### 1. **Filtering Data Per Wilayah**
- **Method 1 - Bounding Box**: Filtering cepat menggunakan kotak lat/lon
- **Method 2 - GeoJSON**: Filtering akurat menggunakan batas provinsi sebenarnya
- Support 35+ provinsi di Indonesia
- Fokus pada hotspot area: Sumatera & Kalimantan

### 2. **Clustering dengan K-Means**
- Preprocessing & normalisasi data
- Elbow Method untuk menentukan K optimal
- Evaluasi: Silhouette Score & Davies-Bouldin Index
- Clustering berdasarkan: latitude, longitude, brightness, FRP

### 3. **Visualisasi Lengkap**
- ✅ Scatter plots (lokasi, brightness vs FRP)
- ✅ Peta interaktif dengan Folium
- ✅ Heatmap intensitas titik api
- ✅ Visualisasi bounding box & polygon
- ✅ EDA plots (histogram, distribusi)

## 🚀 Setup

### Local Environment
```bash
# Clone repository
git clone https://github.com/akhiyarwaladi/fire_spot.git
cd fire_spot

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Google Colab
```python
# Upload file CSV dan script Python ke Colab

# Install dependencies
!pip install pandas numpy matplotlib seaborn scikit-learn folium shapely

# Jalankan script
!python clustering_fire_spots.py
```

## 📂 Struktur Folder

```
fire_spot/
├── data/
│   ├── raw/                                    # Data mentah
│   │   ├── DL_FIRE_J2V-C2_669817.zip          # Original zip
│   │   └── fire_nrt_J2V-C2_669817.csv         # 57,767 fire spots
│   ├── geojson/                                # GeoJSON boundaries
│   │   └── indonesia-prov-38.json              # 38 provinsi
│   └── filtered/                               # Data per wilayah
│       ├── method_1_bounding_box/
│       │   ├── Riau.csv                       # 3,881 fire spots
│       │   ├── Sumatera_Selatan.csv           # 3,432 fire spots
│       │   └── maps/                          # Visual verification
│       │       ├── overview_all_regions.html
│       │       └── Riau.html
│       └── method_2_geojson/
│           ├── Riau.csv                       # 2,992 fire spots (lebih akurat)
│           └── maps/
│               └── Riau.html
│
├── output/                                     # Hasil clustering
│   └── {dataset_name}/                         # Auto-generated per dataset
│       ├── eda/                               # Exploratory Data Analysis
│       │   ├── statistics.csv
│       │   ├── cluster_stats.csv
│       │   └── distribution.png
│       ├── plots/                             # Scatter plots
│       │   ├── elbow.png
│       │   ├── clusters_location.png
│       │   └── clusters_brightness_frp.png
│       └── maps/                              # Interactive maps
│           ├── clusters.html                  # ← Buka di browser!
│           └── heatmap.html
│
├── filter_by_city_guide.py                    # Filter data per wilayah
├── clustering_fire_spots.py                   # Script utama clustering
└── requirements.txt                           # Dependencies
```

## 🔄 Workflow

### **STEP 1: Filter Data Per Wilayah**

#### Mode Interaktif:
```bash
python filter_by_city_guide.py
```
Pilih opsi:
- `1` = Filter SEMUA wilayah
- `2` = Filter wilayah SUMATERA (10 provinsi)
- `3` = Filter wilayah KALIMANTAN (5 provinsi)
- `4` = Custom

#### Mode Command Line:
```bash
# Bounding Box (cepat)
python filter_by_city_guide.py --sumatera
python filter_by_city_guide.py --kalimantan
python filter_by_city_guide.py --all

# GeoJSON (akurat)
python filter_by_city_guide.py --sumatera --geojson
python filter_by_city_guide.py --all --geojson

# Help
python filter_by_city_guide.py --help
```

**Output:**
- CSV files di `data/filtered/method_1_bounding_box/` atau `method_2_geojson/`
- HTML maps untuk verifikasi visual

---

### **STEP 2: Edit Konfigurasi Clustering**

Buka `clustering_fire_spots.py` dan edit:

```python
# =============================================================================
# ⚙️ KONFIGURASI - EDIT DI SINI
# =============================================================================

# Pilih sumber data
DATA_SOURCE = 'data/raw/fire_nrt_J2V-C2_669817.csv'                    # Semua Indonesia
# DATA_SOURCE = 'data/filtered/method_1_bounding_box/Riau.csv'        # Riau (bounding box)
# DATA_SOURCE = 'data/filtered/method_2_geojson/Riau.csv'             # Riau (geojson)

# Jumlah cluster
NUM_CLUSTERS = 5

# Sampling (None = pakai semua data, atau set angka misal 10000)
SAMPLE_SIZE = 10000  # Ubah ke None untuk pakai semua data
```

---

### **STEP 3: Jalankan Clustering**

```bash
python clustering_fire_spots.py
```

**Proses yang dilakukan:**
1. ✅ Load Data
2. ✅ Exploratory Data Analysis (EDA)
3. ✅ Preprocessing & Normalization
4. ✅ Elbow Method (cari K optimal)
5. ✅ K-Means Clustering
6. ✅ Evaluation Metrics
7. ✅ Generate Visualizations
8. ✅ Create Interactive Maps

---

### **STEP 4: Analisis Hasil**

```bash
# Buka peta interaktif di browser
firefox output/fire_nrt_J2V-C2_669817/maps/clusters.html
firefox output/Riau/maps/clusters.html

# Atau double-click file HTML
```

**File yang dihasilkan:**
- `eda/statistics.csv` - Statistik deskriptif
- `eda/cluster_stats.csv` - Karakteristik setiap cluster
- `plots/elbow.png` - Grafik untuk menentukan K optimal
- `plots/clusters_location.png` - Distribusi spatial
- `maps/clusters.html` - **Peta interaktif** ← BUKA INI!
- `maps/heatmap.html` - Heatmap intensitas

## 🗺️ Metode Filtering

### **Method 1 - Bounding Box** (Lebih Cepat)
✅ Menggunakan kotak latitude/longitude
✅ Cepat untuk dataset besar
⚠️ Bisa include area di luar batas provinsi sebenarnya

**Contoh: Riau = 3,881 fire spots**

### **Method 2 - GeoJSON** (Lebih Akurat)
✅ Menggunakan polygon batas provinsi dari OpenStreetMap
✅ Akurat mengikuti batas wilayah administratif
⚠️ Sedikit lebih lambat (point-in-polygon check)

**Contoh: Riau = 2,992 fire spots (23% lebih akurat!)**

### Perbandingan Visual
Buka kedua map side-by-side untuk membandingkan:
```bash
firefox data/filtered/method_1_bounding_box/maps/Riau.html
firefox data/filtered/method_2_geojson/maps/Riau.html
```

## 🎯 Cara Penggunaan

### **Contoh 1: Analisis Semua Indonesia**
```bash
# Edit clustering_fire_spots.py:
#   DATA_SOURCE = 'data/raw/fire_nrt_J2V-C2_669817.csv'
#   SAMPLE_SIZE = 10000

python clustering_fire_spots.py

# Hasil di: output/fire_nrt_J2V-C2_669817/
```

### **Contoh 2: Analisis Riau (Bounding Box)**
```bash
# 1. Filter Riau
python filter_by_city_guide.py --sumatera

# 2. Edit clustering_fire_spots.py:
#   DATA_SOURCE = 'data/filtered/method_1_bounding_box/Riau.csv'
#   SAMPLE_SIZE = None  # Pakai semua data

# 3. Jalankan clustering
python clustering_fire_spots.py

# Hasil di: output/Riau/
```

### **Contoh 3: Analisis Riau (GeoJSON - Lebih Akurat)**
```bash
# 1. Filter dengan GeoJSON
python filter_by_city_guide.py --sumatera --geojson

# 2. Edit clustering_fire_spots.py:
#   DATA_SOURCE = 'data/filtered/method_2_geojson/Riau.csv'

# 3. Jalankan clustering
python clustering_fire_spots.py

# Hasil lebih akurat karena hanya fire spots dalam batas provinsi
```

### **Contoh 4: Analisis Semua Provinsi Sumatera**
```bash
# Filter 10 provinsi Sumatera
python filter_by_city_guide.py --sumatera --geojson

# Untuk setiap provinsi, ubah DATA_SOURCE dan jalankan clustering
# Misal: Riau, Sumatera_Selatan, Sumatera_Utara, dll
```

## 📊 Provinsi yang Tersedia

Total: **35 provinsi** (fokus pada hotspot area)

**Sumatera** (10 provinsi - HOTSPOT UTAMA):
- Aceh, Sumatera Utara, Riau, Kepulauan Riau
- Sumatera Barat, Jambi, Sumatera Selatan
- Bengkulu, Lampung, Bangka Belitung

**Kalimantan** (5 provinsi - HOTSPOT UTAMA):
- Kalimantan Barat, Tengah, Selatan, Timur, Utara

**Jawa** (6 provinsi):
- Banten, DKI Jakarta, Jawa Barat, Jawa Tengah, DI Yogyakarta, Jawa Timur

**Lainnya**:
- Sulawesi (6), Bali & Nusa Tenggara (3), Maluku & Papua (4)

## 📈 Interpretasi Hasil

### Metrik Evaluasi
- **Silhouette Score**: 0.0 - 1.0 (mendekati 1 = lebih baik)
- **Davies-Bouldin Index**: 0.0 - ∞ (mendekati 0 = lebih baik)

### Karakteristik Cluster (Contoh)
Setelah clustering, lihat `cluster_stats.csv`:
- **Cluster 0**: Intensitas rendah (FRP ≈ 3-5)
- **Cluster 1**: Intensitas sedang (FRP ≈ 6-10)
- **Cluster 2**: Intensitas tinggi (FRP ≈ 15-30)
- dst...

### Elbow Method
Lihat `plots/elbow.png`:
- Titik "siku" pada grafik = K optimal
- Balance antara jumlah cluster dan kualitas clustering

## 🎓 Untuk Mahasiswa

### **Tugas 1: Perbandingan Metode Filtering**
1. Filter Riau dengan kedua metode (bounding box & geojson)
2. Jalankan clustering untuk kedua dataset
3. Bandingkan:
   - Jumlah fire spots
   - Distribusi cluster
   - Silhouette Score
4. Kesimpulan: Metode mana yang lebih baik? Mengapa?

### **Tugas 2: Analisis Multi-Provinsi Sumatera**
1. Filter 5 provinsi Sumatera: Riau, Sumatera Selatan, Sumatera Utara, Jambi, Bengkulu
2. Lakukan clustering untuk masing-masing provinsi
3. Bandingkan:
   - Provinsi mana yang paling banyak fire spots?
   - Provinsi mana yang paling tinggi rata-rata FRP-nya?
   - Pola distribusi cluster: apakah sama atau beda?
4. Interpretasi: Apa penyebab perbedaan antar provinsi?

### **Tugas 3: Optimasi Jumlah Cluster**
1. Pilih 1 provinsi (misal: Riau)
2. Coba K = 3, 5, 7, 10
3. Untuk setiap K:
   - Hitung Silhouette Score
   - Hitung Davies-Bouldin Index
   - Analisis karakteristik cluster
4. K berapa yang optimal? Mengapa?

### **Tugas 4: Feature Engineering**
1. Tambahkan fitur baru: `acq_time` (waktu deteksi)
2. Clustering dengan fitur tambahan
3. Bandingkan hasil dengan clustering tanpa fitur waktu
4. Apakah penambahan fitur waktu meningkatkan kualitas clustering?

### **Tugas 5: Analisis Temporal**
1. Filter data per bulan (Juli, Agustus, September)
2. Clustering untuk setiap bulan
3. Bandingkan pola:
   - Apakah ada perubahan lokasi cluster?
   - Apakah ada perubahan intensitas?
4. Kesimpulan: Bagaimana tren fire spots dari Juli-September?

## 💡 Tips & Troubleshooting

### 1. File tidak ditemukan?
```
❌ File tidak ditemukan: data/filtered/method_1_bounding_box/Riau.csv
💡 Jalankan dulu: python filter_by_city_guide.py --sumatera
```

### 2. Memory Error?
Kurangi `SAMPLE_SIZE`:
```python
SAMPLE_SIZE = 5000  # Atau lebih kecil
```

### 3. Ingin ubah jumlah cluster?
Edit `NUM_CLUSTERS` di `clustering_fire_spots.py`. Lihat grafik `elbow.png` untuk menentukan K optimal.

### 4. Peta tidak muncul?
Pastikan buka file `.html` dengan browser (Firefox, Chrome, dll). Jangan buka dengan text editor.

### 5. Shapely error saat install?
```bash
# Linux
sudo apt-get install libgeos-dev

# Mac
brew install geos

# Windows: shapely sudah include binary
```

## 📦 Dependencies

```
pandas          # Data manipulation
numpy           # Numerical operations
matplotlib      # Static plots
seaborn         # Statistical visualization
scikit-learn    # K-Means clustering
folium          # Interactive maps
shapely         # Geometric operations (GeoJSON)
```

Install semua:
```bash
pip install -r requirements.txt
```

## 📚 Dataset

**Source**: VIIRS (Visible Infrared Imaging Radiometer Suite) Fire Detection
**Period**: 2025-07-01 to 2025-10-01
**Total**: 57,767 fire spots
**Coverage**: Indonesia

**Features:**
- `latitude`, `longitude`: Lokasi
- `brightness`: Kecerahan (Kelvin)
- `frp`: Fire Radiative Power (MW)
- `acq_date`, `acq_time`: Waktu deteksi

**Geographic Distribution:**
- Kalimantan: ~44% (hotspot utama)
- Sumatera: ~25% (hotspot utama)
- Papua: ~18%
- Others: ~13%

## 🤝 Contributing

Project ini untuk keperluan pengajaran Data Mining. Mahasiswa dan dosen dipersilakan untuk:
- Menambahkan metode clustering lain (DBSCAN, Hierarchical, dll)
- Menambahkan fitur baru
- Meningkatkan visualisasi
- Menambahkan dokumentasi

## 📄 License

Educational use only.

---

**Happy Clustering! 🔥📊**

*Dibuat untuk keperluan pengajaran Data Mining - Analisis Clustering Titik Api Indonesia*
