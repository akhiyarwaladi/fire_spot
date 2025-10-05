# 🔥 WORKFLOW CLUSTERING FIRE SPOTS

Panduan lengkap untuk menjalankan analisis clustering fire spots per kota.

## 📁 Struktur Folder

```
fire_spot/
├── data/
│   ├── raw/                              # Data mentah
│   │   └── fire_nrt_J2V-C2_669817.csv   # Dataset asli
│   └── filtered/                         # Data yang sudah difilter per kota
│       └── method_1_bounding_box/
│           ├── Jakarta.csv
│           ├── Surabaya.csv
│           └── ...
│
├── output/                               # Hasil clustering
│   └── method_1_bounding_box/
│       ├── jakarta/
│       │   ├── eda/                     # Exploratory Data Analysis
│       │   │   ├── dataset_info.txt
│       │   │   ├── descriptive_statistics.csv
│       │   │   ├── cluster_statistics.csv
│       │   │   └── data_distribution.png
│       │   ├── plots/                   # Visualisasi scatter plots
│       │   │   ├── elbow_method.png
│       │   │   ├── clustering_location.png
│       │   │   └── clustering_brightness_frp.png
│       │   └── maps/                    # Peta interaktif
│       │       ├── interactive_clusters.html
│       │       └── heatmap.html
│       └── surabaya/
│           └── ...
│
├── filter_by_city_guide.py              # Script untuk filter data per kota
├── clustering_fire_spots.py             # Script utama untuk clustering
└── requirements.txt                     # Dependencies

```

## 🚀 WORKFLOW

### **STEP 1: Filter Data Per Kota**

Jalankan `filter_by_city_guide.py` untuk memfilter data berdasarkan kota:

#### **Mode Interaktif:**
```bash
python filter_by_city_guide.py
```

Pilih opsi:
- `1` = Filter SEMUA kota (34 kota)
- `2` = Filter kota-kota UTAMA (10 kota): Jakarta, Surabaya, Bandung, dll
- `3` = Custom - pilih kota sendiri

#### **Mode Command Line:**
```bash
# Filter semua kota
python filter_by_city_guide.py --all

# Filter kota utama saja
python filter_by_city_guide.py --major

# Lihat help
python filter_by_city_guide.py --help
```

**Output:**
- File CSV per kota disimpan di: `data/filtered/method_1_bounding_box/`
- Contoh: `Jakarta.csv`, `Surabaya.csv`, dll

---

### **STEP 2: Edit Konfigurasi Clustering**

Buka file `clustering_fire_spots.py` dan edit bagian **KONFIGURASI** (baris 31-54):

```python
# =============================================================================
# ⚙️ KONFIGURASI - EDIT DI SINI
# =============================================================================

# Pilih sumber data
DATA_SOURCE = 'data/filtered/method_1_bounding_box/Jakarta.csv'  # Ubah sesuai kota

# Nama kota (untuk penamaan output)
CITY_NAME = 'Jakarta'  # Sesuaikan dengan file data

# Metode filter yang digunakan
METHOD = 'method_1_bounding_box'

# Clustering settings
NUM_CLUSTERS = 3  # Ubah jumlah cluster jika perlu
RANDOM_STATE = 42

# Sampling (None untuk semua data)
SAMPLE_SIZE = None

# Visualization settings
PLOT_DPI = 300
MAP_ZOOM = 12
MAX_MARKERS = 300
```

**Yang perlu diubah:**
- `DATA_SOURCE`: Path ke file CSV hasil filter (STEP 1)
- `CITY_NAME`: Nama kota (untuk penamaan output)
- `NUM_CLUSTERS`: Jumlah cluster (default: 3)

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

**Output:**
```
output/method_1_bounding_box/jakarta/
├── eda/
│   ├── dataset_info.txt              # Info dataset
│   ├── descriptive_statistics.csv    # Statistik deskriptif
│   ├── cluster_statistics.csv        # Statistik per cluster
│   ├── missing_values.csv            # Check missing values
│   └── data_distribution.png         # Histogram distribusi

├── plots/
│   ├── elbow_method.png              # Grafik elbow + silhouette
│   ├── clustering_location.png       # Scatter plot lokasi
│   └── clustering_brightness_frp.png # Scatter plot brightness vs FRP

└── maps/
    ├── interactive_clusters.html     # Peta interaktif (BUKA DI BROWSER!)
    └── heatmap.html                  # Heatmap intensitas
```

---

### **STEP 4: Analisis Hasil**

1. **Buka peta interaktif:**
   ```bash
   firefox output/method_1_bounding_box/jakarta/maps/interactive_clusters.html
   # atau double-click file HTML nya
   ```

2. **Analisis visualisasi:**
   - `elbow_method.png`: Lihat K optimal
   - `clustering_location.png`: Lihat distribusi spatial cluster
   - `clustering_brightness_frp.png`: Lihat karakteristik intensitas

3. **Baca statistik:**
   - `cluster_statistics.csv`: Karakteristik setiap cluster
   - `descriptive_statistics.csv`: Statistik deskriptif data

---

## 🎯 CONTOH PENGGUNAAN

### **Contoh 1: Analisis Jakarta**

```bash
# 1. Filter data Jakarta
python filter_by_city_guide.py --major

# 2. Edit clustering_fire_spots.py:
#    DATA_SOURCE = 'data/filtered/method_1_bounding_box/Jakarta.csv'
#    CITY_NAME = 'Jakarta'
#    NUM_CLUSTERS = 3

# 3. Jalankan clustering
python clustering_fire_spots.py

# 4. Buka hasil
firefox output/method_1_bounding_box/jakarta/maps/interactive_clusters.html
```

### **Contoh 2: Analisis Surabaya dengan K=5**

```bash
# 1. Data sudah ada (dari step sebelumnya)

# 2. Edit clustering_fire_spots.py:
#    DATA_SOURCE = 'data/filtered/method_1_bounding_box/Surabaya.csv'
#    CITY_NAME = 'Surabaya'
#    NUM_CLUSTERS = 5

# 3. Jalankan clustering
python clustering_fire_spots.py

# Output: output/method_1_bounding_box/surabaya/
```

### **Contoh 3: Analisis Custom Cities**

```bash
# 1. Filter kota-kota tertentu
python filter_by_city_guide.py
# Pilih opsi 3, masukkan: Bandung, Semarang, Yogyakarta

# 2. Analisis Bandung
# Edit: DATA_SOURCE = 'data/filtered/method_1_bounding_box/Bandung.csv'
#       CITY_NAME = 'Bandung'
python clustering_fire_spots.py

# 3. Analisis Semarang
# Edit: DATA_SOURCE = 'data/filtered/method_1_bounding_box/Semarang.csv'
#       CITY_NAME = 'Semarang'
python clustering_fire_spots.py
```

---

## 📊 KOTA YANG TERSEDIA

Total: **34 kota** di seluruh Indonesia

**Pulau Jawa** (10):
Jakarta, Bogor, Depok, Tangerang, Bekasi, Bandung, Semarang, Yogyakarta, Surabaya, Malang

**Pulau Sumatera** (8):
Medan, Pekanbaru, Padang, Palembang, Jambi, Bengkulu, Bandar Lampung, Banda Aceh

**Pulau Kalimantan** (5):
Pontianak, Palangkaraya, Banjarmasin, Balikpapan, Samarinda

**Pulau Sulawesi** (4):
Makassar, Manado, Palu, Kendari

**Bali & Nusa Tenggara** (3):
Denpasar, Mataram, Kupang

**Maluku & Papua** (2):
Ambon, Jayapura

---

## 💡 TIPS & TROUBLESHOOTING

### **1. File tidak ditemukan?**
```
❌ File tidak ditemukan: data/filtered/method_1_bounding_box/Jakarta.csv
💡 Jalankan dulu: python filter_by_city_guide.py
```

**Solusi:** Jalankan STEP 1 dulu untuk generate filtered data.

### **2. Tidak ada fire spots di kota tertentu?**
Beberapa kota mungkin tidak memiliki data fire spots dalam periode dataset. Coba kota lain.

### **3. Ingin ubah jumlah cluster?**
Edit `NUM_CLUSTERS` di `clustering_fire_spots.py`. Lihat grafik `elbow_method.png` untuk menentukan K optimal.

### **4. Output terlalu banyak marker di peta?**
Edit `MAX_MARKERS` di `clustering_fire_spots.py` (default: 300 per cluster).

### **5. Ingin analisis semua data (tanpa filter kota)?**
Edit `DATA_SOURCE = 'data/raw/fire_nrt_J2V-C2_669817.csv'` dan `CITY_NAME = 'Indonesia'`

---

## 📚 UNTUK MAHASISWA

### **Tugas 1: Analisis Clustering Jakarta**
1. Filter data Jakarta
2. Lakukan clustering dengan K=3, K=5, K=7
3. Bandingkan Silhouette Score dan Davies-Bouldin Index
4. Interpretasi hasil: cluster mana yang paling tinggi intensitas apinya?

### **Tugas 2: Perbandingan Antar Kota**
1. Filter data 5 kota: Jakarta, Surabaya, Bandung, Medan, Makassar
2. Lakukan clustering untuk masing-masing kota
3. Bandingkan:
   - Jumlah fire spots
   - Rata-rata FRP
   - Distribusi cluster
4. Kota mana yang paling banyak titik api?

### **Tugas 3: Eksplorasi Metode**
1. Coba ubah fitur clustering (tambah/kurangi features)
2. Coba ubah normalization method
3. Bandingkan hasil dengan metode berbeda
4. Tulis laporan analisis

---

## 🔧 REQUIREMENTS

```bash
pip install -r requirements.txt
```

Dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- folium

---

## 📞 BANTUAN

Jika ada pertanyaan atau error, cek:
1. `WORKFLOW.md` (file ini)
2. `README.md` (dokumentasi umum)
3. Comment di dalam script Python

---

**Happy Clustering! 🔥📊**
