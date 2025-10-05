"""
PANDUAN: CARA FILTER DATA FIRE SPOTS BERDASARKAN KOTA DI INDONESIA
===================================================================

Berdasarkan riset dari berbagai sumber (2024-2025):
- GeoPandas documentation
- Stack Overflow & GIS Stack Exchange
- GitHub: indonesia-geojson, batas-administrasi-indonesia
- Humanitarian Data Exchange (HDX)

Ada 4 CARA UTAMA untuk filter data berdasarkan kota:

CARA 1: BOUNDING BOX (Paling Sederhana) ⭐ RECOMMENDED untuk pemula
CARA 2: GEOPANDAS + GEOJSON (Paling Akurat)
CARA 3: REVERSE GEOCODING API (Paling Fleksibel)
CARA 4: DICTIONARY KOORDINAT MANUAL (Paling Cepat)

Author: Untuk keperluan pengajaran Data Mining
"""

import pandas as pd
import numpy as np

# =============================================================================
# CARA 1: BOUNDING BOX (Koordinat Min/Max) - PALING SEDERHANA ⭐
# =============================================================================
print("="*80)
print("CARA 1: FILTER MENGGUNAKAN BOUNDING BOX (Koordinat Min/Max)")
print("="*80)
print("""
Kelebihan:
- Tidak perlu library tambahan
- Sangat cepat
- Mudah dipahami pemula

Kekurangan:
- Kurang akurat (berbentuk kotak, bukan bentuk sebenarnya)
- Bisa include area di luar kota
""")

# Dictionary bounding box kota-kota besar di Indonesia
# Format: [min_lat, max_lat, min_lon, max_lon]
CITY_BOUNDS = {
    'Jakarta': [-6.35, -6.08, 106.65, 106.98],
    'Surabaya': [-7.35, -7.15, 112.65, 112.85],
    'Bandung': [-7.05, -6.80, 107.50, 107.75],
    'Medan': [3.45, 3.65, 98.60, 98.75],
    'Semarang': [-7.05, -6.90, 110.35, 110.50],
    'Makassar': [-5.20, -5.05, 119.35, 119.50],
    'Palembang': [-3.05, -2.85, 104.65, 104.85],
    'Pekanbaru': [0.45, 0.60, 101.35, 101.50],
    'Balikpapan': [-1.35, -1.15, 116.75, 116.95],
    'Pontianak': [-0.10, 0.10, 109.25, 109.40],
}

def filter_by_bounding_box(df, city_name):
    """
    Filter dataframe berdasarkan bounding box kota

    Parameters:
    - df: pandas DataFrame dengan kolom 'latitude' dan 'longitude'
    - city_name: nama kota (harus ada di CITY_BOUNDS)

    Returns:
    - filtered DataFrame
    """
    if city_name not in CITY_BOUNDS:
        print(f"Error: Kota '{city_name}' tidak ditemukan!")
        print(f"Kota yang tersedia: {list(CITY_BOUNDS.keys())}")
        return None

    min_lat, max_lat, min_lon, max_lon = CITY_BOUNDS[city_name]

    filtered = df[
        (df['latitude'] >= min_lat) &
        (df['latitude'] <= max_lat) &
        (df['longitude'] >= min_lon) &
        (df['longitude'] <= max_lon)
    ]

    print(f"✓ Filter {city_name}: {len(filtered)} data dari {len(df)} total data")
    return filtered


# CONTOH PENGGUNAAN CARA 1:
print("\nCONTOH PENGGUNAAN:")
print("-" * 80)
print("""
# Load data
df = pd.read_csv('fire_nrt_J2V-C2_669817.csv')

# Filter hanya Jakarta
df_jakarta = filter_by_bounding_box(df, 'Jakarta')

# Filter hanya Surabaya
df_surabaya = filter_by_bounding_box(df, 'Surabaya')

# Lalu lakukan clustering seperti biasa pada df_jakarta atau df_surabaya
""")

print()

# =============================================================================
# CARA 2: GEOPANDAS + GEOJSON - PALING AKURAT
# =============================================================================
print("="*80)
print("CARA 2: FILTER MENGGUNAKAN GEOPANDAS + GEOJSON (Polygon Sebenarnya)")
print("="*80)
print("""
Kelebihan:
- Sangat akurat (menggunakan boundary polygon sebenarnya)
- Support semua kota/kabupaten di Indonesia
- Professional dan scalable

Kekurangan:
- Perlu install geopandas
- Perlu download file GeoJSON/Shapefile
- Sedikit lebih kompleks

Sumber Data GeoJSON Indonesia (GRATIS):
1. HDX: https://data.humdata.org/dataset/cod-ab-idn (522 kota/kabupaten)
2. GitHub superpikar: https://github.com/superpikar/indonesia-geojson
3. GitHub Alf-Anas: https://github.com/Alf-Anas/batas-administrasi-indonesia (514 boundaries)
4. GitHub thetrisatria: https://github.com/thetrisatria/geojson-indonesia
""")

print("\nINSTALL LIBRARY:")
print("pip install geopandas shapely")
print()

print("CODE EXAMPLE:")
print("-" * 80)
print("""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 1. Load data fire spots
df = pd.read_csv('fire_nrt_J2V-C2_669817.csv')

# 2. Konversi ke GeoDataFrame
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

# 3. Load GeoJSON kota/kabupaten Indonesia
# Download dari: https://github.com/superpikar/indonesia-geojson
# atau https://data.humdata.org/dataset/cod-ab-idn
indonesia_cities = gpd.read_file('indonesia-kota-kabupaten.geojson')

# 4. Filter untuk kota tertentu (misalnya Jakarta)
jakarta_boundary = indonesia_cities[indonesia_cities['NAME_2'] == 'Jakarta']

# 5. Spatial join - ambil points yang ada di dalam polygon Jakarta
df_jakarta = gpd.sjoin(gdf, jakarta_boundary, predicate='within')

# ATAU gunakan .within() method:
# df_jakarta = gdf[gdf.geometry.within(jakarta_boundary.geometry.iloc[0])]

print(f"Total fire spots di Jakarta: {len(df_jakarta)}")

# 6. Konversi kembali ke pandas DataFrame jika perlu
df_jakarta = pd.DataFrame(df_jakarta.drop(columns='geometry'))
""")

print()

# =============================================================================
# CARA 3: REVERSE GEOCODING API - PALING FLEKSIBEL
# =============================================================================
print("="*80)
print("CARA 3: FILTER MENGGUNAKAN REVERSE GEOCODING API")
print("="*80)
print("""
Kelebihan:
- Otomatis mendapatkan nama kota dari koordinat
- Tidak perlu download shapefile
- Update real-time

Kekurangan:
- Butuh koneksi internet
- Ada rate limit (jumlah request terbatas)
- Lambat untuk data besar (50K+ data)

API yang bisa digunakan (GRATIS):
1. Nominatim (OpenStreetMap) - Gratis, no API key
2. Google Maps Geocoding API - Gratis 40,000 request/bulan
3. Geopy library - wrapper untuk berbagai API
""")

print("\nINSTALL LIBRARY:")
print("pip install geopy")
print()

print("CODE EXAMPLE (Nominatim):")
print("-" * 80)
print("""
import pandas as pd
from geopy.geocoders import Nominatim
from time import sleep

# 1. Load data
df = pd.read_csv('fire_nrt_J2V-C2_669817.csv')

# 2. Sample data (karena API ada rate limit)
df_sample = df.sample(n=100, random_state=42)  # Ambil 100 data saja

# 3. Setup geocoder
geolocator = Nominatim(user_agent="fire_spot_app")

# 4. Function untuk get city name dari koordinat
def get_city(lat, lon):
    try:
        location = geolocator.reverse(f"{lat}, {lon}", language='id')
        address = location.raw.get('address', {})

        # Cari city/town/village
        city = address.get('city') or address.get('town') or \\
               address.get('village') or address.get('municipality', 'Unknown')

        return city
    except:
        return 'Unknown'

# 5. Apply ke dataframe (dengan sleep untuk rate limit)
cities = []
for idx, row in df_sample.iterrows():
    city = get_city(row['latitude'], row['longitude'])
    cities.append(city)
    sleep(1)  # Delay 1 detik per request (rate limit)

    if idx % 10 == 0:
        print(f"Progress: {idx}/{len(df_sample)}")

df_sample['city'] = cities

# 6. Filter berdasarkan kota
df_jakarta = df_sample[df_sample['city'].str.contains('Jakarta', case=False, na=False)]

print(df_sample['city'].value_counts())
""")

print("\nCATATAN: Untuk data besar (50K+), TIDAK DISARANKAN menggunakan API.")
print("Gunakan Cara 1 (Bounding Box) atau Cara 2 (GeoJSON) saja.")
print()

# =============================================================================
# CARA 4: DICTIONARY KOORDINAT MANUAL - PALING CEPAT
# =============================================================================
print("="*80)
print("CARA 4: DICTIONARY DENGAN KOORDINAT RANGE (Manual tapi Cepat)")
print("="*80)
print("""
Sama seperti Cara 1, tapi dengan data lebih lengkap.
Cocok jika Anda hanya butuh beberapa kota tertentu saja.
""")

# Expanded dictionary dengan lebih banyak kota
INDONESIA_CITIES = {
    # Pulau Jawa
    'Jakarta': {'lat': (-6.35, -6.08), 'lon': (106.65, 106.98)},
    'Bogor': {'lat': (-6.65, -6.50), 'lon': (106.75, 106.85)},
    'Depok': {'lat': (-6.45, -6.35), 'lon': (106.75, 106.90)},
    'Tangerang': {'lat': (-6.25, -6.10), 'lon': (106.55, 106.75)},
    'Bekasi': {'lat': (-6.30, -6.15), 'lon': (106.90, 107.10)},
    'Bandung': {'lat': (-7.05, -6.80), 'lon': (107.50, 107.75)},
    'Semarang': {'lat': (-7.05, -6.90), 'lon': (110.35, 110.50)},
    'Yogyakarta': {'lat': (-7.90, -7.75), 'lon': (110.30, 110.45)},
    'Surabaya': {'lat': (-7.35, -7.15), 'lon': (112.65, 112.85)},
    'Malang': {'lat': (-8.00, -7.90), 'lon': (112.60, 112.70)},

    # Pulau Sumatera
    'Medan': {'lat': (3.45, 3.65), 'lon': (98.60, 98.75)},
    'Pekanbaru': {'lat': (0.45, 0.60), 'lon': (101.35, 101.50)},
    'Padang': {'lat': (-1.05, -0.85), 'lon': (100.30, 100.45)},
    'Palembang': {'lat': (-3.05, -2.85), 'lon': (104.65, 104.85)},
    'Jambi': {'lat': (-1.65, -1.50), 'lon': (103.55, 103.70)},
    'Bengkulu': {'lat': (-3.85, -3.75), 'lon': (102.25, 102.35)},
    'Bandar Lampung': {'lat': (-5.50, -5.35), 'lon': (105.20, 105.35)},
    'Banda Aceh': {'lat': (5.50, 5.60), 'lon': (95.30, 95.40)},

    # Pulau Kalimantan
    'Pontianak': {'lat': (-0.10, 0.10), 'lon': (109.25, 109.40)},
    'Palangkaraya': {'lat': (-2.30, -2.15), 'lon': (113.85, 114.00)},
    'Banjarmasin': {'lat': (-3.35, -3.25), 'lon': (114.55, 114.65)},
    'Balikpapan': {'lat': (-1.35, -1.15), 'lon': (116.75, 116.95)},
    'Samarinda': {'lat': (-0.60, -0.40), 'lon': (117.10, 117.20)},

    # Pulau Sulawesi
    'Makassar': {'lat': (-5.20, -5.05), 'lon': (119.35, 119.50)},
    'Manado': {'lat': (1.45, 1.50), 'lon': (124.80, 124.90)},
    'Palu': {'lat': (-0.95, -0.85), 'lon': (119.85, 119.95)},
    'Kendari': {'lat': (-4.00, -3.95), 'lon': (122.50, 122.60)},

    # Bali & Nusa Tenggara
    'Denpasar': {'lat': (-8.75, -8.60), 'lon': (115.15, 115.30)},
    'Mataram': {'lat': (-8.65, -8.55), 'lon': (116.05, 116.15)},
    'Kupang': {'lat': (-10.25, -10.15), 'lon': (123.55, 123.65)},

    # Maluku & Papua
    'Ambon': {'lat': (-3.75, -3.65), 'lon': (128.15, 128.25)},
    'Jayapura': {'lat': (-2.65, -2.50), 'lon': (140.65, 140.75)},
}

def filter_by_city_advanced(df, city_name):
    """
    Filter dataframe berdasarkan kota dengan dictionary yang lebih lengkap
    """
    if city_name not in INDONESIA_CITIES:
        print(f"Error: Kota '{city_name}' tidak ditemukan!")
        print(f"\nKota yang tersedia ({len(INDONESIA_CITIES)}):")
        for region, cities in [
            ('Jawa', ['Jakarta', 'Bogor', 'Depok', 'Tangerang', 'Bekasi', 'Bandung',
                     'Semarang', 'Yogyakarta', 'Surabaya', 'Malang']),
            ('Sumatera', ['Medan', 'Pekanbaru', 'Padang', 'Palembang', 'Jambi',
                         'Bengkulu', 'Bandar Lampung', 'Banda Aceh']),
            ('Kalimantan', ['Pontianak', 'Palangkaraya', 'Banjarmasin', 'Balikpapan', 'Samarinda']),
            ('Sulawesi', ['Makassar', 'Manado', 'Palu', 'Kendari']),
            ('Bali & Nusa Tenggara', ['Denpasar', 'Mataram', 'Kupang']),
            ('Maluku & Papua', ['Ambon', 'Jayapura'])
        ]:
            print(f"  {region}: {', '.join(cities)}")
        return None

    city_bounds = INDONESIA_CITIES[city_name]
    min_lat, max_lat = city_bounds['lat']
    min_lon, max_lon = city_bounds['lon']

    filtered = df[
        (df['latitude'] >= min_lat) &
        (df['latitude'] <= max_lat) &
        (df['longitude'] >= min_lon) &
        (df['longitude'] <= max_lon)
    ]

    print(f"✓ Filter {city_name}: {len(filtered)} data dari {len(df)} total data")
    return filtered


print("\nCONTOH PENGGUNAAN:")
print("-" * 80)
print("""
# Load data
df = pd.read_csv('fire_nrt_J2V-C2_669817.csv')

# Filter berbagai kota
df_jakarta = filter_by_city_advanced(df, 'Jakarta')
df_surabaya = filter_by_city_advanced(df, 'Surabaya')
df_medan = filter_by_city_advanced(df, 'Medan')
df_makassar = filter_by_city_advanced(df, 'Makassar')

# Lihat statistik per kota
for city in ['Jakarta', 'Surabaya', 'Medan', 'Bandung']:
    city_data = filter_by_city_advanced(df, city)
    if city_data is not None and len(city_data) > 0:
        avg_frp = city_data['frp'].mean()
        print(f"{city}: {len(city_data)} titik api, rata-rata FRP: {avg_frp:.2f}")
""")

print()

# =============================================================================
# RINGKASAN & REKOMENDASI
# =============================================================================
print("="*80)
print("RINGKASAN & REKOMENDASI")
print("="*80)
print("""
┌─────────────────────┬──────────────┬────────────┬─────────────┬──────────────┐
│ METODE              │ AKURASI      │ KECEPATAN  │ KOMPLEKSITAS│ REKOMENDASI  │
├─────────────────────┼──────────────┼────────────┼─────────────┼──────────────┤
│ 1. Bounding Box     │ ⭐⭐⭐       │ ⭐⭐⭐⭐⭐ │ ⭐          │ ✓ PEMULA     │
│ 2. GeoJSON/Shapefile│ ⭐⭐⭐⭐⭐   │ ⭐⭐⭐⭐   │ ⭐⭐⭐      │ ✓ ADVANCED   │
│ 3. Reverse Geocoding│ ⭐⭐⭐⭐     │ ⭐         │ ⭐⭐        │ ✗ Data Besar │
│ 4. Dictionary Manual│ ⭐⭐⭐       │ ⭐⭐⭐⭐⭐ │ ⭐          │ ✓ Quick Fix  │
└─────────────────────┴──────────────┴────────────┴─────────────┴──────────────┘

REKOMENDASI BERDASARKAN KEBUTUHAN:
───────────────────────────────────

👨‍🎓 Untuk MAHASISWA/PEMULA:
   → Gunakan CARA 1 atau CARA 4 (Bounding Box)
   Alasan: Simpel, cepat, tidak perlu install library tambahan

👨‍💼 Untuk PENELITIAN/PROFESSIONAL:
   → Gunakan CARA 2 (GeoJSON + GeoPandas)
   Alasan: Akurat, reproducible, support semua kota

⚡ Untuk DEMO/PROTOTYPE CEPAT:
   → Gunakan CARA 4 (Dictionary lengkap)
   Alasan: Tinggal copy-paste, langsung jalan

🌐 Untuk WEB APPLICATION:
   → Gunakan CARA 2 dengan caching
   Alasan: Scalable dan bisa dioptimasi

NEXT STEPS:
-----------
1. Pilih metode yang sesuai kebutuhan Anda
2. Lihat file: clustering_with_city_filter.py untuk implementasi lengkap
3. Download GeoJSON jika pilih Cara 2:
   https://github.com/Alf-Anas/batas-administrasi-indonesia
""")

print("="*80)
print("SELESAI! Semoga membantu 🚀")
print("="*80)
