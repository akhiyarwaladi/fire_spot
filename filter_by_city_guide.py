"""
FILTER DATA FIRE SPOTS BERDASARKAN KOTA
========================================

Script ini akan memfilter data fire spots berdasarkan kota dan menyimpannya
ke dalam folder data/filtered/ untuk digunakan oleh clustering_fire_spots.py

CARA PAKAI:
-----------
1. Jalankan script ini: python filter_by_city_guide.py
2. Pilih mode (all/major/custom cities)
3. Data akan disimpan di: data/filtered/method_1_bounding_box/

OUTPUT:
-------
data/filtered/method_1_bounding_box/
    ├── Jakarta.csv
    ├── Surabaya.csv
    ├── Bandung.csv
    └── ...

Author: Untuk keperluan pengajaran Data Mining
"""

import pandas as pd
import os

# =============================================================================
# KOTA-KOTA DI INDONESIA DENGAN BOUNDING BOX
# =============================================================================

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

# =============================================================================
# FUNGSI FILTER
# =============================================================================

def filter_by_bounding_box(df, city_name):
    """
    Filter dataframe berdasarkan bounding box kota

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame dengan kolom 'latitude' dan 'longitude'
    city_name : str
        Nama kota (harus ada di INDONESIA_CITIES)

    Returns:
    --------
    pandas.DataFrame atau None
    """
    if city_name not in INDONESIA_CITIES:
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
# FUNGSI UTAMA
# =============================================================================

def generate_filtered_datasets(cities_to_filter=None):
    """
    Generate filtered datasets untuk kota-kota yang dipilih

    Parameters:
    -----------
    cities_to_filter : list atau None
        List nama kota. Jika None, akan filter semua kota yang ada
    """

    print("="*80)
    print("GENERATE FILTERED DATA PER KOTA")
    print("="*80)
    print()

    # Load raw data
    print("📂 Loading raw data...")
    raw_data_path = 'data/raw/fire_nrt_J2V-C2_669817.csv'

    if not os.path.exists(raw_data_path):
        print(f"   ❌ File tidak ditemukan: {raw_data_path}")
        print()
        print("   💡 Pastikan file CSV ada di folder data/raw/")
        return

    df = pd.read_csv(raw_data_path)
    print(f"   ✓ Total data: {len(df):,} fire spots")
    print(f"   📅 Periode: {df['acq_date'].min()} - {df['acq_date'].max()}")
    print()

    # Determine which cities to filter
    if cities_to_filter is None:
        cities_to_filter = list(INDONESIA_CITIES.keys())

    # Create output directory
    output_dir = 'data/filtered/method_1_bounding_box'
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 Output directory: {output_dir}/")
    print(f"🏙️  Kota yang akan difilter: {len(cities_to_filter)} kota")
    print()

    # Filter and save for each city
    print("="*80)
    print("PROCESSING...")
    print("="*80)

    results = []
    for i, city in enumerate(cities_to_filter, 1):
        print(f"\n[{i}/{len(cities_to_filter)}] {city}")

        # Filter data
        df_city = filter_by_bounding_box(df, city)

        if df_city is None:
            print(f"   ⚠️  Kota tidak ditemukan di database")
            continue

        num_points = len(df_city)

        if num_points == 0:
            print(f"   ⚠️  Tidak ada fire spots")
            results.append({
                'city': city,
                'status': 'no_data',
                'num_points': 0
            })
            continue

        # Save to CSV
        filename = f"{city.replace(' ', '_')}.csv"
        file_path = os.path.join(output_dir, filename)
        df_city.to_csv(file_path, index=False)

        # Calculate statistics
        avg_frp = df_city['frp'].mean()
        date_range = f"{df_city['acq_date'].min()} - {df_city['acq_date'].max()}"

        print(f"   ✓ {num_points:,} fire spots")
        print(f"   📊 Avg FRP: {avg_frp:.2f}")
        print(f"   📅 {date_range}")
        print(f"   💾 Saved: {filename}")

        results.append({
            'city': city,
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

    print(f"\n✅ Berhasil: {len(successful)} kota")
    print(f"⚠️  Tidak ada data: {len(no_data)} kota")
    print()

    if successful:
        print("Top 10 kota dengan fire spots terbanyak:")
        print("-" * 70)
        sorted_results = sorted(successful, key=lambda x: x['num_points'], reverse=True)[:10]
        for i, r in enumerate(sorted_results, 1):
            print(f"{i:2d}. {r['city']:20s} : {r['num_points']:6,} fire spots (avg FRP: {r['avg_frp']:6.2f})")

    print()
    print(f"📁 Semua file disimpan di: {output_dir}/")
    print()
    print("="*80)
    print("✅ SELESAI!")
    print("="*80)
    print()
    print("Next step:")
    print("  1. Edit clustering_fire_spots.py")
    print("  2. Ubah DATA_SOURCE ke salah satu file yang sudah dibuat")
    print(f"     Contoh: DATA_SOURCE = '{output_dir}/Jakarta.csv'")
    print("  3. Jalankan: python clustering_fire_spots.py")
    print()

    return results

# =============================================================================
# MODE INTERAKTIF
# =============================================================================

def interactive_mode():
    """Mode interaktif untuk memilih kota"""

    print("="*80)
    print("FILTER DATA FIRE SPOTS PER KOTA")
    print("="*80)
    print()

    print("Pilihan:")
    print("  1. Filter SEMUA kota (34 kota)")
    print("  2. Filter kota-kota UTAMA saja (10 kota)")
    print("  3. Custom - pilih kota sendiri")
    print()

    choice = input("Pilihan Anda (1/2/3): ").strip()

    if choice == '1':
        cities = None  # Semua kota
        print("\n✓ Akan memfilter SEMUA kota (34 kota)")
    elif choice == '2':
        cities = ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Makassar',
                  'Semarang', 'Palembang', 'Pekanbaru', 'Denpasar', 'Balikpapan']
        print(f"\n✓ Akan memfilter {len(cities)} kota utama")
    elif choice == '3':
        print("\nKota yang tersedia:")
        cities_list = list(INDONESIA_CITIES.keys())
        for i in range(0, len(cities_list), 5):
            print("  " + ", ".join(cities_list[i:i+5]))
        print()
        cities_input = input("Masukkan nama kota (pisahkan dengan koma): ").strip()
        cities = [c.strip() for c in cities_input.split(',')]
        print(f"\n✓ Akan memfilter {len(cities)} kota")
    else:
        print("\n❌ Pilihan tidak valid. Menggunakan mode default (10 kota utama)")
        cities = ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Makassar',
                  'Semarang', 'Palembang', 'Pekanbaru', 'Denpasar', 'Balikpapan']

    print()
    confirm = input("Lanjutkan? (y/n): ").strip().lower()

    if confirm == 'y':
        return generate_filtered_datasets(cities_to_filter=cities)
    else:
        print("\n❌ Dibatalkan")
        return None

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            # Filter all cities
            generate_filtered_datasets(cities_to_filter=None)
        elif sys.argv[1] == '--major':
            # Filter major cities only
            major_cities = ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Makassar',
                           'Semarang', 'Palembang', 'Pekanbaru', 'Denpasar', 'Balikpapan']
            generate_filtered_datasets(cities_to_filter=major_cities)
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python filter_by_city_guide.py              # Interactive mode")
            print("  python filter_by_city_guide.py --all        # Filter all 34 cities")
            print("  python filter_by_city_guide.py --major      # Filter 10 major cities")
        else:
            print("❌ Invalid argument. Use --help for usage information.")
    else:
        # Interactive mode
        interactive_mode()
