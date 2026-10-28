"""Lokasi data dan parameter yang dipakai bersama."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data/raw/viirs_noaa21_southeast_asia_7d.csv"
BOUNDARIES_PATH = PROJECT_ROOT / "data/boundaries/indonesia_provinces_34.geojson"
OUTPUT_ROOT = PROJECT_ROOT / "output"
FEATURES = ["latitude", "longitude", "brightness", "frp"]
RANDOM_STATE = 42
SILHOUETTE_SAMPLE_SIZE = 2000

# Bounding box merupakan pendekatan kasar dan dapat saling tumpang tindih.
INDONESIA_REGIONS = {
    "Aceh": {"lat": (2.5, 6.0), "lon": (95.0, 98.5)},
    "Sumatera_Utara": {"lat": (1.0, 4.5), "lon": (98.0, 100.5)},
    "Riau": {"lat": (-1.5, 2.5), "lon": (100.0, 105.0)},
    "Kepulauan_Riau": {"lat": (-1.5, 2.0), "lon": (103.0, 108.0)},
    "Sumatera_Barat": {"lat": (-3.5, 0.5), "lon": (98.5, 102.0)},
    "Jambi": {"lat": (-3.0, 0.5), "lon": (101.0, 105.0)},
    "Sumatera_Selatan": {"lat": (-5.0, -1.5), "lon": (102.0, 106.0)},
    "Bengkulu": {"lat": (-5.5, -2.0), "lon": (101.0, 104.0)},
    "Lampung": {"lat": (-6.0, -3.5), "lon": (103.5, 106.0)},
    "Bangka_Belitung": {"lat": (-3.5, -1.5), "lon": (105.0, 108.5)},
    "Banten": {"lat": (-7.0, -5.5), "lon": (105.0, 107.0)},
    "DKI_Jakarta": {"lat": (-6.5, -5.8), "lon": (106.5, 107.2)},
    "Jawa_Barat": {"lat": (-8.0, -5.5), "lon": (106.0, 109.0)},
    "Jawa_Tengah": {"lat": (-8.5, -6.0), "lon": (108.5, 111.5)},
    "DI_Yogyakarta": {"lat": (-8.5, -7.5), "lon": (110.0, 111.0)},
    "Jawa_Timur": {"lat": (-9.0, -6.5), "lon": (111.0, 115.0)},
    "Kalimantan_Barat": {"lat": (-3.5, 2.5), "lon": (108.5, 112.5)},
    "Kalimantan_Tengah": {"lat": (-4.0, 0.5), "lon": (111.0, 115.5)},
    "Kalimantan_Selatan": {"lat": (-4.5, -1.5), "lon": (114.0, 117.0)},
    "Kalimantan_Timur": {"lat": (-2.5, 3.0), "lon": (115.0, 119.5)},
    "Kalimantan_Utara": {"lat": (1.5, 4.5), "lon": (115.5, 118.5)},
    "Sulawesi_Utara": {"lat": (0.0, 2.5), "lon": (123.5, 127.0)},
    "Gorontalo": {"lat": (0.0, 1.5), "lon": (121.5, 123.5)},
    "Sulawesi_Tengah": {"lat": (-3.5, 1.5), "lon": (119.5, 124.0)},
    "Sulawesi_Barat": {"lat": (-3.5, -1.5), "lon": (118.5, 120.0)},
    "Sulawesi_Selatan": {"lat": (-7.0, -2.5), "lon": (118.5, 121.5)},
    "Sulawesi_Tenggara": {"lat": (-6.0, -2.5), "lon": (120.5, 124.0)},
    "Bali": {"lat": (-8.8, -8.0), "lon": (114.5, 115.8)},
    "NTB": {"lat": (-9.5, -8.0), "lon": (115.5, 119.5)},
    "NTT": {"lat": (-11.0, -8.0), "lon": (118.5, 125.0)},
    "Maluku": {"lat": (-9.0, -2.0), "lon": (124.0, 132.0)},
    "Maluku_Utara": {"lat": (-2.5, 3.5), "lon": (124.0, 129.5)},
    "Papua_Barat": {"lat": (-4.5, 1.5), "lon": (130.0, 135.0)},
    "Papua": {"lat": (-9.5, 0.5), "lon": (135.0, 141.5)},
}

SUMATRA = [
    "Aceh",
    "Sumatera_Utara",
    "Riau",
    "Kepulauan_Riau",
    "Sumatera_Barat",
    "Jambi",
    "Sumatera_Selatan",
    "Bengkulu",
    "Lampung",
    "Bangka_Belitung",
]
KALIMANTAN = [name for name in INDONESIA_REGIONS if name.startswith("Kalimantan_")]
