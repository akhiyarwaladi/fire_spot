#!/bin/bash
# Quick demo script untuk generate peta berbagai kota

echo "🚀 Generating maps untuk beberapa kota..."
echo ""

# Aktifkan virtual environment
source venv/bin/activate

# Function untuk ubah kota dan run
generate_map() {
    CITY=$1
    echo "📍 Processing: $CITY"

    # Replace CITY_NAME di script
    sed -i "s/CITY_NAME = '.*'/CITY_NAME = '$CITY'/" clustering_with_city_filter.py

    # Run script
    python clustering_with_city_filter.py 2>&1 | grep -E "(Ditemukan|Saved|SELESAI)"

    echo ""
}

# Generate untuk beberapa kota
generate_map "Surabaya"
generate_map "Bandung"
generate_map "Medan"

echo "✅ Selesai! Cek file HTML yang dihasilkan."
