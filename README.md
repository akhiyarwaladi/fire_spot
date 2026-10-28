# Titik api VIIRS Indonesia

Unduh deteksi VIIRS NOAA-21 terbaru dari NASA FIRMS, pilih wilayah, lalu lihat kelompok deteksi di peta. Analisis memakai K-Means, dengan wilayah default Sumatra dan jumlah kelompok awal K=5. Kode memakai modul Python biasa, tanpa framework aplikasi atau notebook.

## Mulai

Gunakan Python 3.10 atau lebih baru. Jalankan dari folder repository:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m fire_spot download
python -m fire_spot analyze
```

Di Linux/macOS, aktivasi dengan `source .venv/bin/activate`. Jika PowerShell membatasi aktivasi, gunakan `.venv\Scripts\python.exe` sebagai pengganti `python`.

CSV snapshot terbaru sudah disertakan, sehingga `analyze` juga bisa dijalankan tanpa mengunduh ulang. Perintah `download` mengambil feed publik tujuh hari; tidak memerlukan akun atau API key. CSV asli disimpan bersama tanggal pengamatan, URL sumber, waktu unduh, dan checksum SHA-256 dalam berkas `.source.json`. Feed yang deteksi terakhirnya lebih tua dari tiga hari ditolak agar tidak mengganti snapshot lokal dengan data usang. Unduhan gagal tidak menimpa CSV yang sudah ada.

Buka hasil melalui HTTP lokal:

```powershell
python -m http.server 8765 --bind 127.0.0.1
```

[Peta Sumatra](http://127.0.0.1:8765/output/sumatra/maps/index.html). Hentikan server dengan `Ctrl+C` setelah selesai.

## Snapshot bawaan

Data diunduh pada **28 Oktober 2026**. Feed NOAA-21 memuat **58.158 deteksi Asia Tenggara** dengan waktu pengamatan 21 Oktober 2026 pukul 03.06 UTC sampai 27 Oktober 2026 pukul 20.34 UTC. Tanggal unduh berbeda dari tanggal pengamatan.

Setelah validasi, penyaringan confidence nominal/tinggi, dan pemilihan polygon Indonesia:

| Wilayah | Deteksi |
| --- | ---: |
| Indonesia | 38.232 |
| Sumatra, termasuk provinsi kepulauannya | 5.116 |
| Kalimantan | 22.210 |
| Sumatera Selatan | 3.504 |
| Jambi | 663 |
| Lampung | 513 |

Jumlah kelompok ditentukan melalui `--clusters` (default K=5). Angka deteksi di atas menggambarkan snapshot bawaan, bukan data langsung yang diperbarui otomatis. Jalankan `download` dan `analyze` untuk memperbarui hasil.

## Pilihan analisis

```powershell
# K-Means pada seluruh Sumatra, K=5
python -m fire_spot analyze
python -m fire_spot analyze --clusters 3

# Satu provinsi atau seluruh Indonesia
python -m fire_spot analyze --region jambi --clusters 4
python -m fire_spot analyze --region indonesia --clusters 5

python -m fire_spot analyze --help
```

`--data` memilih CSV lain; `--boundaries` memilih GeoJSON lain. `--basemap topo` memilih OpenTopoMap. `--include-low-confidence` menyertakan confidence rendah. Default path berasal dari lokasi repository; path yang diberikan lewat CLI mengikuti folder kerja terminal. Pemanggilan `python -m fire_spot` dilakukan dari repository.

Alurnya: baca dan validasi data, pilih wilayah, standardisasi empat fitur dengan `StandardScaler`, jalankan `KMeans`, lalu simpan evaluasi dan peta. Fitur yang dipakai adalah `latitude`, `longitude`, `brightness`, dan `frp`. `random_state=42` dan `n_init=10` membuat percobaan dapat diulang dengan input dan versi pustaka yang sama.

Jumlah kelompok mengikuti `--clusters`; K=5 berarti data dibagi menjadi lima kelompok, bukan ditemukan lima kejadian kebakaran. Kurva elbow dan silhouette menguji K dari dua sampai `--max-k` (default 10), dibatasi jumlah baris dan kombinasi fitur unik. Kurva membantu membandingkan pilihan K; K akhir tetap mengikuti pilihan pengguna. Silhouette memakai sampel tetap maksimal 2.000 titik, sedangkan Davies-Bouldin memakai seluruh data analisis. Jika sampel tidak mendukung perhitungan silhouette, skor dicatat kosong.

K-Means menghitung jarak pada fitur yang distandardisasi, bukan kilometer. Anggota satu cluster dapat berjauhan karena pengelompokan juga mempertimbangkan brightness dan FRP. Nomor cluster bukan peringkat bahaya. Seluruh tanggal dalam feed digabung; pengamatan pada hari berbeda tetap dipertahankan.

## Filter CSV per provinsi

```powershell
# Default: provinsi Sumatra, menggunakan polygon
python -m fire_spot filter
python -m fire_spot filter --regions Jambi Riau
python -m fire_spot filter --regions "Kepulauan Riau"
python -m fire_spot filter --all --no-maps

# Kotak perkiraan, untuk membandingkan metode
python -m fire_spot filter --sumatra --method bounding_box
```

Hasil berada di `output/filtered/geojson/` atau `output/filtered/bounding_box/`. Setiap provinsi mendapat CSV, termasuk yang kosong; `summary.csv` mencatat jumlah deteksi. Satu peta `regions.html` menampilkan semua wilayah terpilih. `--no-maps` hanya memperbarui CSV dan ringkasan. Bounding box dapat tumpang tindih dan memasukkan deteksi dari provinsi tetangga.

Batas bawaan bernama `indonesia_provinces_34.geojson` karena berisi **34 fitur provinsi**, bukan 38. Berkas ini belum mengikuti pemekaran provinsi terbaru di Papua. Beberapa geometri diperbaiki di memori dengan Shapely `make_valid` dan dilaporkan melalui log; sumber tidak ditulis ulang. Untuk statistik lintas Indonesia, setiap deteksi diberi maksimal satu provinsi agar tidak dihitung ganda di batas yang bersinggungan.

## Struktur dan hasil

```text
fire_spot/
  __main__.py              # Pintu masuk python -m fire_spot
  cli.py                   # Perintah download, filter, analyze
  config.py                # Path default dan daftar wilayah
  download.py              # Unduhan resmi NASA dan catatan sumber
  data.py                  # Pembacaan dan validasi CSV
  filtering.py             # Batas provinsi dan pemilihan wilayah
  analysis.py              # K-Means, evaluasi, ringkasan, dan grafik
  maps.py                  # Peta Folium dan kontrol tampilannya
data/
  boundaries/
    indonesia_provinces_34.geojson
  raw/
    viirs_noaa21_southeast_asia_7d.csv
    viirs_noaa21_southeast_asia_7d.source.json
tests/
output/                    # Hasil turunan, diabaikan Git
  sumatra/
    detections.csv         # Data valid, metadata, provinsi, dan label cluster
    clusters.csv           # Jumlah deteksi, pusat, FRP, dan tanggal per cluster
    k_scores.csv           # Inertia dan silhouette untuk kandidat K
    daily_counts.csv       # Deteksi harian dalam UTC, termasuk hari tanpa deteksi
    province_counts.csv    # Jumlah deteksi per provinsi pada wilayah analisis
    indonesia_counts.csv   # Pembanding provinsi Indonesia dari feed yang sama
    metrics.json           # Parameter, jumlah, dan sumber yang dapat diperiksa
    report.md              # Temuan dan batas interpretasi
    charts/overview.png
    charts/elbow.png
    maps/
      index.html
      clusters.html
      heatmap.html
```

Semua nama hasil memakai huruf kecil dengan pemisah garis bawah. Hasil filter, peta, dan grafik tidak disimpan di dalam `data/`. Berkas mentah Juli 2025 dan hasil lama sudah dikeluarkan dari alur aktif. Enam variasi peta lama diringkas menjadi dua; lapisan WMS WorldCover yang tidak stabil juga dihapus.

Pengulangan analisis untuk wilayah dan folder output yang sama menimpa hasil sebelumnya. Gunakan `--output-dir` berbeda untuk membandingkan nilai K. `clusters.csv` merangkum setiap kelompok; seluruh deteksi beserta labelnya ada di `detections.csv`.

## Membaca hasil secara tepat

VIIRS mendeteksi anomali panas pada piksel sekitar 375 m. Deteksi dapat berasal dari kebakaran vegetasi, pembakaran pertanian, gunung api, atau sumber panas lain. Satu kebakaran dapat menghasilkan banyak deteksi dan diamati kembali oleh satelit. Karena itu, kelompok pada peta merupakan **kandidat area aktivitas panas**, sedangkan jumlah cluster belum sama dengan jumlah kejadian karhutla. Verifikasi membutuhkan tutupan lahan, citra lain, atau informasi lapangan. Rujukan: [deskripsi produk VIIRS NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html).

Data ini dapat menunjukkan pemusatan deteksi terkini di Sumatera Selatan. Untuk menyimpulkan kenaikan karhutla, perlu periode pembanding dengan sensor, cakupan waktu, dan penyaringan yang sama. Jumlah harian juga dipengaruhi awan, lintasan, dan keterlambatan pengolahan NRT; hari di tepi jendela dapat parsial. Snapshot tujuh hari ini belum membuktikan kenaikan dibanding minggu sebelumnya.

Kolom `bright_ti4` dari feed baru dibaca sebagai `brightness`; CSV asli tetap utuh. Confidence `nominal/high/low` maupun `n/h/l` dikenali. Baris dengan fitur kosong, tak hingga, koordinat tidak valid, brightness tidak positif, atau FRP negatif diabaikan dengan log. Duplikat baris identik dihapus; pengamatan pada waktu berbeda tetap dipertahankan. Analisis periode membutuhkan `acq_date` yang valid.

Pada peta cluster, lingkaran bernomor menggabungkan marker untuk tampilan saat zoom jauh. Klik untuk memperbesar; warna tiap titik mengikuti label hasil analisis. Angka pada lingkaran merupakan jumlah marker yang digabung, sedangkan jumlah cluster analisis tercatat di `metrics.json`.

Heatmap memakai bobot `FRP / FRP maksimum dataset`, sehingga warnanya relatif dan juga dipengaruhi kepadatan serta zoom. Grafik harian memakai UTC, bukan waktu setempat. Deteksi di luar batas polygon tidak masuk statistik Indonesia.

Peta dasar memakai URL OpenStreetMap `https://tile.openstreetmap.org/{z}/{x}/{y}.png` dengan atribusi; OpenTopoMap tersedia sebagai alternatif. Tile serta JavaScript/CSS Folium memerlukan internet. Buka melalui server HTTP lokal agar Referer dikirim. CARTO tanpa API key tidak dipakai karena dapat mengembalikan gambar “API KEY REQUIRED” meskipun status HTTP sukses. Rujukan: [kebijakan tile OpenStreetMap](https://operations.osmfoundation.org/policies/tiles/) dan [API key CARTO](https://www.carto.com/basemaps/apikey/index.html).

## Sumber dan pemeriksaan

[Feed CSV resmi NOAA-21 tujuh hari](https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_SouthEast_Asia_7d.csv) berasal dari [NASA FIRMS Active Fire Data](https://firms.modaps.eosdis.nasa.gov/content/active_fire/). Feed bergulir dan isinya berubah. Untuk arsip di luar tujuh hari, gunakan layanan [Archive Download](https://firms.modaps.eosdis.nasa.gov/download/); akses API wilayah memerlukan MAP_KEY. Pipeline ini memakai satu satelit agar jumlah deteksi tidak mencampur sensor dengan frekuensi lintasan berbeda.

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Tes berjalan tanpa mengakses layanan NASA atau penyedia tile. Unduhan nyata dan tampilan peta diperiksa terpisah.
