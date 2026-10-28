# Titik api VIIRS Indonesia

Proyek ini mengelompokkan deteksi titik api VIIRS NOAA-21 dari NASA FIRMS dengan K-Means dan menampilkannya pada peta Folium. Analisis bawaan memakai wilayah Sumatra dengan K=5.

## Mulai

Gunakan Python 3.10 atau lebih baru. Jalankan dari folder repositori:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m fire_spot download
python -m fire_spot analyze
```

Di Linux/macOS, aktifkan lingkungan Python dengan `source .venv/bin/activate`. Jika PowerShell membatasi aktivasi, gunakan `.venv\Scripts\python.exe` sebagai pengganti `python`.

Salinan data sudah tersedia di `data/raw/`, sehingga `analyze` dapat dijalankan tanpa mengunduh ulang. Perintah `download` mengambil CSV publik NASA untuk tujuh hari pengamatan tanpa memerlukan akun atau kunci API.

Berkas pendamping `.source.json` mencatat URL sumber, waktu unduh, rentang pengamatan, dan *checksum* SHA-256 untuk memeriksa keutuhan CSV. Program menolak unduhan jika deteksi terakhirnya berusia lebih dari tiga hari. CSV lokal dipertahankan jika unduhan atau validasi gagal.

Untuk melihat peta, jalankan server HTTP lokal:

```powershell
python -m http.server 8765 --bind 127.0.0.1
```

Buka [peta Sumatra](http://127.0.0.1:8765/output/sumatra/maps/index.html) di peramban. Biarkan terminal server terbuka selama melihat peta; hentikan dengan `Ctrl+C` setelah selesai.

## Data yang disertakan

Salinan data di repositori ini diunduh pada 28 Oktober 2026. CSV NOAA-21 berisi 58.158 deteksi Asia Tenggara dengan waktu pengamatan 21 Oktober 2026 pukul 03.06 UTC sampai 27 Oktober 2026 pukul 20.34 UTC.

Setelah validasi, penyaringan tingkat keyakinan deteksi (`confidence`) nominal/tinggi, dan pemilihan poligon batas provinsi bawaan, jumlah deteksinya sebagai berikut:

| Wilayah | Deteksi |
| --- | ---: |
| Indonesia | 38.232 |
| Sumatra, termasuk provinsi kepulauannya | 5.116 |
| Kalimantan | 22.210 |
| Sumatera Selatan | 3.504 |
| Jambi | 663 |
| Lampung | 513 |

Angka tersebut berlaku untuk salinan data bertanggal di atas. Jalankan `download` untuk mengambil data terbaru, lalu `analyze` untuk menghitung ulang hasilnya.

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

`--data` memilih CSV masukan, `--boundaries` memilih GeoJSON batas wilayah, dan `--output-dir` menentukan folder induk hasil. Gunakan `--basemap topo` untuk memilih OpenTopoMap atau `--include-low-confidence` untuk menyertakan deteksi dengan tingkat keyakinan rendah.

Lokasi berkas bawaan ditentukan dari folder repositori. Jika argumen perintah memakai lokasi relatif, program mengacu pada folder kerja terminal. Jalankan `python -m fire_spot` dari folder repositori.

Analisis dimulai dengan membaca dan memvalidasi data, lalu memilih wilayah. Empat fitur, yaitu `latitude`, `longitude`, `brightness`, dan `frp`, distandardisasi dengan `StandardScaler` sebelum diolah oleh `KMeans`. Pengaturan `random_state=42` dan `n_init=10` memungkinkan percobaan diulang dengan data dan versi pustaka yang sama.

Jumlah kelompok mengikuti `--clusters`: K=5 membagi data menjadi lima kelompok. Kurva *elbow* dan skor *silhouette* membantu membandingkan pilihan K. Keduanya dihitung untuk K dari dua sampai `--max-k` (bawaan 10), dengan batas sesuai jumlah baris dan kombinasi fitur unik. Program memakai K yang dipilih pengguna untuk hasil akhir.

Skor *silhouette* memakai sampel tetap maksimal 2.000 titik, sedangkan indeks Davies-Bouldin memakai seluruh data analisis. Jika skor *silhouette* tidak dapat dihitung dari sampel, nilainya dicatat kosong.

K-Means menghitung jarak pada fitur yang telah distandardisasi. Jarak ini tidak menyatakan kilometer. Anggota satu kelompok dapat berjauhan karena pengelompokan juga mempertimbangkan `brightness` dan `frp`. Seluruh tanggal dalam data unduhan dianalisis bersama; pengamatan pada hari berbeda tetap dipertahankan.

## Filter CSV per provinsi

```powershell
# Bawaan: provinsi Sumatra, menggunakan poligon
python -m fire_spot filter
python -m fire_spot filter --regions Jambi Riau
python -m fire_spot filter --regions "Kepulauan Riau"
python -m fire_spot filter --all --no-maps

# Kotak pembatas, untuk membandingkan metode
python -m fire_spot filter --sumatra --method bounding_box
```

Hasil tersimpan di `output/filtered/geojson/` atau `output/filtered/bounding_box/`. Setiap provinsi mendapat CSV, termasuk provinsi tanpa deteksi. Berkas `summary.csv` mencatat jumlah deteksi, sedangkan peta `regions.html` menampilkan semua wilayah terpilih. Gunakan `--no-maps` untuk memperbarui CSV dan ringkasan saja.

Metode `geojson` mengikuti poligon batas provinsi. Metode `bounding_box` memakai kotak pembatas berupa rentang koordinat perkiraan. Kotak ini dapat saling tumpang tindih dan memasukkan deteksi dari provinsi tetangga.

Berkas batas wilayah `indonesia_provinces_34.geojson` berisi 34 provinsi dan belum mengikuti pemekaran terbaru di Papua. Saat memuat berkas, program memperbaiki geometri yang tidak valid di memori dengan Shapely `make_valid` dan mencatatnya dalam pesan log. Berkas sumber tetap utuh. Dalam statistik lintas Indonesia, setiap deteksi diberi maksimal satu provinsi agar tidak dihitung ganda pada batas yang bersinggungan.

## Struktur dan hasil

```text
fire_spot/
  __main__.py              # Perintah python -m fire_spot
  cli.py                   # Perintah download, filter, analyze
  config.py                # Lokasi berkas bawaan dan daftar wilayah
  download.py              # Unduhan NASA dan catatan sumber
  data.py                  # Pembacaan dan validasi CSV
  filtering.py             # Batas provinsi dan pemilihan wilayah
  analysis.py              # K-Means, evaluasi, ringkasan, dan grafik
  maps.py                  # Peta Folium dan kontrol tampilan
data/
  boundaries/
    indonesia_provinces_34.geojson
  raw/
    viirs_noaa21_southeast_asia_7d.csv
    viirs_noaa21_southeast_asia_7d.source.json
tests/
output/                    # Hasil yang dibuat program; diabaikan Git
  sumatra/
    detections.csv         # Data valid, provinsi, dan label kelompok
    clusters.csv           # Jumlah deteksi, pusat, FRP, dan tanggal per kelompok
    k_scores.csv           # Inertia dan silhouette untuk kandidat K
    daily_counts.csv       # Jumlah deteksi per tanggal dalam UTC
    province_counts.csv    # Jumlah deteksi per provinsi pada wilayah analisis
    indonesia_counts.csv   # Jumlah deteksi per provinsi Indonesia
    metrics.json           # Pengaturan, skor evaluasi, dan catatan sumber
    report.md              # Temuan dan batas interpretasi
    charts/overview.png
    charts/elbow.png
    maps/
      index.html
      clusters.html
      heatmap.html
```

Nama berkas hasil memakai huruf kecil dengan pemisah garis bawah. Program menyimpan hasil filter, peta, dan grafik di `output/`. Analisis untuk wilayah dan folder hasil yang sama akan menimpa hasil sebelumnya. Gunakan `--output-dir` berbeda untuk membandingkan nilai K.

`clusters.csv` merangkum setiap kelompok; `detections.csv` menyimpan seluruh deteksi beserta labelnya.

## Membaca hasil

VIIRS mendeteksi anomali panas pada piksel sekitar 375 m. Sumbernya dapat berupa kebakaran vegetasi, pembakaran pertanian, gunung api, atau sumber panas lain. Satu kebakaran dapat menghasilkan banyak deteksi dan diamati kembali oleh satelit. Mengaitkan kelompok deteksi dengan kejadian kebakaran hutan dan lahan (karhutla) memerlukan informasi tutupan lahan, citra lain, atau pemeriksaan lapangan. Rujukan: [deskripsi produk VIIRS NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html).

Salinan data Oktober 2026 yang disertakan menunjukkan pemusatan deteksi Sumatra di Sumatera Selatan. Untuk menilai kenaikan karhutla, bandingkan periode dengan sensor, durasi pengamatan, dan penyaringan yang sama. Awan, lintasan satelit, serta keterlambatan pengolahan memengaruhi jumlah deteksi. Perhatikan cakupan waktu pada tanggal awal dan akhir rentang unduhan saat membandingkan jumlah harian.

Kolom `bright_ti4` dibaca sebagai `brightness` tanpa mengubah CSV asli. Program mengenali nilai `confidence` berupa `nominal`/`high`/`low` maupun `n`/`h`/`l`. Baris dengan fitur kosong, nilai tak hingga, koordinat tidak valid, `brightness` tidak positif, atau `frp` negatif diabaikan, dan jumlahnya dicatat dalam log. Duplikat baris identik dihapus; pengamatan pada waktu berbeda tetap dipertahankan. Analisis memerlukan `acq_date` yang valid.

Pada peta pengelompokan, lingkaran bernomor menggabungkan penanda lokasi saat tampilan diperkecil. Klik lingkaran untuk memperbesar peta; warna tiap titik mengikuti label hasil K-Means. Angka pada lingkaran menunjukkan jumlah penanda yang digabung. Jumlah kelompok hasil analisis tercatat di `metrics.json`. Nomor kelompok merupakan label; urutannya tidak menunjukkan tingkat bahaya.

Peta intensitas (*heatmap*) memakai bobot `FRP / FRP maksimum dataset`. Warnanya relatif terhadap data yang dianalisis dan dipengaruhi kepadatan titik serta tingkat pembesaran peta. Grafik harian memakai UTC. Statistik Indonesia hanya mencakup deteksi di dalam poligon batas wilayah bawaan.

Peta dasar memakai URL OpenStreetMap `https://tile.openstreetmap.org/{z}/{x}/{y}.png` dengan atribusi. OpenTopoMap tersedia sebagai pilihan lain. Gambar peta dasar (*tile*) serta JavaScript/CSS Folium memerlukan internet. Buka peta melalui server HTTP lokal agar peramban mengirim informasi asal halaman (`Referer`). CARTO memerlukan kunci API; permintaan tanpa kunci dapat menghasilkan gambar “API KEY REQUIRED” meskipun status HTTP berhasil. Rujukan: [kebijakan tile OpenStreetMap](https://operations.osmfoundation.org/policies/tiles/) dan [kunci API CARTO](https://www.carto.com/basemaps/apikey/index.html).

## Sumber dan pemeriksaan

[CSV resmi NOAA-21 untuk tujuh hari](https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_SouthEast_Asia_7d.csv) berasal dari [NASA FIRMS Active Fire Data](https://firms.modaps.eosdis.nasa.gov/content/active_fire/). Rentang data bergeser seiring pembaruan NASA. Proyek ini memakai satu satelit agar jumlah deteksi tidak mencampur sensor dengan frekuensi lintasan berbeda.

Untuk data yang lebih lama, gunakan layanan [Archive Download](https://firms.modaps.eosdis.nasa.gov/download/). Layanan API wilayah memerlukan `MAP_KEY`.

Untuk memeriksa kode:

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Pengujian otomatis berjalan tanpa mengakses NASA atau penyedia peta dasar. Periksa unduhan dan tampilan peta secara terpisah dengan koneksi internet.
