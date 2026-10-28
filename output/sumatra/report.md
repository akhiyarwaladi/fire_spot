# Deteksi VIIRS: sumatra

Periode pengamatan: 2026-10-21 sampai 2026-10-27 (UTC).
Metode: K-Means (K=5). Confidence: nominal/high (jika kolom tersedia).

Sebanyak 5,116 deteksi dikelompokkan menjadi 5 kelompok sesuai pilihan K.
Dalam feed yang sama, Sumatra memiliki 5,116 deteksi dari 38,232 deteksi di Indonesia (13.4%). Kalimantan memiliki 22,210 deteksi.

## Provinsi pada wilayah analisis

| Provinsi | Deteksi |
| --- | ---: |
| Sumatera Selatan | 3,504 |
| Jambi | 663 |
| Lampung | 513 |
| Bangka Belitung | 387 |
| Bengkulu | 30 |
| Riau | 16 |
| Sumatera Utara | 1 |
| Aceh | 1 |
| Kepulauan Riau | 1 |
| Sumatera Barat | 0 |

## Cluster terbesar

| Cluster | Deteksi | Provinsi | Tanggal aktif |
| --- | ---: | --- | ---: |
| 2 | 1,778 | Bangka Belitung, Bengkulu, Lampung, Sumatera Selatan | 6 |
| 0 | 1,527 | Bangka Belitung, Bengkulu, Lampung, Sumatera Selatan | 7 |
| 1 | 981 | Aceh, Bangka Belitung, Jambi, Riau, Sumatera Selatan, Sumatera Utara | 7 |
| 4 | 740 | Bangka Belitung, Bengkulu, Jambi, Kepulauan Riau, Sumatera Selatan | 7 |
| 3 | 90 | Bangka Belitung, Jambi, Lampung, Sumatera Selatan | 4 |

## Membaca hasil

Deteksi VIIRS adalah anomali panas pada piksel sekitar 375 m. Deteksi dapat berasal dari kebakaran vegetasi, pembakaran pertanian, gunung api, atau sumber panas lain. Satu kebakaran dapat menghasilkan banyak piksel dan teramati kembali pada lintasan berikutnya.
Cluster merangkum kemiripan lokasi, brightness, dan FRP. Jumlah kelompok mengikuti pilihan K dan tidak menentukan jumlah kejadian kebakaran. Verifikasi karhutla membutuhkan informasi tutupan lahan, citra lain, dan/atau pemeriksaan lapangan.
K-Means memakai latitude, longitude, brightness, dan FRP setelah StandardScaler. Seluruh tanggal dalam feed digabung; pengamatan pada hari berbeda tetap dipertahankan. Jarak dihitung pada fitur yang distandardisasi, bukan kilometer, dan anggota satu kelompok dapat berjauhan. Gunakan elbow dan silhouette untuk membandingkan K; nomor cluster bukan peringkat bahaya.
Angka harian mengikuti cakupan feed NRT dan dapat berubah akibat keterlambatan pengolahan, awan, serta ketersediaan lintasan. Hari di tepi jendela dapat parsial. Data tujuh hari ini belum cukup untuk menyimpulkan kenaikan kebakaran dibanding periode sebelumnya.

[Deskripsi produk VIIRS NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html)
[Peta cluster](maps/clusters.html) · [Heatmap](maps/heatmap.html) · [Grafik](charts/overview.png)

[Sumber unduhan NASA](https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_SouthEast_Asia_7d.csv). Diunduh 2026-10-28T15:40:39+00:00. SHA-256: `db7b152a2643897047ac3c33bd6f9bdca0c7d58a470d9ed34741457baf4261e8`.
