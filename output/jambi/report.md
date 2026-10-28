# Deteksi VIIRS: jambi

Periode pengamatan: 2026-10-21 sampai 2026-10-27 (UTC).
Metode: K-Means (K=4). Confidence: nominal/high (jika kolom tersedia).

Sebanyak 663 deteksi dikelompokkan menjadi 4 kelompok sesuai pilihan K.
Dalam feed yang sama, Sumatra memiliki 5,116 deteksi dari 38,232 deteksi di Indonesia (13.4%). Kalimantan memiliki 22,210 deteksi.

## Provinsi pada wilayah analisis

| Provinsi | Deteksi |
| --- | ---: |
| Jambi | 663 |

## Cluster terbesar

| Cluster | Deteksi | Provinsi | Tanggal aktif |
| --- | ---: | --- | ---: |
| 1 | 369 | Jambi | 5 |
| 3 | 198 | Jambi | 4 |
| 0 | 85 | Jambi | 3 |
| 2 | 11 | Jambi | 2 |

## Membaca hasil

Deteksi VIIRS adalah anomali panas pada piksel sekitar 375 m. Deteksi dapat berasal dari kebakaran vegetasi, pembakaran pertanian, gunung api, atau sumber panas lain. Satu kebakaran dapat menghasilkan banyak piksel dan teramati kembali pada lintasan berikutnya.
Cluster merangkum kemiripan lokasi, brightness, dan FRP. Jumlah kelompok mengikuti pilihan K dan tidak menentukan jumlah kejadian kebakaran. Verifikasi karhutla membutuhkan informasi tutupan lahan, citra lain, dan/atau pemeriksaan lapangan.
K-Means memakai latitude, longitude, brightness, dan FRP setelah StandardScaler. Seluruh tanggal dalam feed digabung; pengamatan pada hari berbeda tetap dipertahankan. Jarak dihitung pada fitur yang distandardisasi, bukan kilometer, dan anggota satu kelompok dapat berjauhan. Gunakan elbow dan silhouette untuk membandingkan K; nomor cluster bukan peringkat bahaya.
Angka harian mengikuti cakupan feed NRT dan dapat berubah akibat keterlambatan pengolahan, awan, serta ketersediaan lintasan. Hari di tepi jendela dapat parsial. Data tujuh hari ini belum cukup untuk menyimpulkan kenaikan kebakaran dibanding periode sebelumnya.

[Deskripsi produk VIIRS NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html)
[Peta cluster](maps/clusters.html) · [Heatmap](maps/heatmap.html) · [Grafik](charts/overview.png)

[Sumber unduhan NASA](https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_SouthEast_Asia_7d.csv). Diunduh 2026-10-28T15:40:39+00:00. SHA-256: `db7b152a2643897047ac3c33bd6f9bdca0c7d58a470d9ed34741457baf4261e8`.
