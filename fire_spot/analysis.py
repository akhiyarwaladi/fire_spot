"""Analisis kelompok deteksi VIIRS dan ringkasan per hari/provinsi."""

import hashlib
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from fire_spot.config import (
    BOUNDARIES_PATH,
    FEATURES,
    KALIMANTAN,
    OUTPUT_ROOT,
    RANDOM_STATE,
    SILHOUETTE_SAMPLE_SIZE,
    SUMATRA,
)
from fire_spot.data import load_fire_data
from fire_spot.filtering import assign_provinces, canonical_region, load_boundaries, normalize_name
from fire_spot.maps import cluster_color, create_analysis_maps, write_map_index

logger = logging.getLogger(__name__)


def validate_cluster_count(data: np.ndarray, n_clusters: int) -> None:
    unique_count = len(np.unique(data, axis=0))
    if not 2 <= n_clusters < len(data):
        raise ValueError("Jumlah cluster harus minimal 2 dan lebih kecil dari jumlah baris valid.")
    if n_clusters > unique_count:
        raise ValueError(f"Jumlah cluster melebihi {unique_count} kombinasi fitur unik.")


def sampled_silhouette(data: np.ndarray, labels: np.ndarray) -> float | None:
    if not 2 <= len(np.unique(labels)) < len(data):
        return None
    try:
        return float(
            silhouette_score(
                data,
                labels,
                sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(data)),
                random_state=RANDOM_STATE,
            )
        )
    except ValueError:
        # Sampel dapat melewatkan kelompok yang sangat kecil.
        return None


def find_optimal_k(data: np.ndarray, max_k: int = 10) -> pd.DataFrame:
    if max_k < 2:
        raise ValueError("K maksimum harus minimal 2.")
    upper = min(max_k, len(data) - 1, len(np.unique(data, axis=0)))
    rows = []
    for k in range(2, upper + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(data)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": sampled_silhouette(data, model.labels_),
            }
        )
    return pd.DataFrame(rows, columns=["k", "inertia", "silhouette"])


def select_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    name = normalize_name(region)
    if name in ("SUMATRA", "SUMATERA"):
        selected = df.loc[df.province.isin(SUMATRA)]
    elif name == "INDONESIA":
        selected = df.loc[df.province.notna()]
    elif name == "ALL":
        selected = df
    else:
        selected = df.loc[df.province.eq(canonical_region(region))]
    if selected.empty:
        raise ValueError(
            f"Tidak ada deteksi pada wilayah '{region}'. Periksa data dan nama wilayah."
        )
    return selected.copy()


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby("cluster")
    return groups.agg(
        detections=("frp", "size"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        mean_frp_mw=("frp", "mean"),
        max_frp_mw=("frp", "max"),
        first_date=("acq_date", "min"),
        last_date=("acq_date", "max"),
        active_dates=("acq_date", "nunique"),
        provinces=("province", lambda values: ", ".join(sorted(values.dropna().unique()))),
    ).sort_values("detections", ascending=False)


def save_charts(df: pd.DataFrame, daily: pd.Series, provinces: pd.Series, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), layout="constrained")
    for cluster_id, group in df.groupby("cluster"):
        axes[0].scatter(
            group.longitude, group.latitude, s=8, alpha=0.7, color=cluster_color(int(cluster_id))
        )
    axes[0].set(title="Kelompok deteksi (K-Means)", xlabel="Longitude", ylabel="Latitude")
    axes[0].grid(alpha=0.2)
    daily.plot.bar(ax=axes[1], color="#1565c0", rot=45)
    axes[1].set(
        title="Deteksi per hari (UTC)", xlabel="Tanggal pengamatan", ylabel="Jumlah deteksi"
    )
    if provinces.empty:
        axes[2].text(0.1, 0.5, "Tidak ada provinsi Indonesia teridentifikasi")
    else:
        provinces.head(10).sort_values().rename(
            index=lambda value: value.replace("_", " ")
        ).plot.barh(ax=axes[2], color="#c62828")
    axes[2].set(title="Provinsi dengan deteksi terbanyak", xlabel="Jumlah deteksi", ylabel="")
    fig.savefig(output / "overview.png", dpi=180)
    plt.close(fig)


def write_report(metrics: dict, provinces: pd.Series, clusters: pd.DataFrame, output: Path) -> None:
    lines = [
        f"# Deteksi VIIRS: {metrics['region']}",
        "",
        f"Periode pengamatan: {metrics['period_start']} sampai {metrics['period_end']} (UTC).",
        f"Metode: K-Means (K={metrics['n_clusters']}). Confidence: {metrics['confidence']}.",
        "",
        f"Sebanyak {metrics['n_points']:,} deteksi dikelompokkan menjadi {metrics['n_clusters']} kelompok sesuai pilihan K.",
        f"Dalam feed yang sama, Sumatra memiliki {metrics['sumatra_detections']:,} deteksi dari {metrics['indonesia_detections']:,} deteksi di Indonesia ({metrics['sumatra_share_pct']:.1f}%). Kalimantan memiliki {metrics['kalimantan_detections']:,} deteksi.",
        "",
        "## Provinsi pada wilayah analisis",
        "",
        "| Provinsi | Deteksi |",
        "| --- | ---: |",
    ]
    for province, count in provinces.items():
        lines.append(f"| {province.replace('_', ' ')} | {count:,} |")
    lines += [
        "",
        "## Cluster terbesar",
        "",
        "| Cluster | Deteksi | Provinsi | Tanggal aktif |",
        "| --- | ---: | --- | ---: |",
    ]
    for cluster_id, row in clusters.head(10).iterrows():
        lines.append(
            f"| {cluster_id} | {int(row.detections):,} | {row.provinces.replace('_', ' ')} | {int(row.active_dates)} |"
        )
    lines += [
        "",
        "## Membaca hasil",
        "",
        "Deteksi VIIRS adalah anomali panas pada piksel sekitar 375 m. Deteksi dapat berasal dari kebakaran vegetasi, pembakaran pertanian, gunung api, atau sumber panas lain. Satu kebakaran dapat menghasilkan banyak piksel dan teramati kembali pada lintasan berikutnya.",
        "Cluster merangkum kemiripan lokasi, brightness, dan FRP. Jumlah kelompok mengikuti pilihan K dan tidak menentukan jumlah kejadian kebakaran. Verifikasi karhutla membutuhkan informasi tutupan lahan, citra lain, dan/atau pemeriksaan lapangan.",
        "K-Means memakai latitude, longitude, brightness, dan FRP setelah StandardScaler. Seluruh tanggal dalam feed digabung; pengamatan pada hari berbeda tetap dipertahankan. Jarak dihitung pada fitur yang distandardisasi, bukan kilometer, dan anggota satu kelompok dapat berjauhan. Gunakan elbow dan silhouette untuk membandingkan K; nomor cluster bukan peringkat bahaya.",
        "Angka harian mengikuti cakupan feed NRT dan dapat berubah akibat keterlambatan pengolahan, awan, serta ketersediaan lintasan. Hari di tepi jendela dapat parsial. Data tujuh hari ini belum cukup untuk menyimpulkan kenaikan kebakaran dibanding periode sebelumnya.",
        "",
        "[Deskripsi produk VIIRS NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html)",
        "[Peta cluster](maps/clusters.html) · [Heatmap](maps/heatmap.html) · [Grafik](charts/overview.png)",
    ]
    if metrics["source"]:
        source = metrics["source"]
        lines += [
            "",
            f"[Sumber unduhan NASA]({source['source_url']}). Diunduh {source['downloaded_at_utc']}. SHA-256: `{source['sha256']}`.",
        ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    data_source: str | Path,
    region: str = "sumatra",
    n_clusters: int = 5,
    output_root: str | Path = OUTPUT_ROOT,
    max_k: int = 10,
    basemap: str = "street",
    boundaries_path: str | Path = BOUNDARIES_PATH,
    include_low_confidence: bool = False,
) -> Path:
    data_source = Path(data_source)
    source_path = data_source.with_suffix(".source.json")
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else None
    if source and hashlib.sha256(data_source.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Checksum CSV berbeda dari catatan unduhan. Unduh ulang data NASA.")
    data = load_fire_data(data_source, include_low_confidence=include_low_confidence)
    if "acq_date" not in data:
        raise ValueError("Kolom acq_date diperlukan untuk analisis periode pengamatan.")
    data["acq_date"] = pd.to_datetime(data.acq_date, format="%Y-%m-%d", errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    if data.acq_date.isna().any():
        raise ValueError("Tanggal pengamatan tidak boleh kosong.")
    data = assign_provinces(data, load_boundaries(boundaries_path))
    indonesia = data.loc[data.province.notna()]
    n_sumatra = int(data.province.isin(SUMATRA).sum())
    df = select_region(data, region)

    scaled = StandardScaler().fit_transform(df[FEATURES])
    validate_cluster_count(scaled, n_clusters)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    df["cluster"] = model.fit_predict(scaled)
    if df.cluster.nunique() != n_clusters:
        raise ValueError("K-Means tidak menghasilkan jumlah cluster yang diminta.")
    scores = find_optimal_k(scaled, max_k)

    name = normalize_name(region)
    if name in ("SUMATRA", "SUMATERA"):
        output_name = "sumatra"
    elif name in ("INDONESIA", "ALL"):
        output_name = name.lower()
    else:
        output_name = canonical_region(region).lower()
    output_dir = Path(output_root) / output_name
    for folder in ("charts", "maps"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "k_scores.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    scores.plot(x="k", y="inertia", marker="o", ax=axes[0])
    scores.plot(x="k", y="silhouette", marker="o", ax=axes[1])
    figure.savefig(output_dir / "charts/elbow.png", dpi=180)
    plt.close(figure)

    period_start, period_end = data.acq_date.min(), data.acq_date.max()
    dates = pd.date_range(period_start, period_end).strftime("%Y-%m-%d")
    daily = df.groupby("acq_date").size().reindex(dates, fill_value=0)
    daily.index.name = "acq_date"
    provinces = df.groupby("province").size()
    if normalize_name(region) in ("SUMATRA", "SUMATERA"):
        provinces = provinces.reindex(SUMATRA, fill_value=0)
    provinces = provinces.sort_values(ascending=False)
    provinces.index.name = "province"
    clusters = cluster_summary(df)
    metrics = {
        "dataset": data_source.name,
        "region": region,
        "method": "kmeans",
        "period_start": period_start,
        "period_end": period_end,
        "confidence": "nominal/high/low"
        if include_low_confidence
        else "nominal/high (jika kolom tersedia)",
        "n_points": len(df),
        "n_clusters": n_clusters,
        "features": FEATURES,
        "random_state": RANDOM_STATE,
        "n_init": 10,
        "silhouette": sampled_silhouette(scaled, model.labels_),
        "silhouette_sample_size": min(SILHOUETTE_SAMPLE_SIZE, len(df)),
        "davies_bouldin": float(davies_bouldin_score(scaled, model.labels_)),
        "inertia": float(model.inertia_),
        "indonesia_detections": len(indonesia),
        "sumatra_detections": n_sumatra,
        "kalimantan_detections": int(data.province.isin(KALIMANTAN).sum()),
        "sumatra_share_pct": 100 * n_sumatra / len(indonesia) if len(indonesia) else 0,
        "source": source,
    }
    df.to_csv(output_dir / "detections.csv", index=False)
    clusters.to_csv(output_dir / "clusters.csv")
    daily.to_csv(output_dir / "daily_counts.csv", header=["detections"])
    provinces.to_csv(output_dir / "province_counts.csv", header=["detections"])
    indonesia.groupby("province").size().sort_values(ascending=False).to_csv(
        output_dir / "indonesia_counts.csv", header=["detections"]
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    save_charts(df, daily, provinces, output_dir / "charts")
    note = f"K-Means: K={n_clusters}, memakai lokasi, brightness, dan FRP. Jumlah kelompok mengikuti pilihan K; belum menentukan jumlah kejadian kebakaran."
    maps = create_analysis_maps(df, output_dir / "maps", basemap, note)
    write_map_index(maps, output_dir / "maps", f"Deteksi VIIRS: {region}")
    write_report(metrics, provinces, clusters, output_dir)
    logger.info(
        "%s: %s deteksi, K=%s, silhouette=%s.", region, len(df), n_clusters, metrics["silhouette"]
    )
    logger.info(
        "Periode UTC: %s sampai %s. Hasil: %s", period_start, period_end, output_dir.resolve()
    )
    return output_dir
