import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from fire_spot.analysis import (
    cluster_summary,
    find_optimal_k,
    run_analysis,
    sampled_silhouette,
    validate_cluster_count,
)
from fire_spot.config import FEATURES


def test_small_dataset_caps_elbow_k(fire_data):
    scaled = StandardScaler().fit_transform(fire_data[FEATURES].iloc[:3])
    assert find_optimal_k(scaled).k.tolist() == [2]


@pytest.mark.parametrize("k", [0, 1, 6, 10])
def test_cluster_count_bounds(fire_data, k):
    with pytest.raises(ValueError):
        validate_cluster_count(fire_data[FEATURES].to_numpy(), k)


def test_duplicate_features_cannot_create_extra_clusters():
    data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    with pytest.raises(ValueError, match="unik"):
        validate_cluster_count(data, 3)
    assert find_optimal_k(data).k.tolist() == [2]


def test_silhouette_handles_degenerate_and_rare_clusters():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(2500, 4))
    assert sampled_silhouette(data, np.zeros(2500)) is None
    labels = np.zeros(2500, dtype=int)
    labels[-1] = 1
    first = sampled_silhouette(data, labels)
    assert first is None or np.isfinite(first)
    assert first == sampled_silhouette(data, labels)


def test_kmeans_exports_metadata_metrics_and_two_maps(fire_data, tmp_path):
    source = tmp_path / "demo.csv"
    fire_data.to_csv(source, index=False)
    output = run_analysis(
        source,
        region="all",
        n_clusters=2,
        output_root=tmp_path / "results",
        max_k=3,
    )
    result = pd.read_csv(output / "detections.csv")
    assert result.acq_date.tolist() == fire_data.acq_date.tolist()
    assert result.cluster.nunique() == 2
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["n_points"] == 6
    assert -1 <= metrics["silhouette"] <= 1
    assert len(list((output / "maps").glob("*.html"))) == 3
    assert (output / "charts/elbow.png").is_file()


def test_cluster_table_counts_all_labels(fire_data):
    points = fire_data.assign(cluster=[0, 0, 0, 1, 1, 1], province="Jambi")
    summary = cluster_summary(points)
    assert set(summary.index) == {0, 1}
    assert summary.detections.tolist() == [3, 3]


def test_checksum_mismatch_is_not_misattributed_to_nasa(fire_data, tmp_path):
    source = tmp_path / "demo.csv"
    fire_data.to_csv(source, index=False)
    source.with_suffix(".source.json").write_text(json.dumps({"sha256": "wrong"}))
    with pytest.raises(ValueError, match="Checksum"):
        run_analysis(source, region="all", output_root=tmp_path / "results")


def test_daily_table_keeps_dates_without_detections(fire_data, tmp_path):
    source = tmp_path / "demo.csv"
    fire_data["acq_date"] = ["2025-07-01"] * 2 + ["2025-07-03"] * 4
    fire_data.to_csv(source, index=False)
    output = run_analysis(source, region="all", output_root=tmp_path / "results")
    daily = pd.read_csv(output / "daily_counts.csv")
    assert daily.detections.tolist() == [2, 0, 4]


def test_analysis_handles_points_outside_indonesia(fire_data, tmp_path):
    source = tmp_path / "foreign.csv"
    fire_data.assign(latitude=30, longitude=120).to_csv(source, index=False)
    output = run_analysis(source, region="all", output_root=tmp_path / "results")
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["indonesia_detections"] == 0
    assert pd.read_csv(output / "province_counts.csv").empty
    assert (output / "charts/overview.png").is_file()


def test_invalid_parameters_do_not_create_empty_output_folders(fire_data, tmp_path):
    source = tmp_path / "demo.csv"
    fire_data.to_csv(source, index=False)
    destination = tmp_path / "results"
    with pytest.raises(ValueError):
        run_analysis(source, region="all", n_clusters=1, output_root=destination)
    assert not destination.exists()
