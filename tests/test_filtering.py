import json

import pandas as pd
import pytest

from fire_spot.config import BOUNDARIES_PATH, INDONESIA_REGIONS
from fire_spot.filtering import (
    assign_provinces,
    filter_by_geojson,
    filter_by_region,
    find_province_feature,
    generate_filtered_datasets,
    load_boundaries,
    normalize_name,
)


def test_riau_never_matches_kepulauan_riau(boundaries):
    assert find_province_feature("Riau", boundaries)["properties"]["Propinsi"] == "RIAU"
    assert (
        find_province_feature("Kepulauan Riau", boundaries)["properties"]["Propinsi"]
        == "KEPULAUAN RIAU"
    )
    with pytest.raises(ValueError):
        find_province_feature("Sulawesi", boundaries)


def test_polygon_includes_border_and_uses_longitude_first(boundaries):
    points = pd.DataFrame({"latitude": [-1.0, -1.0, -1.0], "longitude": [102.5, 102.0, 104.5]})
    assert filter_by_geojson(points, "Riau", boundaries).index.tolist() == [0, 1]
    assert filter_by_geojson(points, "Kepulauan_Riau", boundaries).index.tolist() == [2]


def test_province_assignment_never_counts_one_detection_twice(boundaries):
    points = pd.DataFrame({"latitude": [-1, -1, -1], "longitude": [102.5, 104.5, 110]})
    result = assign_provinces(points, boundaries)
    assert result.province.tolist()[:2] == ["Riau", "Kepulauan_Riau"]
    assert result.province.isna().sum() == 1
    assert len(result) == len(points)


def test_bounding_box_accepts_spaces_and_rejects_unknown(fire_data):
    assert len(filter_by_region(fire_data, "jambi")) == 6
    with pytest.raises(ValueError):
        filter_by_region(fire_data, "unknown")


def test_bundled_geojson_names_match_all_configured_regions():
    data = load_boundaries(BOUNDARIES_PATH)
    assert len(data["features"]) == 34
    for region in INDONESIA_REGIONS:
        find_province_feature(region, data)
    assert normalize_name("DI. ACEH") == "ACEH"
    assert normalize_name("NUSATENGGARA BARAT") == "NTB"


def test_invalid_polygon_is_repaired_without_rewriting_source(boundaries, tmp_path):
    feature = boundaries["features"][0]
    feature["geometry"]["coordinates"] = [[[0, 0], [2, 2], [2, 0], [0, 2], [0, 0]]]
    source = tmp_path / "boundary.json"
    original = json.dumps(boundaries)
    source.write_text(original, encoding="utf-8")
    result = load_boundaries(source)
    assert result["features"][0]["geometry"]["type"] == "MultiPolygon"
    assert source.read_text(encoding="utf-8") == original


def test_filter_reports_empty_regions_and_overwrites_stale_csv(fire_data, tmp_path):
    source = tmp_path / "source.csv"
    destination = tmp_path / "filtered"
    fire_data.to_csv(source, index=False)
    results = generate_filtered_datasets(
        ["Jambi", "Aceh"], raw_data_path=source, output_dir=destination, make_maps=False
    )
    assert [item["num_points"] for item in results] == [6, 0]
    assert pd.read_csv(destination / "aceh.csv").empty
    fire_data.assign(latitude=50).to_csv(source, index=False)
    generate_filtered_datasets(
        ["Jambi"], raw_data_path=source, output_dir=destination, make_maps=False
    )
    assert pd.read_csv(destination / "jambi.csv").empty
    assert (destination / "summary.csv").is_file()


def test_empty_filter_refreshes_map_index(fire_data, tmp_path):
    source = tmp_path / "source.csv"
    destination = tmp_path / "filtered"
    maps_dir = destination / "maps"
    maps_dir.mkdir(parents=True)
    (maps_dir / "regions.html").write_text("stale")
    fire_data.assign(latitude=50).to_csv(source, index=False)
    generate_filtered_datasets(["Jambi"], raw_data_path=source, output_dir=destination)
    assert not (maps_dir / "regions.html").exists()
    assert "Tidak ada titik api" in (maps_dir / "index.html").read_text(encoding="utf-8")


def test_unknown_method_fails_before_writing(fire_data, tmp_path):
    with pytest.raises(ValueError, match="Metode"):
        generate_filtered_datasets(method="typo", output_dir=tmp_path / "output")
    assert not (tmp_path / "output").exists()
