import numpy as np
import pytest

from fire_spot.maps import (
    OSM_TILE_URL,
    add_cluster_points,
    create_base_map,
    finish_map,
    heat_data,
    point_popup,
)


def test_map_has_canonical_xyz_attribution_fit_bounds_and_one_basemap(fire_data, tmp_path):
    map_obj = create_base_map(fire_data, "Test")
    tiles = [
        child for child in map_obj._children.values() if child.__class__.__name__ == "TileLayer"
    ]
    assert sum(tile.show for tile in tiles) == 1
    assert next(tile.tiles for tile in tiles if tile.show) == OSM_TILE_URL
    path = finish_map(map_obj, tmp_path / "map.html")
    html = path.read_text(encoding="utf-8")
    assert OSM_TILE_URL in html
    assert "openstreetmap.org/copyright" in html
    assert ".fitBounds(" in html
    assert "layer_control" in html
    assert "tileerror" in html


def test_heatmap_normalizes_frp_and_keeps_zero_weights(fire_data):
    data = np.array(heat_data(fire_data))
    assert np.array_equal(data[:, :2], fire_data[["latitude", "longitude"]].to_numpy())
    assert data[:, 2].min() == 0
    assert data[:, 2].max() == 1
    assert np.array(heat_data(fire_data.assign(frp=0)))[:, 2].sum() == 0


def test_nonconsecutive_cluster_labels_and_more_than_ten_colors(fire_data):
    df = fire_data.assign(cluster=[0, 2, 11, 2, 11, 0])
    map_obj = create_base_map(df, "Test")
    legend = add_cluster_points(map_obj, df)
    assert "Cluster 11" in legend
    html = map_obj.get_root().render()
    assert html.count("L.circleMarker(") == len(df)


def test_popup_escapes_csv_content(fire_data):
    fire_data.loc[0, "acq_date"] = "<script>alert(1)</script>"
    popup = point_popup(next(fire_data.itertuples(index=False)), "<unsafe>")
    html = popup.html.render()
    assert "&lt;script&gt;" in html
    assert "&lt;unsafe&gt;" in html
    assert "waktu tidak tersedia" in html


def test_empty_map_is_a_clear_error(fire_data):
    with pytest.raises(ValueError, match="setidaknya"):
        create_base_map(fire_data.iloc[:0], "Test")
