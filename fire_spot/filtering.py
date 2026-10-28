"""Filter berdasarkan kotak koordinat atau polygon GeoJSON provinsi."""

import json
import logging
import re
from html import escape
from pathlib import Path

import folium
import pandas as pd
import shapely
from shapely.geometry import mapping, shape

from fire_spot.config import BOUNDARIES_PATH, INDONESIA_REGIONS, OUTPUT_ROOT, RAW_DATA
from fire_spot.data import load_fire_data
from fire_spot.maps import (
    InfoPanel,
    add_points,
    cluster_color,
    create_base_map,
    finish_map,
    write_map_index,
)

logger = logging.getLogger(__name__)
ALIASES = {
    "DI_ACEH": "ACEH",
    "DAERAH_ISTIMEWA_ACEH": "ACEH",
    "NANGGROE_ACEH_DARUSSALAM": "ACEH",
    "NUSATENGGARA_BARAT": "NTB",
    "NUSA_TENGGARA_BARAT": "NTB",
    "NUSATENGGARA_TIMUR": "NTT",
    "NUSA_TENGGARA_TIMUR": "NTT",
    "DAERAH_ISTIMEWA_YOGYAKARTA": "DI_YOGYAKARTA",
    "KEPULAUAN_BANGKA_BELITUNG": "BANGKA_BELITUNG",
}


def normalize_name(name: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
    return ALIASES.get(normalized, normalized)


def canonical_region(name: str) -> str:
    normalized = normalize_name(name)
    return next(
        (region for region in INDONESIA_REGIONS if normalize_name(region) == normalized),
        normalized.title(),
    )


def province_name(feature: dict) -> str:
    properties = feature.get("properties", {})
    for key in ("Propinsi", "provinsi", "PROVINSI", "name", "NAME_1"):
        if properties.get(key):
            return str(properties[key])
    raise ValueError("Fitur GeoJSON tidak memiliki properti nama provinsi yang didukung.")


def load_boundaries(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("type") != "FeatureCollection" or not data.get("features"):
        raise ValueError("GeoJSON harus berupa FeatureCollection dengan fitur provinsi.")
    for feature in data["features"]:
        province_name(feature)
        geometry = shape(feature["geometry"])
        if geometry.geom_type not in ("Polygon", "MultiPolygon") or geometry.is_empty:
            raise ValueError(f"Polygon tidak valid: {province_name(feature)}")
        if not geometry.is_valid:
            repaired = shapely.make_valid(geometry)
            if repaired.geom_type == "GeometryCollection":
                polygons = [
                    part for part in repaired.geoms if part.geom_type in ("Polygon", "MultiPolygon")
                ]
                repaired = shapely.union_all(polygons)
            if repaired.is_empty or repaired.geom_type not in ("Polygon", "MultiPolygon"):
                raise ValueError(f"Polygon tidak dapat diperbaiki: {province_name(feature)}")
            feature["geometry"] = mapping(repaired)
            logger.warning("Memperbaiki geometri di memori: %s", province_name(feature))
    return data


def find_province_feature(name: str, geojson_data: dict) -> dict:
    matches = [
        feature
        for feature in geojson_data["features"]
        if normalize_name(province_name(feature)) == normalize_name(name)
    ]
    if len(matches) != 1:
        raise ValueError(f"Provinsi '{name}' tidak ditemukan secara unik dalam GeoJSON.")
    return matches[0]


def filter_by_region(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    region = canonical_region(region_name)
    if region not in INDONESIA_REGIONS:
        raise ValueError(f"Bounding box tidak tersedia: {region_name}")
    bounds = INDONESIA_REGIONS[region]
    return df.loc[df.latitude.between(*bounds["lat"]) & df.longitude.between(*bounds["lon"])].copy()


def filter_by_geojson(df: pd.DataFrame, province: str, geojson_data: dict) -> pd.DataFrame:
    polygon = shape(find_province_feature(province, geojson_data)["geometry"])
    # GeoJSON/Shapely memakai (longitude, latitude); Folium memakai [latitude, longitude].
    points = shapely.points(df.longitude.to_numpy(), df.latitude.to_numpy())
    return df.loc[shapely.covers(polygon, points)].copy()


def assign_provinces(df: pd.DataFrame, boundaries: dict) -> pd.DataFrame:
    """Setiap deteksi mendapat satu provinsi; titik di batas tidak dihitung ganda."""
    result = df.copy()
    result["province"] = pd.Series(pd.NA, index=result.index, dtype="string")
    points = shapely.points(result.longitude.to_numpy(), result.latitude.to_numpy())
    for feature in boundaries["features"]:
        polygon = shape(feature["geometry"])
        shapely.prepare(polygon)
        mask = shapely.covers(polygon, points) & result.province.isna()
        result.loc[mask, "province"] = canonical_region(province_name(feature))
    return result


def add_boundary(map_obj: folium.Map, region: str, count: int, geojson_data: dict | None) -> None:
    label = f"Batas {region.replace('_', ' ')} ({count:,} titik)"
    if geojson_data is not None:
        feature = find_province_feature(region, geojson_data)
        folium.GeoJson(
            feature,
            name=label,
            style_function=lambda _: {"color": "#1565c0", "weight": 2, "fillOpacity": 0},
            tooltip=escape(label),
        ).add_to(map_obj)
    else:
        bounds = INDONESIA_REGIONS[canonical_region(region)]
        layer = folium.FeatureGroup(name=label).add_to(map_obj)
        folium.Rectangle(
            [[bounds["lat"][0], bounds["lon"][0]], [bounds["lat"][1], bounds["lon"][1]]],
            color="#1565c0",
            fill=False,
            weight=2,
            tooltip=escape(label),
        ).add_to(layer)


def create_filter_maps(
    datasets: dict[str, pd.DataFrame],
    output_dir: Path,
    geojson_data: dict | None,
    basemap: str,
) -> list[Path]:
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    all_points = pd.concat(datasets.values(), ignore_index=True)
    overview = create_base_map(all_points, "Titik api per wilayah", basemap)
    paths = []
    for index, (region, df) in enumerate(datasets.items()):
        layer = folium.FeatureGroup(name=f"{region.replace('_', ' ')} ({len(df):,} titik)")
        layer.add_to(overview)
        add_points(df, layer, cluster_color(index), region.replace("_", " "))
        add_boundary(overview, region, len(df), geojson_data)
    method = "polygon GeoJSON" if geojson_data else "bounding box (dapat tumpang tindih)"
    InfoPanel(
        "Titik api per wilayah",
        f"<p>{len(datasets)} wilayah; metode {method}. "
        "Gunakan kontrol lapisan untuk memilih wilayah. "
        "Jumlah titik antarwilayah tidak selalu dapat dijumlahkan.</p>",
    ).add_to(overview)
    paths.append(finish_map(overview, maps_dir / "regions.html"))
    write_map_index(paths, maps_dir, "Peta hasil filter wilayah")
    return paths


def generate_filtered_datasets(
    regions_to_filter: list[str] | None = None,
    method: str = "bounding_box",
    raw_data_path: str | Path = RAW_DATA,
    boundaries_path: str | Path = BOUNDARIES_PATH,
    output_dir: str | Path | None = None,
    make_maps: bool = True,
    basemap: str = "street",
    include_low_confidence: bool = False,
) -> list[dict]:
    if method not in ("bounding_box", "geojson"):
        raise ValueError("Metode harus 'bounding_box' atau 'geojson'.")
    df = load_fire_data(raw_data_path, include_low_confidence=include_low_confidence)
    boundaries = load_boundaries(boundaries_path) if method == "geojson" else None
    if regions_to_filter is None:
        regions_to_filter = (
            [province_name(feature) for feature in boundaries["features"]]
            if boundaries
            else list(INDONESIA_REGIONS)
        )
    regions = list(dict.fromkeys(canonical_region(name) for name in regions_to_filter))
    if not regions:
        raise ValueError("Pilih setidaknya satu wilayah.")
    for region in regions:
        if boundaries:
            find_province_feature(region, boundaries)
        elif region not in INDONESIA_REGIONS:
            raise ValueError(f"Bounding box tidak tersedia: {region}")
    method_dir = "geojson" if boundaries else "bounding_box"
    output_dir = (
        Path(output_dir) if output_dir is not None else OUTPUT_ROOT / "filtered" / method_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results, datasets = [], {}
    for region in regions:
        filtered = (
            filter_by_geojson(df, region, boundaries)
            if boundaries
            else filter_by_region(df, region)
        )
        # Timpa CSV kosong agar hasil lama tidak tampak sebagai hasil terbaru.
        filtered.to_csv(output_dir / f"{region.lower()}.csv", index=False)
        results.append(
            {
                "region": region,
                "status": "success" if not filtered.empty else "no_data",
                "num_points": len(filtered),
                "avg_frp": float(filtered.frp.mean()) if not filtered.empty else None,
            }
        )
        if not filtered.empty:
            datasets[region] = filtered
        logger.info("%s: %s titik", region, f"{len(filtered):,}")
    pd.DataFrame(results).to_csv(output_dir / "summary.csv", index=False)
    if make_maps and datasets:
        create_filter_maps(datasets, output_dir, boundaries, basemap)
    elif make_maps:
        maps_dir = output_dir / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        (maps_dir / "regions.html").unlink(missing_ok=True)
        write_map_index([], maps_dir, "Tidak ada titik api pada wilayah terpilih")
    logger.info("Hasil filter: %s", output_dir.resolve())
    return results
