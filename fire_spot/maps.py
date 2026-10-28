"""Peta Folium: tile XYZ, kontrol lapisan, legenda, dan batas tampilan."""

from colorsys import hls_to_rgb
from html import escape
from pathlib import Path

import folium
import pandas as pd
from branca.element import MacroElement, Template
from folium.plugins import Fullscreen, HeatMap, MarkerCluster

OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)
BASE_TILES = {
    "street": {
        "tiles": OSM_TILE_URL,
        "attr": OSM_ATTRIBUTION,
        "name": "OpenStreetMap",
        "max_native_zoom": 19,
    },
    "topo": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": OSM_ATTRIBUTION
        + ', SRTM | &copy; <a href="https://opentopomap.org">OpenTopoMap</a> '
        + '(<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
        "name": "OpenTopoMap",
        "max_native_zoom": 17,
    },
}
COLORS = [
    "#c62828",
    "#1565c0",
    "#2e7d32",
    "#ad5b00",
    "#7b1fa2",
    "#00796b",
    "#c2185b",
    "#455a64",
    "#6d4c41",
    "#536d00",
]
HEAT_GRADIENT = {0.0: "#253494", 0.35: "#2c7fb8", 0.55: "#ffffb2", 0.75: "#fd8d3c", 1.0: "#bd0026"}


def cluster_color(cluster_id: int) -> str:
    """Warna yang sama dipakai oleh scatter plot, marker, dan legenda."""
    if cluster_id < len(COLORS):
        return COLORS[cluster_id]
    red, green, blue = hls_to_rgb((cluster_id * 0.61803398875) % 1, 0.36, 0.70)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


class InfoPanel(MacroElement):
    """Kontrol Leaflet mengikuti tata letak peta, termasuk saat layar penuh."""

    _template = Template("""
        {% macro script(this, kwargs) %}
        var {{ this.get_name() }} = L.control({position: 'bottomleft'});
        {{ this.get_name() }}.onAdd = function() {
            var panel = L.DomUtil.create('div', 'fire-info leaflet-control');
            panel.innerHTML = {{ this.content | tojson }};
            if (window.matchMedia('(max-width: 600px)').matches) {
                panel.querySelector('details').removeAttribute('open');
            }
            L.DomEvent.disableClickPropagation(panel);
            L.DomEvent.disableScrollPropagation(panel);
            return panel;
        };
        {{ this.get_name() }}.addTo({{ this._parent.get_name() }});
        {% endmacro %}
    """)

    def __init__(self, title: str, body: str):
        super().__init__()
        self.content = (
            f"<details open><summary>{escape(title)}</summary>"
            f'<div class="fire-info-body">{body}</div></details>'
        )


class TileStatus(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
        var {{ this.get_name() }} = L.control({position: 'bottomright'});
        {{ this.get_name() }}.onAdd = function() {
            this.box = L.DomUtil.create('div', 'fire-tile-status');
            this.box.setAttribute('role', 'status');
            this.box.textContent = 'Tile gagal dimuat. Periksa internet atau pilih peta dasar lain.';
            this.box.hidden = true;
            L.DomEvent.disableClickPropagation(this.box);
            return this.box;
        };
        {{ this.get_name() }}.addTo({{ this._parent.get_name() }});
        function {{ this.get_name() }}_watch(layer) {
            if (layer instanceof L.TileLayer && !layer.fireSpotStatusWatched) {
                layer.fireSpotStatusWatched = true;
                layer.on('tileerror', function() { {{ this.get_name() }}.box.hidden = false; });
                layer.on('tileload', function() { {{ this.get_name() }}.box.hidden = true; });
            }
        }
        {{ this._parent.get_name() }}.eachLayer({{ this.get_name() }}_watch);
        {{ this._parent.get_name() }}.on('layeradd', function(event) {
            {{ this.get_name() }}_watch(event.layer);
        });
        {% endmacro %}
    """)


def create_base_map(df: pd.DataFrame, title: str, basemap: str = "street") -> folium.Map:
    if df.empty:
        raise ValueError("Peta membutuhkan setidaknya satu titik api.")
    if basemap not in BASE_TILES:
        raise ValueError(f"Peta dasar tidak dikenal: {basemap}")
    result = folium.Map(
        location=[float(df.latitude.mean()), float(df.longitude.mean())],
        tiles=None,
        prefer_canvas=True,
        control_scale=True,
        max_zoom=20,
        zoom_animation=False,
        fade_animation=False,
    )
    for key in [basemap, *(key for key in BASE_TILES if key != basemap)]:
        folium.TileLayer(**BASE_TILES[key], overlay=False, show=key == basemap, max_zoom=20).add_to(
            result
        )
    result.fit_bounds(
        [
            [float(df.latitude.min()), float(df.longitude.min())],
            [float(df.latitude.max()), float(df.longitude.max())],
        ],
        padding=(35, 35),
        max_zoom=13,
    )
    result.get_root().header.add_child(folium.Element(f"<title>{escape(title)}</title>"))
    result.get_root().header.add_child(
        folium.Element("""
        <style>
        .fire-info { background: white; color: #202124; border: 1px solid #777;
            border-radius: 4px; width: min(300px, calc(100vw - 86px));
            font: 13px/1.5 Arial, sans-serif; box-shadow: 0 1px 5px #0003; }
        .fire-info summary { padding: 10px 12px; cursor: pointer; font-weight: 700; }
        .fire-info-body { padding: 0 12px 10px; max-height: 32vh; overflow: auto; }
        .fire-info p { margin: 6px 0; }
        .fire-info ul { padding-left: 18px; margin: 6px 0; }
        .fire-info li { margin: 6px 0; }
        .fire-key { display: inline-block; width: 12px; height: 12px;
            border-radius: 50%; margin-right: 6px; border: 1px solid #555; }
        .fire-gradient { height: 12px; background: linear-gradient(to right,
            #253494, #2c7fb8, #ffffb2, #fd8d3c, #bd0026); }
        .fire-tile-status { max-width: 240px; padding: 8px; color: #611;
            background: #fff3e0; border: 1px solid #b55; }
        .leaflet-control-layers { max-width: calc(100vw - 80px);
            max-height: 45vh; overflow: auto; }
        .leaflet-control-attribution { max-width: calc(100vw - 20px); }
        @media (max-width: 600px) {
            .fire-info-body { max-height: 25vh; }
        }
        </style>
    """)
    )
    Fullscreen(position="topleft").add_to(result)
    return result


def finish_map(map_obj: folium.Map, path: str | Path) -> Path:
    folium.LayerControl(collapsed=True).add_to(map_obj)
    TileStatus().add_to(map_obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(path))
    return path


def point_popup(row, label: str) -> folium.Popup:
    date = escape(str(getattr(row, "acq_date", "Tidak tersedia")))
    raw_time = str(getattr(row, "acq_time", ""))
    if raw_time.isdigit() and len(raw_time) <= 4:
        raw_time = raw_time.zfill(4)
        time = f"{raw_time[:2]}:{raw_time[2:]}"
    else:
        time = "waktu tidak tersedia"
    confidence = str(getattr(row, "confidence", "Tidak tersedia"))
    confidence = {"n": "nominal", "h": "tinggi", "l": "rendah"}.get(confidence, confidence)
    return folium.Popup(
        f"<b>{escape(label)}</b><br>"
        f"Koordinat: {row.latitude:.5f}, {row.longitude:.5f}<br>"
        f"FRP: {row.frp:.2f} MW<br>Brightness: {row.brightness:.2f} K<br>"
        f"Tanggal: {date} {time} UTC<br>Confidence: {escape(confidence)}",
        max_width=280,
    )


def add_points(df: pd.DataFrame, target, color: str, label: str) -> None:
    for row in df.itertuples(index=False):
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=6 if row.frp > 20 else 4,
            color="#ffffff",
            weight=0.8,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"{escape(label)} | FRP {row.frp:.2f} MW",
            popup=point_popup(row, label),
        ).add_to(target)


def add_cluster_points(map_obj: folium.Map, df: pd.DataFrame) -> str:
    layer = MarkerCluster(
        name=f"Deteksi menurut cluster ({len(df):,})",
        show_coverage_on_hover=False,
        disable_clustering_at_zoom=12,
        icon_create_function="""function(cluster) {
            return L.divIcon({
                html: '<div style="background:#263238;color:white;border:2px solid white;'
                    + 'border-radius:50%;width:36px;height:36px;line-height:32px;'
                    + 'text-align:center;font-weight:bold">' + cluster.getChildCount() + '</div>',
                className: '', iconSize: L.point(36, 36)
            });
        }""",
    ).add_to(map_obj)
    items = []
    for cluster_id, group in df.groupby("cluster", sort=True):
        color = cluster_color(int(cluster_id))
        label = f"Cluster {cluster_id}"
        add_points(group, layer, color, label)
        item = (
            f'<li><span class="fire-key" style="background:{color}"></span>'
            f"{label}: {len(group):,} deteksi<br>FRP rata-rata {group.frp.mean():.2f} MW</li>"
        )
        items.append((len(group), item))
    items.sort(key=lambda item: item[0], reverse=True)
    legend = "<ul>" + "".join(html for _, html in items[:10]) + "</ul>"
    if len(items) > 10:
        legend += "<p>Legenda menampilkan 10 kategori terbanyak. Seluruh kelompok tetap ditampilkan pada peta.</p>"
    return legend


def heat_data(df: pd.DataFrame) -> list[list[float]]:
    """Leaflet.heat menerima bobot 0..1; FRP mentah akan menjenuhkan warna."""
    weights = df.frp / df.frp.max() if df.frp.max() > 0 else df.frp
    return [
        [float(lat), float(lon), float(weight)]
        for lat, lon, weight in zip(df.latitude, df.longitude, weights, strict=True)
    ]


def add_heatmap(map_obj: folium.Map, df: pd.DataFrame, show: bool = True) -> None:
    HeatMap(
        heat_data(df),
        name="Intensitas FRP (relatif)",
        show=show,
        radius=20,
        blur=16,
        min_opacity=0.25,
        gradient=HEAT_GRADIENT,
    ).add_to(map_obj)


def summary_html(df: pd.DataFrame) -> str:
    period = ""
    if "acq_date" in df:
        period = (
            f"<p>{escape(str(df.acq_date.min()))} sampai {escape(str(df.acq_date.max()))} (UTC)</p>"
        )
    return period + (
        f"<p>{len(df):,} deteksi | FRP rata-rata {df.frp.mean():.2f} MW<br>"
        f"FRP maksimum {df.frp.max():.2f} MW</p>"
    )


def heat_legend() -> str:
    return (
        '<div class="fire-gradient" aria-label="Intensitas rendah ke tinggi"></div>'
        "<p>Rendah &rarr; tinggi (relatif terhadap FRP maksimum dataset). "
        "Warna juga dipengaruhi kepadatan titik dan zoom."
        "<br>Semua FRP nol menghasilkan heatmap tanpa intensitas.</p>"
    )


def create_analysis_maps(
    df: pd.DataFrame, output_dir: Path, basemap: str = "street", note: str = ""
) -> list[Path]:
    maps = []
    for filename, title in [
        ("clusters.html", "Kelompok deteksi VIIRS"),
        ("heatmap.html", "Intensitas dan lokasi deteksi"),
    ]:
        map_obj = create_base_map(df, title, basemap)
        body = summary_html(df)
        if note:
            body += f"<p>{escape(note)}</p>"
        if filename == "heatmap.html":
            add_heatmap(map_obj, df)
            points = folium.FeatureGroup(name="Deteksi (klik untuk detail)", show=False)
            points.add_to(map_obj)
            add_points(df, points, COLORS[0], "Deteksi VIIRS")
            body += heat_legend()
        else:
            body += add_cluster_points(map_obj, df)
            body += "<p>Nomor cluster bukan peringkat risiko. Lingkaran bernomor menggabungkan marker saat zoom jauh; klik untuk memperbesar.</p>"
        InfoPanel(title, body).add_to(map_obj)
        maps.append(finish_map(map_obj, output_dir / filename))
    return maps


def write_map_index(paths: list[Path], output_dir: Path, title: str) -> Path:
    links = "".join(
        f'<li><a href="{escape(path.name, quote=True)}">{escape(path.stem.replace("_", " "))}</a></li>'
        for path in paths
    )
    path = output_dir / "index.html"
    path.write_text(
        '<!doctype html><html lang="id"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title>"
        "<style>body{font:16px/1.6 system-ui,sans-serif;max-width:720px;margin:32px auto;"
        "padding:0 20px;color:#202124}li{margin:12px 0}a{color:#1355a0}</style></head>"
        f"<body><h1>{escape(title)}</h1><p>Pilih peta untuk melihat hasil. "
        "Gunakan kontrol lapisan di kanan atas untuk mengganti peta dasar.</p>"
        f"<ul>{links}</ul><p>Tile dan pustaka JavaScript memerlukan koneksi internet.</p></body></html>",
        encoding="utf-8",
    )
    return path
