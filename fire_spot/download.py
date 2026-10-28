"""Unduh snapshot VIIRS NOAA-21 tujuh hari dari feed publik NASA FIRMS."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from fire_spot.config import RAW_DATA
from fire_spot.data import clean_fire_data

logger = logging.getLogger(__name__)
SOURCE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
    "noaa-21-viirs-c2/csv/J2_VIIRS_C2_SouthEast_Asia_7d.csv"
)


def download_latest(output: str | Path = RAW_DATA) -> Path:
    """Validasi CSV dan tanggal sebelum mengganti snapshot lokal."""
    request = Request(SOURCE_URL, headers={"User-Agent": "fire-spot/1.0 (VIIRS analysis)"})
    with urlopen(request, timeout=45) as response:
        content = response.read()
        last_modified = response.headers.get("Last-Modified")
    data = pd.read_csv(BytesIO(content), dtype={"acq_time": "string"})
    clean_fire_data(data)
    required = {"acq_date", "acq_time", "satellite", "confidence"}
    if not required.issubset(data.columns):
        raise ValueError("Respons NASA tidak memiliki kolom tanggal/satelit/confidence VIIRS.")
    timestamps = pd.to_datetime(
        data.acq_date + " " + data.acq_time.str.zfill(4), format="%Y-%m-%d %H%M", utc=True
    )
    if timestamps.isna().any():
        raise ValueError("Feed NASA memuat tanggal atau waktu pengamatan kosong.")
    now = datetime.now(timezone.utc)
    age = (now - timestamps.max().to_pydatetime()).total_seconds() / 86400
    if not 0 <= age <= 3:
        raise ValueError(f"Feed NASA belum cukup mutakhir: deteksi terakhir {timestamps.max()}.")
    if not data.satellite.astype(str).eq("N21").all():
        raise ValueError("Feed yang diterima bukan khusus VIIRS NOAA-21.")
    metadata = {
        "product": "VIIRS NOAA-21 Collection 2, 375 m, near real-time",
        "region": "Southeast Asia",
        "source_url": SOURCE_URL,
        "downloaded_at_utc": now.isoformat(timespec="seconds"),
        "last_modified": last_modified,
        "observed_from_utc": timestamps.min().isoformat(),
        "observed_to_utc": timestamps.max().isoformat(),
        "rows": len(data),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    temporary.write_bytes(content)
    temporary.replace(output)
    output.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        "Diunduh %s deteksi: %s sampai %s UTC", len(data), timestamps.min(), timestamps.max()
    )
    logger.info("Sumber dan checksum: %s", output.with_suffix(".source.json"))
    return output
