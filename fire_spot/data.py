"""Pembacaan dan validasi CSV tanpa membuang metadata pengamatan."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fire_spot.config import FEATURES

logger = logging.getLogger(__name__)


def clean_fire_data(df: pd.DataFrame) -> pd.DataFrame:
    """Konversi fitur numerik dan keluarkan baris yang tidak dapat dianalisis."""
    if "brightness" not in df and "bright_ti4" in df:
        df = df.rename(columns={"bright_ti4": "brightness"})
    missing = sorted(set(FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")
    cleaned = df.copy()
    cleaned[FEATURES] = cleaned[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = (
        np.isfinite(cleaned[FEATURES]).all(axis=1)
        & cleaned["latitude"].between(-90, 90)
        & cleaned["longitude"].between(-180, 180)
        & cleaned["frp"].ge(0)
        & cleaned["brightness"].gt(0)
    )
    removed = int((~valid).sum())
    if removed:
        logger.warning("Mengabaikan %s baris dengan fitur kosong/tidak valid.", removed)
    cleaned = cleaned.loc[valid].copy()
    if "confidence" in cleaned:
        cleaned["confidence"] = (
            cleaned.confidence.astype("string")
            .str.lower()
            .replace({"nominal": "n", "high": "h", "low": "l"})
        )
    if cleaned.empty:
        raise ValueError("Tidak ada titik api valid dalam dataset.")
    return cleaned


def load_fire_data(
    path: str | Path,
    include_low_confidence: bool = False,
) -> pd.DataFrame:
    """Validasi CSV, hapus baris identik, dan terapkan pilihan confidence."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV tidak ditemukan: {path}. Jalankan: python -m fire_spot download"
        )
    data = clean_fire_data(pd.read_csv(path, dtype={"acq_time": "string"}))
    data = data.drop_duplicates().copy()
    if "confidence" in data:
        allowed = ["n", "h", "l"] if include_low_confidence else ["n", "h"]
        removed = int((~data.confidence.isin(allowed)).sum())
        if removed:
            logger.info("Mengabaikan %s deteksi dengan confidence di luar %s.", removed, allowed)
        data = data.loc[data.confidence.isin(allowed)].copy()
    if data.empty:
        raise ValueError("Tidak ada deteksi yang memenuhi pilihan confidence.")
    logger.info("Memuat %s deteksi dari %s", f"{len(data):,}", path.name)
    return data.reset_index(drop=True)
