import numpy as np
import pytest

from fire_spot.config import FEATURES
from fire_spot.data import clean_fire_data, load_fire_data


def test_clean_numeric_strings_without_losing_metadata(fire_data):
    fire_data["frp"] = fire_data.frp.astype(str)
    result = clean_fire_data(fire_data)
    assert result.frp.tolist() == [0, 5, 10, 20, 30, 40]
    assert result.satellite.tolist() == ["N21"] * 6
    assert fire_data.frp.iloc[0] == "0.0"


@pytest.mark.parametrize(
    "column,value",
    [
        ("latitude", 91),
        ("longitude", -181),
        ("frp", -1),
        ("brightness", 0),
        ("frp", np.inf),
        ("latitude", "invalid"),
    ],
)
def test_remove_invalid_rows(fire_data, column, value):
    fire_data[column] = fire_data[column].astype(object)
    fire_data.loc[0, column] = value
    assert len(clean_fire_data(fire_data)) == 5


def test_missing_columns_and_empty_data(fire_data):
    with pytest.raises(ValueError, match="frp"):
        clean_fire_data(fire_data.drop(columns="frp"))
    with pytest.raises(ValueError, match="Tidak ada"):
        clean_fire_data(fire_data.iloc[:0])


def test_loader_keeps_all_valid_points_and_date_is_optional(fire_data, tmp_path):
    path = tmp_path / "data.csv"
    fire_data[FEATURES].to_csv(path, index=False)
    first = load_fire_data(path)
    assert len(first) == len(fire_data)
    assert "acq_date" not in first
    with pytest.raises(FileNotFoundError):
        load_fire_data(tmp_path / "missing.csv")


def test_recent_viirs_schema_and_confidence_filter(fire_data, tmp_path):
    path = tmp_path / "viirs.csv"
    data = fire_data.rename(columns={"brightness": "bright_ti4"})
    data["confidence"] = ["nominal", "high", "low", "n", "h", "l"]
    data.to_csv(path, index=False)
    selected = load_fire_data(path)
    assert len(selected) == 4
    assert "brightness" in selected
    assert set(selected.confidence) == {"n", "h"}
    assert len(load_fire_data(path, include_low_confidence=True)) == 6
