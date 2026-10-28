import hashlib
import json
from datetime import datetime, timedelta, timezone
from io import BytesIO

import pytest

from fire_spot import download


class Response(BytesIO):
    headers = {"Last-Modified": "test snapshot"}


def test_download_saves_original_csv_dates_and_checksum(fire_data, tmp_path, monkeypatch):
    today = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    csv = fire_data.rename(columns={"brightness": "bright_ti4"})
    csv = csv.assign(acq_date=today, acq_time="0630", confidence="nominal")
    content = csv.to_csv(index=False).encode()
    monkeypatch.setattr(download, "urlopen", lambda request, timeout: Response(content))
    output = download.download_latest(tmp_path / "viirs.csv")
    assert output.read_bytes() == content
    metadata = json.loads(output.with_suffix(".source.json").read_text())
    assert metadata["sha256"] == hashlib.sha256(content).hexdigest()
    assert metadata["observed_to_utc"].startswith(today)
    assert metadata["rows"] == len(csv)


@pytest.mark.parametrize(
    "response", [b"<html>service unavailable</html>", b"error,message\n1,failed"]
)
def test_bad_response_does_not_replace_valid_snapshot(tmp_path, monkeypatch, response):
    output = tmp_path / "viirs.csv"
    output.write_bytes(b"previous valid snapshot")
    monkeypatch.setattr(download, "urlopen", lambda request, timeout: Response(response))
    with pytest.raises(ValueError):
        download.download_latest(output)
    assert output.read_bytes() == b"previous valid snapshot"


def test_old_feed_is_not_saved_as_latest(fire_data, tmp_path, monkeypatch):
    content = fire_data.assign(acq_time="0600", confidence="n").to_csv(index=False).encode()
    monkeypatch.setattr(download, "urlopen", lambda request, timeout: Response(content))
    with pytest.raises(ValueError, match="mutakhir"):
        download.download_latest(tmp_path / "viirs.csv")
    assert not (tmp_path / "viirs.csv").exists()


def test_missing_observation_time_does_not_replace_snapshot(fire_data, tmp_path, monkeypatch):
    today = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    csv = fire_data.assign(acq_date=today, acq_time="", confidence="n")
    content = csv.to_csv(index=False).encode()
    monkeypatch.setattr(download, "urlopen", lambda request, timeout: Response(content))
    with pytest.raises(ValueError):
        download.download_latest(tmp_path / "viirs.csv")
    assert not (tmp_path / "viirs.csv").exists()
