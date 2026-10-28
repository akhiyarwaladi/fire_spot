import pandas as pd
import pytest


@pytest.fixture
def fire_data():
    return pd.DataFrame(
        {
            "latitude": [-2.1, -2.2, -2.3, -1.2, -1.3, -1.4],
            "longitude": [102.1, 102.2, 102.3, 103.1, 103.2, 103.3],
            "brightness": [320.0, 325.0, 330.0, 350.0, 360.0, 370.0],
            "frp": [0.0, 5.0, 10.0, 20.0, 30.0, 40.0],
            "acq_date": ["2025-07-01"] * 6,
            "satellite": ["N21"] * 6,
        }
    )


@pytest.fixture
def boundaries():
    def feature(name, west):
        return {
            "type": "Feature",
            "properties": {"Propinsi": name},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[west, -3], [west + 1, -3], [west + 1, 0], [west, 0], [west, -3]]],
            },
        }

    return {
        "type": "FeatureCollection",
        "features": [feature("KEPULAUAN RIAU", 104), feature("RIAU", 102)],
    }
