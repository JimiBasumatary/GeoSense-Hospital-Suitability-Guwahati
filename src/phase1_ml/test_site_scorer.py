import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

import pytest

from site_scorer import score_site


# ============================================================
# TEST 1 — VALID COORDINATES
# ============================================================

def test_valid_coordinates():

    result = score_site(
        latitude=26.128019883114455,
        longitude=91.63553723918953
    )

    assert isinstance(result, dict)

    assert result["latitude"] == pytest.approx(
        26.128019883114455
    )

    assert result["longitude"] == pytest.approx(
        91.63553723918953
    )


# ============================================================
# TEST 2 — PREDICTED CLASS
# ============================================================

def test_predicted_suitability():

    result = score_site(
        latitude=26.128019883114455,
        longitude=91.63553723918953
    )

    assert result["predicted_suitability"] in [
        "Low",
        "Moderate",
        "High"
    ]


# ============================================================
# TEST 3 — CONFIDENCE RANGE
# ============================================================

def test_confidence_range():

    result = score_site(
        latitude=26.128019883114455,
        longitude=91.63553723918953
    )

    assert 0.0 <= result["confidence"] <= 1.0


# ============================================================
# TEST 4 — DISTANCES ARE NON-NEGATIVE
# ============================================================

def test_distances_non_negative():

    result = score_site(
        latitude=26.128019883114455,
        longitude=91.63553723918953
    )

    assert result["dist_road_m"] >= 0

    assert result["dist_hospital_m"] >= 0

    assert result["dist_landuse_commercial_m"] >= 0


# ============================================================
# TEST 5 — INVALID LATITUDE
# ============================================================

def test_invalid_latitude():

    with pytest.raises(ValueError):

        score_site(
            latitude=100,
            longitude=91.63553723918953
        )


# ============================================================
# TEST 6 — INVALID LONGITUDE
# ============================================================

def test_invalid_longitude():

    with pytest.raises(ValueError):

        score_site(
            latitude=26.128019883114455,
            longitude=200
        )


# ============================================================
# TEST 7 — REQUIRED OUTPUT FIELDS
# ============================================================

def test_required_output_fields():

    result = score_site(
        latitude=26.128019883114455,
        longitude=91.63553723918953
    )

    required_fields = [
        "latitude",
        "longitude",
        "dist_road_m",
        "dist_hospital_m",
        "population_value",
        "elevation_m",
        "dist_landuse_commercial_m",
        "land_use_class",
        "prediction",
        "predicted_suitability",
        "confidence"
    ]

    for field in required_fields:

        assert field in result


# ============================================================
# TEST 8 — PREDICTION CODE
# ============================================================

def test_prediction_code():

    result = score_site(
        latitude=26.128019883114455,
        longitude=91.63553723918953
    )

    assert result["prediction"] in [0, 1, 2]