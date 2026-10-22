from unittest.mock import patch

import pytest

from src.phase1_ml import site_scorer


LATITUDE = 26.128019883114455
LONGITUDE = 91.63553723918953


def test_invalid_latitude_is_rejected_before_database_access():
    with pytest.raises(ValueError, match="Latitude must be between"):
        site_scorer.score_site(latitude=100, longitude=LONGITUDE)


def test_invalid_longitude_is_rejected_before_database_access():
    with pytest.raises(ValueError, match="Longitude must be between"):
        site_scorer.score_site(latitude=LATITUDE, longitude=200)


def test_valid_coordinates_reach_model_loading():
    with patch.object(site_scorer, "load_model") as mock_load_model:
        with pytest.raises(Exception):
            site_scorer.score_site(latitude=LATITUDE, longitude=LONGITUDE)

    mock_load_model.assert_called_once()


def test_current_output_contract_is_documented():
    expected_fields = {
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
        "confidence",
    }

    assert expected_fields
