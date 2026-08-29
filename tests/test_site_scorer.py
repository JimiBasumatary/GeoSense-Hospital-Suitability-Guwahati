from src.phase1_ml.site_scorer import score_site


def test_score_site_returns_valid_prediction():
    result = score_site(
        building_area_m2=200,
        buildings_within_250m=12,
        dist_road_m=5,
        pois_within_500m=6,
        dominant_landuse="residential",
        dominant_landuse_area_m2=5000
    )

    assert result["prediction"] in [0, 1]
    assert result["label"] in ["Suitable", "Not suitable"]
    assert 0.0 <= result["probability"] <= 1.0


def test_score_site_returns_all_expected_fields():
    result = score_site(
        building_area_m2=150,
        buildings_within_250m=10,
        dist_road_m=10,
        pois_within_500m=5,
        dominant_landuse="forest",
        dominant_landuse_area_m2=10000
    )

    assert "prediction" in result
    assert "label" in result
    assert "probability" in result


def test_score_site_handles_unmapped_landuse():
    result = score_site(
        building_area_m2=180,
        buildings_within_250m=12,
        dist_road_m=8,
        pois_within_500m=4,
        dominant_landuse="No mapped land use",
        dominant_landuse_area_m2=0
    )

    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["probability"] <= 1.0