"""GeoSense Phase IV: FastAPI REST API and MCP Streamable HTTP server."""

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.phase4_agent.agent_result_store import get_latest_result

from pydantic import BaseModel, Field
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]

for import_path in (
    ROOT / "src",
    ROOT / "src" / "phase1_ml",
    ROOT / "src" / "phase2_dl",
):
    value = str(import_path)
    if value not in sys.path:
        sys.path.insert(0, value)

from phase4_agent.tools.common import (
    get_db_engine,
    get_earth_engine,
    json_response,
    validate_bbox,
    validate_coordinates,
)
from mcp.server import MCPServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("geosense.phase4")

app = FastAPI(
    title="GeoSense Phase IV API",
    description=(
        "Geospatial site scoring, imagery classification, LiDAR, Street View, "
        "climate, demographics, and nearby candidate-site search."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

mcp = MCPServer(
    name="GeoSense Phase IV",
    version="1.0.0",
    instructions="Use the seven geospatial tools with validated inputs.",
)


# ============================================================
# REQUEST MODELS
# ============================================================

class LocationInput(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class BboxInput(BaseModel):
    min_lat: float = Field(ge=-90, le=90)
    min_lon: float = Field(ge=-180, le=180)
    max_lat: float = Field(ge=-90, le=90)
    max_lon: float = Field(ge=-180, le=180)


class SimilarSitesInput(LocationInput):
    radius_km: float = Field(default=5, gt=0, le=25)
    top_k: int = Field(default=5, ge=1, le=20)


# ============================================================
# SHARED EXECUTION
# ============================================================

def run_analysis(function, *args, **kwargs) -> dict[str, Any]:
    """Execute an analysis and return JSON-safe data."""
    try:
        return json_response(function(*args, **kwargs))

    except HTTPException:
        raise

    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {type(exc).__name__}: {exc}",
        ) from exc


# ============================================================
# TOOL 1: PHASE I SITE SCORING
# ============================================================

def site_score(latitude: float, longitude: float) -> dict:
    validate_coordinates(latitude, longitude)

    from site_scorer import score_site

    return score_site(latitude, longitude)


# ============================================================
# TOOL 2: PHASE II IMAGERY CLASSIFICATION
# ============================================================

def imagery_classification(latitude: float, longitude: float) -> dict:
    validate_coordinates(latitude, longitude)

    from image_classifier import classify_imagery

    return classify_imagery(latitude, longitude)


# ============================================================
# TOOL 3: ERA5-LAND CLIMATE DATA
# ============================================================

def climate_risk(latitude: float, longitude: float) -> dict:
    validate_coordinates(latitude, longitude)

    ee = get_earth_engine()

    point = ee.Geometry.Point([longitude, latitude])

    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate("2020-01-01", "2025-01-01")
        .select([
            "temperature_2m",
            "total_precipitation_sum",
        ])
    )

    values = collection.mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=11000,
        maxPixels=100000,
    ).getInfo()

    if not values or all(
        value is None for value in values.values()
    ):
        raise RuntimeError(
            "No ERA5-Land climate values are available at this location."
        )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "period": "2020-2024",
        "source": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "mean_values": values,
        "interpretation": (
            "Dataset band values are returned without assigning a "
            "climate-risk category. Verify band units before interpretation."
        ),
    }


# ============================================================
# TOOL 4: WORLDPOP DEMOGRAPHICS
# ============================================================

def demographics(latitude: float, longitude: float) -> dict:
    validate_coordinates(latitude, longitude)

    ee = get_earth_engine()

    point = ee.Geometry.Point([longitude, latitude])

    image = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("year", 2020))
        .first()
    )

    if image is None:
        raise RuntimeError(
            "WorldPop population image for 2020 was not found."
        )

    values = image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=100,
        maxPixels=10000,
    ).getInfo()

    if not values or all(
        value is None for value in values.values()
    ):
        raise RuntimeError(
            "No WorldPop pixel value is available at this location."
        )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "year": 2020,
        "source": "WorldPop/GP/100m/pop",
        "pixel_values": values,
        "interpretation": (
            "Raw raster pixel value. Verify the image and band metadata "
            "before treating the value as a population count."
        ),
    }


# ============================================================
# TOOL 5: NEARBY CANDIDATE SITES IN POSTGIS
# ============================================================

def similar_sites(
    latitude: float,
    longitude: float,
    radius_km: float = 5,
    top_k: int = 5,
) -> dict:
    validate_coordinates(latitude, longitude)

    if not 0 < radius_km <= 25:
        raise ValueError(
            "radius_km must be greater than 0 and at most 25"
        )

    if not 1 <= top_k <= 20:
        raise ValueError(
            "top_k must be between 1 and 20"
        )

    engine = get_db_engine()

    query = text(
        """
        WITH query_point AS (
            SELECT ST_SetSRID(
                ST_MakePoint(:lon, :lat), 4326
            ) AS geom
        )
        SELECT
            c.location_id,
            c.latitude,
            c.longitude,
            c.dist_road_m,
            c.dist_hospital_m,
            c.population_value,
            c.elevation_m,
            c.land_use_class,
            ST_Distance(
                ST_Transform(c.geometry, 4326)::geography,
                q.geom::geography
            ) AS distance_m
        FROM candidate_locations AS c
        CROSS JOIN query_point AS q
        WHERE ST_DWithin(
            ST_Transform(c.geometry, 4326)::geography,
            q.geom::geography,
            :radius_m
        )
        ORDER BY distance_m
        LIMIT :top_k
        """
    )

    try:
        with engine.connect() as connection:
            rows = connection.execute(
                query,
                {
                    "lat": latitude,
                    "lon": longitude,
                    "radius_m": radius_km * 1000,
                    "top_k": top_k,
                },
            ).mappings().all()

        return json_response({
            "query_location": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "radius_km": radius_km,
            "count": len(rows),
            "sites": [dict(row) for row in rows],
            "method": (
                "Nearest candidate locations by geographic distance"
            ),
            "limitation": (
                "This is a proximity search, not a feature-based "
                "similarity score."
            ),
        })

    finally:
        engine.dispose()


# ============================================================
# TOOL 6: LIDAR PROFILE
# ============================================================

def lidar_result(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> dict:
    return {
        "status": "data_unavailable",
        "message": (
            "LiDAR profiling cannot be performed because no usable "
            "LAS/LAZ point-cloud dataset or Phase III LiDAR module "
            "has been configured."
        ),
        "bounding_box": {
            "min_lat": min_lat,
            "min_lon": min_lon,
            "max_lat": max_lat,
            "max_lon": max_lon,
        },
        "lidar_points_processed": 0,
        "profile_generated": False,
    }

# ============================================================
# TOOL 7: STREET VIEW SCENE
# ============================================================
def streetview_result(
    latitude: float,
    longitude: float,
) -> dict:
    return {
        "status": "data_unavailable",
        "message": (
            "Street View scene analysis cannot be performed because "
            "no usable street-level imagery or analysis module "
            "has been configured."
        ),
        "location": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "scene_analyzed": False,
        "scene_description": None,
    }


@app.get("/agent/latest_result")
def agent_latest_result():
    """Return the latest result published by the GeoSense agent."""
    result = get_latest_result()
    if result is None:
        return {"status": "empty", "result": None}
    return {"status": "ok", "result": result}

# ============================================================
# FASTAPI REST ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "service": "GeoSense Phase IV",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "mcp": "/mcp",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "GeoSense Phase IV",
    }


@app.post(
    "/tools/score_location",
    description="Score a site using the Phase I suitability model.",
)
def score_location(data: LocationInput):
    return run_analysis(
        site_score,
        data.latitude,
        data.longitude,
    )


@app.post(
    "/tools/classify_imagery",
    description="Classify satellite imagery using the Phase II model.",
)
def classify_image(data: LocationInput):
    return run_analysis(
        imagery_classification,
        data.latitude,
        data.longitude,
    )


@app.post(
    "/tools/lidar_profile",
    description="Profile LiDAR data within a geographic bounding box.",
)
def lidar_profile(data: BboxInput):
    try:
        validate_bbox(
            data.min_lat,
            data.min_lon,
            data.max_lat,
            data.max_lon,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return run_analysis(
        lidar_result,
        data.min_lat,
        data.min_lon,
        data.max_lat,
        data.max_lon,
    )


@app.post(
    "/tools/streetview_scene",
    description="Analyze a Street View scene at a coordinate.",
)
def streetview_scene(data: LocationInput):
    return run_analysis(
        streetview_result,
        data.latitude,
        data.longitude,
    )


@app.post(
    "/tools/get_climate_risk",
    description="Retrieve ERA5-Land climate values at a coordinate.",
)
def get_climate_risk(data: LocationInput):
    return run_analysis(
        climate_risk,
        data.latitude,
        data.longitude,
    )


@app.post(
    "/tools/get_demographics",
    description="Retrieve WorldPop raster values at a coordinate.",
)
def get_demographics(data: LocationInput):
    return run_analysis(
        demographics,
        data.latitude,
        data.longitude,
    )


@app.post(
    "/tools/find_similar_sites",
    description="Find nearby candidate locations in PostGIS.",
)
def find_similar_sites(data: SimilarSitesInput):
    return run_analysis(
        similar_sites,
        data.latitude,
        data.longitude,
        data.radius_km,
        data.top_k,
    )


# ============================================================
# MCP TOOLS
# ============================================================

@mcp.tool(
    description="Score a geographic location using the Phase I suitability model."
)
def mcp_score_location(
    latitude: float,
    longitude: float,
) -> dict:
    return json_response(
        site_score(latitude, longitude)
    )


@mcp.tool(
    description="Classify satellite imagery at a geographic location."
)
def mcp_classify_imagery(
    latitude: float,
    longitude: float,
) -> dict:
    return json_response(
        imagery_classification(latitude, longitude)
    )


@mcp.tool(
    description="Profile LiDAR data within a geographic bounding box."
)
def mcp_lidar_profile(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> dict:
    validate_bbox(
        min_lat,
        min_lon,
        max_lat,
        max_lon,
    )

    return json_response(
        lidar_result(
            min_lat,
            min_lon,
            max_lat,
            max_lon,
        )
    )


@mcp.tool(
    description="Analyze a Street View scene at a geographic location."
)
def mcp_streetview_scene(
    latitude: float,
    longitude: float,
) -> dict:
    return json_response(
        streetview_result(latitude, longitude)
    )


@mcp.tool(
    description="Retrieve ERA5-Land climate values at a geographic location."
)
def mcp_get_climate_risk(
    latitude: float,
    longitude: float,
) -> dict:
    return json_response(
        climate_risk(latitude, longitude)
    )


@mcp.tool(
    description="Retrieve WorldPop raster values at a geographic location."
)
def demographics(latitude: float, longitude: float) -> dict:
    validate_coordinates(latitude, longitude)

    ee = get_earth_engine()
    point = ee.Geometry.Point([longitude, latitude])

    image = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("year", 2020))
        .filter(ee.Filter.eq("country", "IND"))
        .first()
    )

    values = (
        ee.Image(image)
        .select("population")
        .reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=100,
            maxPixels=10000,
        )
        .getInfo()
    )

    population_value = values.get("population") if values else None

    if population_value is None:
        raise RuntimeError(
            "No WorldPop population pixel is available "
            "at the requested location."
        )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "year": 2020,
        "country": "IND",
        "source": "WorldPop/GP/100m/pop",
        "band": "population",
        "population_value": population_value,
        "note": (
            "WorldPop raster pixel value. This is not necessarily "
            "a census count for the exact coordinate."
        ),
    }

@mcp.tool(
    description="Find nearby candidate locations in PostGIS."
)
def mcp_find_similar_sites(
    latitude: float,
    longitude: float,
    radius_km: float = 5,
    top_k: int = 5,
) -> dict:
    return json_response(
        similar_sites(
            latitude,
            longitude,
            radius_km,
            top_k,
        )
    )



# ============================================================
# MOUNT MCP STREAMABLE HTTP ENDPOINT
# ============================================================

from contextlib import asynccontextmanager


@asynccontextmanager
async def mcp_lifespan(fastapi_app):
    async with mcp.session_manager.run():
        yield


app.router.lifespan_context = mcp_lifespan

app.mount(
    "/mcp",
    mcp.streamable_http_app(
        streamable_http_path="/"
    ),
)

