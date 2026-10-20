"""Shared utilities for GeoSense Phase IV tools."""

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")


def validate_coordinates(latitude: float, longitude: float) -> None:
    """Reject coordinates outside valid geographic ranges."""
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90")
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180")


def validate_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> None:
    """Validate geographic bounding-box coordinates."""
    for lat, lon in ((min_lat, min_lon), (max_lat, max_lon)):
        validate_coordinates(lat, lon)

    if min_lat >= max_lat:
        raise ValueError("min_lat must be less than max_lat")
    if min_lon >= max_lon:
        raise ValueError("min_lon must be less than max_lon")


def to_jsonable(value: Any) -> Any:
    """Convert common Python and scientific values into JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (ValueError, TypeError):
            pass

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if hasattr(value, "wkt"):
        return value.wkt

    return str(value)


def json_response(data: Any) -> dict:
    """Return a JSON-safe dictionary response."""
    if isinstance(data, dict):
        return to_jsonable(data)
    return {"result": to_jsonable(data)}


@lru_cache(maxsize=1)
def get_db_engine() -> Engine:
    """Create a reusable PostGIS SQLAlchemy engine from .env settings."""
    required = ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise RuntimeError(
            "Missing database settings in .env: " + ", ".join(missing)
        )

    host = os.environ["DB_HOST"]
    port = os.getenv("DB_PORT", "5432")
    name = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]

    from sqlalchemy.engine import URL

    url = URL.create(
        "postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=name,
    )
    return create_engine(url, pool_pre_ping=True)


def get_earth_engine():
    """Initialize Earth Engine only when a tool requests it."""
    import ee

    project = (
        os.getenv("EE_PROJECT")
        or os.getenv("EARTH_ENGINE_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
    )

    if not project:
        raise RuntimeError(
            "Earth Engine project is not configured. Set EE_PROJECT in .env "
            "to an existing Google Cloud project enabled for Earth Engine."
        )

    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Check authentication, "
            "project access, and Earth Engine registration."
        ) from exc

    return ee