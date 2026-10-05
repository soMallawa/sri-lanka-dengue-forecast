from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from dengue_forecast.config import DISTRICTS, normalize_district_name
from dengue_forecast.contracts import ContractError

GEOMETRY_URL = (
    "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09/"
    "releaseData/gbOpen/LKA/ADM2/geoBoundaries-LKA-ADM2.geojson"
)
GEOMETRY_SOURCE = "geoBoundaries gbOpen LKA ADM2"
GEOMETRY_SOURCE_VERSION = (
    "9469f09; represented_year=2017; source=OpenStreetMap/Wambacher; license=ODbL-1.0"
)
EQUAL_AREA_CRS = "EPSG:6933"
OVERLAP_TOLERANCE_KM2 = 0.01


def _strip_district_suffix(value: str) -> str:
    cleaned = str(value).strip()
    if cleaned.casefold().endswith(" district"):
        cleaned = cleaned[: -len(" district")]
    return cleaned.strip()


def load_district_boundaries(path: Path | str) -> gpd.GeoDataFrame:
    """Load pinned geoBoundaries ADM2 polygons and normalize to canonical district IDs."""

    gdf = gpd.read_file(path)
    if len(gdf) != 25:
        raise ContractError(f"Expected 25 boundary features, found {len(gdf)}")
    if "shapeName" not in gdf.columns:
        raise ContractError("Boundary file missing shapeName")
    if gdf.crs is None:
        raise ContractError("Boundary file missing CRS")
    gdf = gdf.to_crs("EPSG:4326")
    if not gdf.geometry.is_valid.all():
        raise ContractError("Boundary file contains invalid geometries")

    records: list[dict[str, object]] = []
    for _, row in gdf.iterrows():
        district_id, district_name = normalize_district_name(
            _strip_district_suffix(row["shapeName"])
        )
        records.append(
            {
                "district_id": district_id,
                "district_name": district_name,
                "shape_name": row["shapeName"],
                "shape_iso": row.get("shapeISO"),
                "geometry": row.geometry,
            }
        )

    out = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    expected_ids = {district.district_id for district in DISTRICTS}
    actual_ids = set(out["district_id"])
    if actual_ids != expected_ids:
        raise ContractError(f"Boundary district IDs mismatch: {sorted(expected_ids ^ actual_ids)}")
    if out["district_id"].duplicated().any():
        raise ContractError("Boundary file contains duplicate canonical districts")

    equal_area = out.to_crs(EQUAL_AREA_CRS)
    area_km2 = equal_area.geometry.area / 1_000_000
    union_area_km2 = equal_area.geometry.union_all().area / 1_000_000
    overlap_km2 = float(area_km2.sum() - union_area_km2)
    if overlap_km2 > OVERLAP_TOLERANCE_KM2:
        raise ContractError(f"Boundary overlap exceeds tolerance: {overlap_km2:.6f} km2")

    centroids = equal_area.geometry.centroid
    centroid_ll = gpd.GeoSeries(centroids, crs=EQUAL_AREA_CRS).to_crs("EPSG:4326")
    out["area_km2"] = area_km2.to_numpy()
    out["centroid_lon"] = centroid_ll.x.to_numpy()
    out["centroid_lat"] = centroid_ll.y.to_numpy()
    out["geometry_source"] = GEOMETRY_SOURCE
    out["geometry_source_version"] = GEOMETRY_SOURCE_VERSION
    out = out.sort_values("district_id").reset_index(drop=True)
    return out


def write_boundary_artifacts(gdf: gpd.GeoDataFrame, geojson_path: Path, parquet_path: Path) -> None:
    """Write canonical geometry outputs without mutating source artifacts."""

    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(geojson_path, driver="GeoJSON")
    gdf.to_parquet(parquet_path, index=False)


def boundary_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "district_id": gdf["district_id"],
            "district_name": gdf["district_name"],
            "area_km2": gdf["area_km2"],
            "centroid_lat": gdf["centroid_lat"],
            "centroid_lon": gdf["centroid_lon"],
        }
    )
