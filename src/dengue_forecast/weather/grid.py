from __future__ import annotations

import math
from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

EQUAL_AREA_CRS = "EPSG:6933"


@dataclass(frozen=True)
class GridCell:
    cell_id: str
    geometry: BaseGeometry
    latitude: float | None = None
    longitude: float | None = None


def build_area_weights(
    districts, cells: list[BaseGeometry | GridCell]
) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    district_gdf = gpd.GeoDataFrame(
        districts, geometry="geometry", crs=getattr(districts, "crs", None) or "EPSG:4326"
    )
    cell_records: list[dict[str, object]] = []
    for index, cell in enumerate(cells):
        if isinstance(cell, GridCell):
            cell_records.append(
                {
                    "cell_id": cell.cell_id,
                    "geometry": cell.geometry,
                    "latitude": cell.latitude,
                    "longitude": cell.longitude,
                }
            )
        else:
            cell_records.append(
                {"cell_id": str(index), "geometry": cell, "latitude": None, "longitude": None}
            )
    cell_gdf = gpd.GeoDataFrame(cell_records, geometry="geometry", crs="EPSG:4326")

    districts_eq = district_gdf.to_crs(EQUAL_AREA_CRS)
    cells_eq = cell_gdf.to_crs(EQUAL_AREA_CRS)
    records: list[dict[str, object]] = []
    for _, district in districts_eq.iterrows():
        district_area = float(district.geometry.area)
        for _, cell in cells_eq.iterrows():
            intersection = district.geometry.intersection(cell.geometry)
            if intersection.is_empty:
                continue
            area = float(intersection.area)
            if area <= 0:
                continue
            records.append(
                {
                    "district_id": district["district_id"],
                    "district_name": district["district_name"],
                    "cell_id": cell["cell_id"],
                    "latitude": cell.get("latitude"),
                    "longitude": cell.get("longitude"),
                    "intersection_area_m2": area,
                    "district_area_m2": district_area,
                    "weight": area / district_area,
                }
            )
    return pd.DataFrame(records)


def native_grid_cells_for_bounds(
    bounds: tuple[float, float, float, float], resolution: float
) -> list[GridCell]:
    minx, miny, maxx, maxy = bounds

    lon_start = math.floor((minx - resolution / 2) / resolution) * resolution
    lon_end = math.ceil((maxx + resolution / 2) / resolution) * resolution
    lat_start = math.floor((miny - resolution / 2) / resolution) * resolution
    lat_end = math.ceil((maxy + resolution / 2) / resolution) * resolution
    cells: list[GridCell] = []
    lat = lat_start
    while lat <= lat_end + 1e-9:
        lon = lon_start
        while lon <= lon_end + 1e-9:
            geom = box(
                lon - resolution / 2,
                lat - resolution / 2,
                lon + resolution / 2,
                lat + resolution / 2,
            )
            cells.append(
                GridCell(
                    cell_id=f"{lat:.4f},{lon:.4f}",
                    geometry=geom,
                    latitude=round(lat, 6),
                    longitude=round(lon, 6),
                )
            )
            lon += resolution
        lat += resolution
    return cells
