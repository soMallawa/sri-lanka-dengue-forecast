from __future__ import annotations

import geopandas as gpd

from dengue_forecast.contracts import validate_district_reference

PROVINCE_BY_DISTRICT: dict[str, str] = {
    "LK-COL": "Western",
    "LK-GAM": "Western",
    "LK-KAL": "Western",
    "LK-KAN": "Central",
    "LK-MAT": "Central",
    "LK-NUW": "Central",
    "LK-GAL": "Southern",
    "LK-MAR": "Southern",
    "LK-HAM": "Southern",
    "LK-JAF": "Northern",
    "LK-KIL": "Northern",
    "LK-MAN": "Northern",
    "LK-MUL": "Northern",
    "LK-VAV": "Northern",
    "LK-BAT": "Eastern",
    "LK-AMP": "Eastern",
    "LK-TRI": "Eastern",
    "LK-KUR": "North Western",
    "LK-PUT": "North Western",
    "LK-ANU": "North Central",
    "LK-POL": "North Central",
    "LK-BAD": "Uva",
    "LK-MON": "Uva",
    "LK-RAT": "Sabaragamuwa",
    "LK-KEG": "Sabaragamuwa",
}


def build_district_reference(boundaries: gpd.GeoDataFrame, population) -> gpd.GeoDataFrame:  # type: ignore[no-untyped-def]
    merged = boundaries.merge(
        population.drop(columns=["district_name"]),
        on="district_id",
        how="left",
        validate="one_to_one",
    )
    merged["province_name"] = merged["district_id"].map(PROVINCE_BY_DISTRICT)
    merged["population_density_per_km2"] = merged["population_reference"] / merged["area_km2"]

    columns = [
        "district_id",
        "district_name",
        "province_name",
        "population_reference",
        "population_reference_year",
        "population_method",
        "population_density_per_km2",
        "area_km2",
        "centroid_lat",
        "centroid_lon",
        "geometry",
        "geometry_source",
        "geometry_source_version",
        "population_source",
        "population_source_version",
    ]
    out = gpd.GeoDataFrame(merged[columns], geometry="geometry", crs=boundaries.crs)
    validate_district_reference(out)
    return out.sort_values("district_id").reset_index(drop=True)
