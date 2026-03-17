# %%
from typing import Dict
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from nird.utils import load_config
import json

import warnings

warnings.simplefilter("ignore")
nist_path = Path(load_config()["paths"]["nist_path"])
nist_path = nist_path / "incoming" / "20260216 - inputs to OxfUni models"


# %%
# compute the total flow and population of all origins to individual city centres
# given a time threshold (e.g.,30mins)
def summarize_by_time(
    od_df: pd.DataFrame,
    time_col: str,
    threshold: float,
    pop_dict: Dict[str, float],
    to_col: str = "to_MSOA21CD",
    from_col: str = "from_MSOA21CD",
) -> pd.DataFrame:
    """
    Filter OD table by time_col <= threshold, group by 'to_MSOA21CD',
    sum flows, keep ordered-unique list of from_MSOA21CD, compute pop sum
    from pop_dict and return df with columns ['flow','pop'].
    """
    # 1) filter
    sel = od_df[od_df[time_col] <= threshold].reset_index(drop=True)
    if sel.empty:
        return pd.DataFrame(columns=["flow", "pop"])

    # 2) group: sum flow and gather ordered unique from_MSOA21CD
    grouped = sel.groupby(to_col, as_index=False).agg(
        {"flow": "sum", from_col: lambda s: list(dict.fromkeys(s))}  # ordered unique
    )

    # 3) compute pop efficiently: explode -> map -> groupby sum
    # give each row an id so we can aggregate back after explode
    grouped = grouped.reset_index().rename(columns={"index": "_grp_id"})
    exploded = grouped[["_grp_id", from_col]].explode(from_col)

    # map MSOA -> pop, missing -> 0
    exploded["pop_part"] = exploded[from_col].map(pop_dict).fillna(0)

    # sum pop_part per group id
    pop_by_grp = exploded.groupby("_grp_id", observed=True)["pop_part"].sum()

    # attach pop back to grouped
    grouped["pop"] = grouped["_grp_id"].map(pop_by_grp).fillna(0).astype(float)

    # final select & tidy
    result = grouped[["to_MSOA21CD", "flow", "pop"]].copy()
    result.rename(columns={"to_MSOA21CD": "MSOA21CD"}, inplace=True)

    return result


def future_pop_estimate_msoa(future_year, future_scenario):
    with open(nist_path / "processed_data" / "oa21_to_msoa21.json", "r") as f:
        oa21_to_msoa21 = json.load(f)
    pop_oa = pd.read_parquet(
        nist_path / "processed_data" / f"oa21_{future_scenario}_estimates.pq"
    )
    pop_oa["MSOA21CD"] = pop_oa["OA21CD"].map(oa21_to_msoa21)
    pop_msoa = pop_oa.groupby(by="MSOA21CD", as_index=False).agg(
        {f"OA21_{future_scenario.upper()}_20{future_year}": "sum"}
    )
    msoa_pop_dict = pop_msoa.set_index("MSOA21CD")[
        f"OA21_{future_scenario.upper()}_20{future_year}"
    ].to_dict()
    pop_msoa.rename(
        columns={
            "MSOA21CD": "from_MSOA21CD",
            f"OA21_{future_scenario.upper()}_20{future_year}": f"MSOA21_{future_scenario.upper()}_{future_year}",
        },
        inplace=True,
    )

    return pop_msoa, msoa_pop_dict


def attach_city_by_max_overlap(
    tests: gpd.GeoDataFrame,
    cities: gpd.GeoDataFrame,
    city_code_col: str = "TCITY15CD",
    city_name_col: str = "TCITY15NM",
    test_id_col: str = "test_id",
) -> gpd.GeoDataFrame:

    tests = tests.reset_index(drop=True).copy()

    # Ensure a public test id exists without also using it as the index label.
    if test_id_col not in tests.columns:
        tests[test_id_col] = range(len(tests))

    internal_id_col = "_test_row_id"
    while internal_id_col in tests.columns:
        internal_id_col = f"_{internal_id_col}"
    tests[internal_id_col] = range(len(tests))

    # Ensure CRS alignment
    if tests.crs != cities.crs:
        cities = cities.to_crs(tests.crs)

    # Spatial join to get candidate intersecting pairs (fast)
    joined = gpd.sjoin(
        tests[[internal_id_col, "geometry"]],
        cities[[city_code_col, city_name_col, "geometry"]],
        how="left",
        predicate="intersects",
    ).rename(columns={"index_right": "city_index"})

    if joined.empty:
        # no candidates -> return tests with empty city columns
        tests[[city_code_col, city_name_col]] = pd.NA
        return tests.drop(columns=[internal_id_col])

    # Filter out rows with no matched city (city_index is NaN)
    matched = joined[~joined["city_index"].isna()].copy()
    if matched.empty:
        tests[[city_code_col, city_name_col]] = pd.NA
        return tests.drop(columns=[internal_id_col])

    # city_index refers to cities_m.index (int). Ensure integer dtype for indexing
    # Note: sjoin returns index_right referencing the index of the right GeoDataFrame
    matched["city_index"] = matched["city_index"].astype(int)

    # Vectorized intersection area calculation:
    # Build arrays of test geometries and matched city geometries aligned row-wise
    test_geoms = matched["geometry"].values
    # Map city geometries by matched.city_index
    # cities_m.loc[matched.city_index, "geometry"].values would fail if index is not monotonic,
    # so we use .reindex to align
    city_geoms_for_rows = cities["geometry"].reindex(matched["city_index"]).values

    # compute intersections and areas. Use list comprehension for clarity and robustness
    int_areas = []
    for tg, cg in zip(test_geoms, city_geoms_for_rows):
        if tg is None or cg is None:
            int_areas.append(0.0)
            continue
        inter = tg.intersection(cg)
        if inter.is_empty:
            int_areas.append(0.0)
        else:
            int_areas.append(float(inter.area))

    matched = matched.assign(intersection_area=np.array(int_areas))

    # keep only positive intersections
    matched = matched[matched["intersection_area"] > 0].copy()
    if matched.empty:
        tests[[city_code_col, city_name_col]] = pd.NA
        return tests.drop(columns=[internal_id_col])

    # For each test pick the city with largest intersection area
    idx = matched.groupby(internal_id_col)["intersection_area"].idxmax()
    best = matched.loc[idx, [internal_id_col, city_code_col, city_name_col]]

    result = tests.merge(best, on=internal_id_col, how="left")
    return gpd.GeoDataFrame(
        result.drop(columns=[internal_id_col, test_id_col]),
        geometry="geometry",
        crs=tests.crs,
    )


def main(future_year, future_scenario):
    major_city_shp = gpd.read_parquet(
        nist_path / "processed_data" / "majorcity_shp.gpq"
    )
    msoa_shp = gpd.read_parquet(nist_path / "processed_data" / "msoa21_shp.gpq")
    msoa_cities = gpd.read_parquet(
        nist_path / "processed_data" / f"msoa_cities_20{future_year}.gpq"
    )
    od_file = pd.read_parquet(
        nist_path / "outputs" / "roads" / f"od_time_{future_year}_{future_scenario}.pq"
    )
    with open(nist_path / "processed_data" / "node_to_msoa21.json", "r") as f:
        node_to_msoa = json.load(f)

    od_time = od_file.copy()
    od_time.drop(columns=["LAD24CD"], inplace=True)
    od_time["from_MSOA21CD"] = od_time["origin_node"].map(node_to_msoa)
    od_time["to_MSOA21CD"] = od_time["destination_node"].map(node_to_msoa)

    msoa_cities = msoa_cities.MSOA21CD.unique().tolist()
    selected_od = od_time[od_time["to_MSOA21CD"].isin(msoa_cities)].reset_index(
        drop=True
    )
    msoa_pop, msoa_pop_dict = future_pop_estimate_msoa(future_year, future_scenario)
    selected_od = selected_od.merge(
        msoa_pop[["from_MSOA21CD", f"MSOA21_{future_scenario.upper()}_{future_year}"]],
        on="from_MSOA21CD",
        how="left",
    )

    # free-flow scenario
    selected_free_gp = summarize_by_time(
        selected_od,
        time_col="total_time_free",
        threshold=30,
        pop_dict=msoa_pop_dict,
    )
    print(f"Results for 20{future_year}-{future_scenario}:")
    print(
        f"30 mins-free: pop: {selected_free_gp['pop'].sum():,.0f}, "
        f"flow: {selected_free_gp['flow'].sum():,.0f}"
    )
    # congested scenario
    selected_congested_gp = summarize_by_time(
        selected_od,
        time_col="total_time_congested",
        threshold=30,
        pop_dict=msoa_pop_dict,
    )
    print(
        f"30 mins-congested: pop: {selected_congested_gp['pop'].sum():,.0f}, "
        f"flow: {selected_congested_gp['flow'].sum():,.0f}"
    )

    # estimate congestion impacts
    selected_free_gp.rename(
        columns={
            "flow": f"flow_free_{future_scenario}_{future_year}",
            "pop": f"pop_free_{future_scenario}_{future_year}",
        },
        inplace=True,
    )
    selected_congested_gp.rename(
        columns={
            "flow": f"flow_congested_{future_scenario}_{future_year}",
            "pop": f"pop_congested_{future_scenario}_{future_year}",
        },
        inplace=True,
    )
    combined = selected_free_gp.merge(selected_congested_gp, on="MSOA21CD", how="outer")
    combined["flow_change"] = (
        combined[f"flow_congested_{future_scenario}_{future_year}"]
        - combined[f"flow_free_{future_scenario}_{future_year}"]
    )
    combined["pop_change"] = (
        combined[f"pop_congested_{future_scenario}_{future_year}"]
        - combined[f"pop_free_{future_scenario}_{future_year}"]
    )
    combined = combined.merge(
        msoa_shp[["MSOA21CD", "geometry"]], on="MSOA21CD", how="left"
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="27700")

    # overlay with major city boundary to get city information
    combined = attach_city_by_max_overlap(combined, major_city_shp)
    # combined.to_parquet(
    #     nist_path / "outputs" / "roads" / f"30mins_{future_scenario}_{future_year}.gpq"
    # )

    group = combined.groupby(by=["TCITY15NM", "TCITY15CD"], as_index=False).sum(
        numeric_only=True
    )
    group.to_csv(
        nist_path / "outputs" / "roads" / f"30mins_{future_scenario}_{future_year}.csv"
    )


# %%
# evaluate accessibility reduction due to congestion in the future
main(30, "ppp")
main(30, "hhh")
main(50, "ppp")
main(50, "hhh")
