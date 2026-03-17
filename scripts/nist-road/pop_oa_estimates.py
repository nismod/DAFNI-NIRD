# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd
from nird.utils import load_config

dafni_path = Path(load_config()["paths"]["base_path"])
nist_path = Path(load_config()["paths"]["nist_path"])
nist_path = nist_path / "incoming" / "20260216 - inputs to OxfUni models"

# %%
cluster = gpd.read_parquet(
    nist_path / "20260216 - inputs to OxfUni models" / "cluster.parquet"
)

cluster_pop_ppp = pd.read_csv(
    nist_path / "20260216 - inputs to OxfUni models" / "cluster_pop_ppp_forecasts.csv"
)

cluster_pop_hhh = pd.read_csv(
    nist_path / "20260216 - inputs to OxfUni models" / "cluster_pop_hhh_forecasts.csv"
)

oa21 = gpd.read_parquet(
    dafni_path
    / "census_datasets"
    / "admin_census_boundary_stats"
    / "gb_oa_2021_estimates.geoparquet"
)
oa21 = oa21[oa21.OA21CD.str.startswith("E")].reset_index(drop=True)  # England only

# %%
joined = gpd.sjoin(
    cluster[["cluster_id", "geometry"]],
    oa21[["OA21CD", "geometry"]],
    how="left",
    predicate="intersects",
)

joined = joined.merge(oa21[["OA21CD", "OA_POP_2021"]], on="OA21CD", how="left")
grouped = joined.groupby(by=["cluster_id"], as_index=False).agg(
    {"OA21CD": list, "OA_POP_2021": list}
)
grouped["ratio"] = grouped["OA_POP_2021"].apply(
    lambda lst: [v / sum(lst) for v in lst] if sum(lst) != 0 else [0] * len(lst)
)
joined = grouped.explode(["OA21CD", "OA_POP_2021", "ratio"], ignore_index=True)

# %%
cluster_pop_ppp["cluster_ppp_2025"] = cluster_pop_ppp["2025"]
cluster_pop_ppp["cluster_ppp_2030"] = cluster_pop_ppp["2030"]
cluster_pop_ppp["cluster_ppp_2050"] = cluster_pop_ppp["2050"]
joined_ppp = joined.merge(
    cluster_pop_ppp[
        ["cluster_id", "cluster_ppp_2025", "cluster_ppp_2030", "cluster_ppp_2050"]
    ],
    on="cluster_id",
    how="left",
)
joined_ppp["OA21_PPP_2025"] = joined_ppp["ratio"] * joined_ppp["cluster_ppp_2025"]
joined_ppp["OA21_PPP_2030"] = joined_ppp["ratio"] * joined_ppp["cluster_ppp_2030"]
joined_ppp["OA21_PPP_2050"] = joined_ppp["ratio"] * joined_ppp["cluster_ppp_2050"]

# %%
cluster_pop_hhh["cluster_hhh_2025"] = cluster_pop_hhh["2025"]
cluster_pop_hhh["cluster_hhh_2030"] = cluster_pop_hhh["2030"]
cluster_pop_hhh["cluster_hhh_2050"] = cluster_pop_hhh["2050"]
joined_hhh = joined.merge(
    cluster_pop_hhh[
        ["cluster_id", "cluster_hhh_2025", "cluster_hhh_2030", "cluster_hhh_2050"]
    ],
    on="cluster_id",
    how="left",
)
joined_hhh["OA21_HHH_2025"] = joined_hhh["ratio"] * joined_hhh["cluster_hhh_2025"]
joined_hhh["OA21_HHH_2030"] = joined_hhh["ratio"] * joined_hhh["cluster_hhh_2030"]
joined_hhh["OA21_HHH_2050"] = joined_hhh["ratio"] * joined_hhh["cluster_hhh_2050"]

# %%
grouped_ppp = (
    joined_ppp.groupby(by=["OA21CD"], as_index=False)[
        ["OA21_PPP_2025", "OA21_PPP_2030", "OA21_PPP_2050"]
    ]
    .sum()
    .merge(oa21[["OA21CD", "OA_POP_2021"]], on="OA21CD", how="left")
)

grouped_hhh = (
    joined_hhh.groupby(by=["OA21CD"], as_index=False)[
        ["OA21_HHH_2025", "OA21_HHH_2030", "OA21_HHH_2050"]
    ]
    .sum()
    .merge(oa21[["OA21CD", "OA_POP_2021"]], on="OA21CD", how="left")
)

# %%
grouped_ppp.to_parquet(nist_path / "processed" / "oa21_ppp_estimates.pq", index=False)
grouped_hhh.to_parquet(nist_path / "processed" / "oa21_hhh_estimates.pq", index=False)
