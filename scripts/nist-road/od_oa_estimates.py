# %%
# import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import warnings
from nird.utils import load_config

warnings.simplefilter("ignore")
dafni_path = Path(load_config()["paths"]["base_path"])
nist_path = Path(load_config()["paths"]["nist_path"])
nist_path = nist_path / "incoming" / "20260216 - inputs to OxfUni models"


# %%
# futrue population estimates at the LAD level (calibrated with subnational estimates)
oa_shp = gpd.read_parquet(
    dafni_path
    / "census_datasets"
    / "admin_census_boundary_stats"
    / "gb_oa_2021_estimates.geoparquet"
)
oa_ppp = pd.read_parquet(nist_path / "processed" / "oa21_ppp_estimates.pq")
oa_hhh = pd.read_parquet(nist_path / "processed" / "oa21_hhh_estimates.pq")
# %%
oa_ppp["origin"] = oa_ppp["OA21CD"]
oa_hhh["origin"] = oa_hhh["OA21CD"]
oa_ppp["destination"] = oa_ppp["OA21CD"]
oa_hhh["destination"] = oa_hhh["OA21CD"]
# %%
# OD flow projection
# load 2021 OD data
od_oa_2021 = pd.read_csv(
    dafni_path / "census_datasets" / "od_matrix" / "od_gb_oa_2021.csv"
)
od_oa_2021 = od_oa_2021[
    (od_oa_2021["Area of usual residence"].str.startswith("E"))
    & (od_oa_2021["Area of workplace"].str.startswith("E"))
].reset_index(drop=True)
od_oa_2021.rename(
    columns={
        "Area of usual residence": "origin",
        "Area of workplace": "destination",
    },
    inplace=True,
)
# %%
od_ppp = (
    od_oa_2021.merge(
        oa_ppp[["origin", "OA_POP_2021", "OA21_PPP_2030", "OA21_PPP_2050"]],
        on="origin",
        how="left",
    )
    .rename(
        columns={
            "OA_POP_2021": "origin_ppp_2021",
            "OA21_PPP_2030": "origin_ppp_2030",
            "OA21_PPP_2050": "origin_ppp_2050",
        }
    )
    .merge(
        oa_ppp[["destination", "OA_POP_2021", "OA21_PPP_2030", "OA21_PPP_2050"]],
        on="destination",
        how="left",
    )
    .rename(
        columns={
            "OA_POP_2021": "destination_ppp_2021",
            "OA21_PPP_2030": "destination_ppp_2030",
            "OA21_PPP_2050": "destination_ppp_2050",
        }
    )
)

# %%
od_hhh = (
    od_oa_2021.merge(
        oa_hhh[["origin", "OA_POP_2021", "OA21_HHH_2030", "OA21_HHH_2050"]],
        on="origin",
        how="left",
    )
    .rename(
        columns={
            "OA_POP_2021": "origin_hhh_2021",
            "OA21_HHH_2030": "origin_hhh_2030",
            "OA21_HHH_2050": "origin_hhh_2050",
        }
    )
    .merge(
        oa_hhh[["destination", "OA_POP_2021", "OA21_HHH_2030", "OA21_HHH_2050"]],
        on="destination",
        how="left",
    )
    .rename(
        columns={
            "OA_POP_2021": "destination_hhh_2021",
            "OA21_HHH_2030": "destination_hhh_2030",
            "OA21_HHH_2050": "destination_hhh_2050",
        }
    )
)
# %%
# projection
od_ppp["Car30"] = (
    od_ppp["Car21"]
    * (od_ppp.origin_ppp_2030 / od_ppp.origin_ppp_2021)
    * (od_ppp.destination_ppp_2030 / od_ppp.destination_ppp_2021)
)
od_ppp["Car50"] = (
    od_ppp["Car21"]
    * (od_ppp.origin_ppp_2050 / od_ppp.origin_ppp_2021)
    * (od_ppp.destination_ppp_2050 / od_ppp.destination_ppp_2021)
)
od_hhh["Car30"] = (
    od_hhh["Car21"]
    * (od_hhh.origin_hhh_2030 / od_hhh.origin_hhh_2021)
    * (od_hhh.destination_hhh_2030 / od_hhh.destination_hhh_2021)
)
od_hhh["Car50"] = (
    od_hhh["Car21"]
    * (od_hhh.origin_hhh_2050 / od_hhh.origin_hhh_2021)
    * (od_hhh.destination_hhh_2050 / od_hhh.destination_hhh_2021)
)

# %%
# export results
od_ppp["Car30"] = od_ppp["Car30"].round().astype(int)
od_ppp["Car50"] = od_ppp["Car50"].round().astype(int)
od_hhh["Car30"] = od_hhh["Car30"].round().astype(int)
od_hhh["Car50"] = od_hhh["Car50"].round().astype(int)
od_ppp.to_parquet(
    nist_path / "outputs" / "roads" / "od_oa_ppp_estimates.pq", index=False
)
od_hhh.to_parquet(
    nist_path / "outputs" / "roads" / "od_oa_hhh_estimates.pq", index=False
)
