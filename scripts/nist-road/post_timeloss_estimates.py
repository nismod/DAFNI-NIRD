# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd  # type: ignore
import warnings
from tqdm import tqdm
import gc
import ast
import json
import logging
import sys
import numpy as np
from nird.utils import load_config

warnings.simplefilter("ignore")
nist_path = Path(load_config()["paths"]["NIST"])


# %%
# estimate time loss due to congestion (aggregated to LAD level)
def calculate_time(edge_flow):
    free_spd = edge_flow["free_flow_speeds"]  # mph
    length = edge_flow.geometry.length / 1609.34  # meter to miles
    congest_spd = edge_flow["acc_speed"]  # mph
    edge_flow["length_miles"] = length  # miles
    edge_flow["time_free"] = length / free_spd * 60  # minutes
    edge_flow["time_congested"] = length / congest_spd * 60  # minutes
    return edge_flow[["e_id", "length_miles", "time_free", "time_congested"]]


def process_chunk(chunk, edge_flow):
    chunk = chunk.reset_index(drop=False).rename(columns={"index": "__od_idx"})
    chunk["path"] = chunk["path"].apply(ast.literal_eval)
    explored = chunk.explode("path").rename(columns={"path": "e_id"})
    merged = explored.merge(edge_flow, on="e_id", how="left")
    agg = merged.groupby("__od_idx", sort=False)[
        ["length_miles", "time_free", "time_congested"]
    ].sum()
    agg = agg.reindex(range(len(chunk)), fill_value=0).reset_index(drop=True)
    agg = agg.rename(
        columns={
            "length_miles": "total_length_miles",
            "time_free": "total_time_free",
            "time_congested": "total_time_congested",
        }
    )
    agg["orig_index"] = chunk["__od_idx"].values
    return agg


def main(future_year, future_scenario):
    # load model inputs
    with open(nist_path/ "tables" / "node_to_lad24_updated.json", "rb") as f:
        node_to_lad = json.load(f)
    lad = gpd.read_parquet(nist_path / "admins" / "lad24_shp.gpq")

    # load computing results
    edge_flow = gpd.read_parquet(
        nist_path.parent / "outputs" / f"edge_flow_{future_year}_{future_scenario}.gpq"
    )
    edge_flow = calculate_time(edge_flow)
    od = pd.read_parquet(
        nist_path.parent / "outputs" / f"odpfc_{future_year}_{future_scenario}.pq"
    )
    # od["path"] = od["path"].apply(ast.literal_eval)
    print(od.loc[0, "path"][0:50])
    sys.exit(0)

    # chunked process
    chunksize = 100_000
    results = []
    n_chunks = (len(od) + chunksize - 1) // chunksize

    for start in tqdm(
        range(0, len(od), chunksize), total=n_chunks, desc="Processing chunks"
    ):
        chunk = od.iloc[start : start + chunksize].copy()
        res = process_chunk(chunk, edge_flow)  # length_miles, time_free, time_congested
        results.append(res)

        del chunk, res
        gc.collect()

    res_df = pd.concat(results, ignore_index=True)
    del results
    gc.collect()

    print(res_df.columns)
    print(res_df)
    od = od.join(res_df, how="left")
    od["timeloss(sec/km)"] = (
        1 / (od.total_length_miles / od.total_time_congested.replace(0, np.nan))
        - 1 / (od.total_length_miles / od.total_time_free.replace(0, np.nan))
    ) * 37.3
    print(od)
    od["LAD24CD"] = od["origin_node"].map(node_to_lad)
    agg = (
        od[["LAD24CD", "total_time_free", "total_time_congested", "timeloss(sec/km)"]]
        .groupby(by="LAD24CD")
        .mean()
        .reset_index()
    )
    merged = lad.merge(agg, on="LAD24CD", how="left")
    merged.to_parquet(
        nist_path.parent / "outputs" / f"lad24_time_{future_year}_{future_scenario}.gpq"
    )

    od.drop(columns=["path", "orig_index"], inplace=True)
    od.to_parquet(nist_path.parent / "outputs" / f"od_time_{future_year}_{future_scenario}.pq")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        future_year, future_scenario = sys.argv[1:]
        main(int(future_year), future_scenario)
    except (IndexError, NameError):
        logging.info("Provide input parameters!")
