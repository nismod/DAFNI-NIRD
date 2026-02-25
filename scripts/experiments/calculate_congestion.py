# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd  # type: ignore
import warnings
from tqdm import tqdm
import gc
from nird.utils import load_config
import json
import logging
import sys

warnings.simplefilter("ignore")
nist_path = Path(load_config()["paths"]["NIST"])


# %%
def calculate_time(edge_flow):
    free_spd = edge_flow["free_flow_speeds"]  # mph
    length = edge_flow.geometry.length / 1609.34  # miles
    congest_spd = edge_flow["acc_speed"]  # mph
    edge_flow["length_miles"] = length
    edge_flow["time_free"] = length / free_spd * 60  # minutes
    edge_flow["time_congested"] = length / congest_spd * 60  # minutes
    return edge_flow[["e_id", "length_miles", "time_free", "time_congested"]]


def process_chunk(chunk, edge_flow):
    chunk = chunk.reset_index(drop=False).rename(columns={"index": "__od_idx"})
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


def main(year):
    # load mappings
    with open(nist_path / "outputs" / "node_to_lad24.json", "r") as f:
        node_to_lad = json.load(f)
    # load lad shapes
    lad = gpd.read_parquet(nist_path / "outputs" / "lad24_shp.gpq")
    # load edge flow
    edge_flow = gpd.read_parquet(nist_path / "outputs" / f"edge_flow_{year}.gpq")
    edge_flow = calculate_time(edge_flow)
    # load od paths
    od = pd.read_parquet(nist_path / "outputs" / f"odpfc_{year}.pq")

    # process in chunks
    chunksize = 100_000
    results = []
    n_chunks = (len(od) + chunksize - 1) // chunksize

    for start in tqdm(
        range(0, len(od), chunksize), total=n_chunks, desc="Processing chunks"
    ):
        chunk = od.iloc[start : start + chunksize].copy()
        res = process_chunk(chunk, edge_flow)
        results.append(res)

        del chunk, res
        gc.collect()

    res_df = pd.concat(results, ignore_index=True)
    del results
    gc.collect()

    # merge results back to od
    od = od.join(res_df, how="left")
    od["LAD24CD"] = od["origin_node"].map(node_to_lad)
    agg = (
        od[["LAD24CD", "total_length_miles", "total_time_free", "total_time_congested"]]
        .groupby(by="LAD24CD")
        .mean()
        .reset_index()
    )
    # export results
    merged = lad.merge(agg, on="LAD24CD", how="left")
    merged.to_parquet(nist_path / "outputs" / f"lad24_time_{year}.gpq")
    od.drop(columns=["path", "orig_index"], inplace=True)
    od.to_parquet(nist_path / "outputs" / f"od_time_{year}.pq")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        year = sys.argv[1]
        main(int(year))
    except (IndexError, NameError):
        logging.info("Enter the year as an argument.")
