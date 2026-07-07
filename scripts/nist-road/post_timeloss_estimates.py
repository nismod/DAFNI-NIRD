# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd  # type: ignore
import warnings
from tqdm import tqdm
import re
import json
import logging
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from nird.utils import load_config

warnings.simplefilter("ignore")
nist_path = Path(load_config()["paths"]["NIST"])


# %%
# estimate time loss due to congestion (aggregated to LAD level)
def parse_path_cell(x):
    # handle missing
    if pd.isna(x):
        return []
    # already a real list
    if isinstance(x, list):
        return x
    # bytes -> str
    if isinstance(x, (bytes, bytearray)):
        x = x.decode()
    s = str(x)
    # grab alpha-numeric + underscore tokens (adjust regex if other chars appear)
    tokens = re.findall(r"[A-Za-z0-9_]+", s)
    return tokens


def calculate_time(edge_flow):
    free_spd = edge_flow["free_flow_speeds"]  # mph
    length = edge_flow.geometry.length / 1609.34  # meter to miles
    congest_spd = edge_flow["acc_speed"]  # mph
    edge_flow["length_miles"] = length  # miles
    edge_flow["time_free"] = length / free_spd * 60  # minutes
    edge_flow["time_congested"] = length / congest_spd * 60  # minutes
    return edge_flow[["e_id", "length_miles", "time_free", "time_congested"]]


def process_batch(batch, edge_flow, node_to_lad):
    batch = batch.reset_index(drop=True).copy()
    batch["__od_idx"] = np.arange(len(batch))
    batch["path"] = batch["path"].apply(parse_path_cell)
    explored = batch.explode("path").rename(columns={"path": "e_id"})
    merged = explored.merge(edge_flow, on="e_id", how="left")
    agg = merged.groupby("__od_idx", sort=False)[
        ["length_miles", "time_free", "time_congested"]
    ].sum()
    agg = agg.reindex(range(len(batch)), fill_value=0).reset_index(drop=True)
    agg = agg.rename(
        columns={
            "length_miles": "total_length_miles",
            "time_free": "total_time_free",
            "time_congested": "total_time_congested",
        }
    )
    result = batch.merge(agg, left_on="__od_idx", right_index=True, how="left")
    result["timeloss(sec/km)"] = (
        1 / (result.total_length_miles / result.total_time_congested.replace(0, np.nan))
        - 1 / (result.total_length_miles / result.total_time_free.replace(0, np.nan))
    ) * 37.3
    result["LAD24CD"] = result["origin_node"].map(node_to_lad)
    return result.drop(columns=["path", "__od_idx"])


def main(future_scenario):
    # load model inputs
    with open(nist_path / "tables" / "node_to_lad24_updated.json", "rb") as f:
        node_to_lad = json.load(f)

    # load computing results
    edge_flow = gpd.read_parquet(
        nist_path.parent / "outputs" / f"edge_flow_{future_scenario}.gpq"
    )
    edge_flow = calculate_time(edge_flow)

    input_path = nist_path.parent / "outputs" / f"odpfc_{future_scenario}.pq"
    output_path = nist_path.parent / "outputs" / f"od_time_{future_scenario}.pq"

    if output_path.exists():
        output_path.unlink()

    chunksize = 100_000
    parquet_file = pq.ParquetFile(input_path)
    writer = None

    try:
        for batch in tqdm(
            parquet_file.iter_batches(batch_size=chunksize),
            desc="Processing batches",
            unit="batch",
        ):
            processed = process_batch(batch.to_pandas(), edge_flow, node_to_lad)
            table = pa.Table.from_pandas(processed, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        future_scenario = sys.argv[1]
        main(future_scenario)
    except (IndexError, NameError):
        logging.info("Provide input parameters!")
