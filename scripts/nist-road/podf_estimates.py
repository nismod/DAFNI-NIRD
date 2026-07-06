# %%
from pathlib import Path
from collections import defaultdict
import pandas as pd
import pyarrow.parquet as pq
from nird.utils import load_config
import pickle
import argparse
import logging

nist_path = Path(load_config()["paths"]["NIST"])  # path to NIST folder
table_path = nist_path / "tables"
out_path = nist_path.parent / "outputs"


# %%
def load_and_process_batches(od_file, dict_file, cols=None, batch_size=200_000):
    """Generator: read parquet batches, process shares and path, yield aggregated Series per batch."""
    with open(table_path / f"{dict_file}.pkl", "rb") as f:
        od_meta = pickle.load(f)
    pf = pq.ParquetFile(out_path / f"{od_file}.pq")
    if cols is None:
        cols = ["origin_node", "destination_node", "path", "flow"]

    def _get_shares(o, d):
        v = od_meta.get((o, d))
        if v is None:
            ks1 = f"{o}|{d}"
            ks2 = f"{o},{d}"
            v = od_meta.get(ks1) or od_meta.get(ks2) or {}
        if not isinstance(v, dict):
            return {}
        return v

    for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
        df = batch.to_pandas()

        # Convert string representation to list of edge IDs
        df["path"] = (
            df["path"]
            .fillna("[]")
            .str.strip("[]")
            .str.split(",")
            .apply(lambda x: [s.strip() for s in x if s and str(s).strip()])
        )

        # One row per edge
        df = df.explode("path", ignore_index=True)
        df = df[df["path"].notna()]

        # Attach and expand shares
        df["_shares"] = [
            list(_get_shares(o, d).items()) or [("unknown", 1.0)]
            for o, d in zip(df["origin_node"], df["destination_node"])
        ]
        df = df.explode("_shares", ignore_index=True)
        df[["purpose", "share"]] = pd.DataFrame(df["_shares"].tolist(), index=df.index)
        df["share"] = df["share"].astype(float)
        df.loc[df["share"] > 1, "share"] = df.loc[df["share"] > 1, "share"] / 100.0

        # split flows by share
        df["flow"] = df["flow"].astype(float) * df["share"]

        # Aggregate within batch and yield
        batch_agg = df.groupby(["path", "purpose"])["flow"].sum()
        yield batch_agg


def main(od_file, dict_file, batch_size=200_000):
    totals = defaultdict(float)
    for batch_agg in load_and_process_batches(
        od_file, dict_file, batch_size=batch_size
    ):
        for key, value in batch_agg.items():
            totals[key] += float(value)

    # Final dataframe
    edge_purpose_flow = (
        pd.DataFrame(
            [(edge, purpose, flow) for (edge, purpose), flow in totals.items()],
            columns=["edge", "purpose", "flow"],
        )
        .sort_values(["edge", "purpose"])
        .reset_index(drop=True)
    )
    edge_purpose_flow.to_parquet(
        out_path / f"{od_file}_edge_purpose_flow.pq", index=False
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Aggregate edge-purpose flows from OD parquet and shares pickle"
    )
    parser.add_argument(
        "od_file", help="Input OD parquet basename (without extension .pq)"
    )
    parser.add_argument("dict_file", help="Shares pickle basename (without .pkl)")
    parser.add_argument(
        "--batch-size", type=int, default=200_000, help="Rows per parquet batch"
    )
    args = parser.parse_args()

    main(args.od_file, args.dict_file, args.batch_size)
