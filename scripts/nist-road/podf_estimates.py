# %%
import pandas as pd

from pathlib import Path
from nird.utils import load_config
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import logging
import warnings
from typing import Optional
import argparse
import pickle

warnings.simplefilter("ignore")
pd.set_option("display.max_columns", None)

nist_path = Path(load_config()["paths"]["NIST"])  # path to NIST folder
table_path = nist_path / "tables"
out_path = nist_path.parent / "outputs"

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


def load_shares(shares_file):
    p = table_path / shares_file
    if not p.exists():
        raise FileNotFoundError(p)
    obj = pickle.load(open(p, "rb"))
    if not isinstance(obj, dict):
        raise ValueError("shares pickle must contain a dict-like mapping")
    rows = []
    for k, v in obj.items():
        if isinstance(k, (list, tuple)):
            a, b = k[0], k[1]
        else:
            ks = str(k)
            if "|" in ks:
                a, b = [x.strip() for x in ks.split("|", 1)]
            elif "," in ks:
                a, b = [x.strip() for x in ks.split(",", 1)]
            else:
                raise ValueError(f"Unrecognized key in shares mapping: {k}")
        row = {"origin_node": a, "destination_node": b}
        row.update(v)
        rows.append(row)
    return pd.DataFrame(rows)


def load_odpfc(od: str):
    p = out_path / f"odpfc_{od}.pq"
    if not p.exists():
        raise FileNotFoundError(p)
    return ds.dataset(p, format="parquet")


def main(
    od: str,
    batch_size: int = 100_000,
    shares_file: Optional[str] = None,
):
    dataset = load_odpfc(od)
    scanner = dataset.scanner(
        columns=["origin_node", "destination_node", "path", "flow"],
        batch_size=batch_size,
    )

    from collections import defaultdict

    # Load shares dict file (user-specified filename under table_path)
    if shares_file is None:
        raise ValueError("A shares file must be provided (filename under table_path).")

    shares_df = load_shares(shares_file)

    # Prepare output path and in-memory accumulator for (path, purpose) -> flow
    acc = defaultdict(float)

    # During scanning: write per-batch aggregated rows into shard CSVs to avoid holding everything in memory
    purposes = [
        c for c in shares_df.columns if c not in ("origin_node", "destination_node")
    ]
    if not purposes:
        raise ValueError("No purpose columns found in shares file")

    for batch in tqdm(scanner.to_batches(), desc="Scanning batches", unit="batch"):
        df = batch.to_pandas()
        # Drop rows without a path
        df = df.dropna(subset=["path"]).copy()
        if df.empty:
            continue
        logging.info("processing batch, first path: %s", df.loc[0, "path"][0])
        # Convert string to list
        df["path"] = (
            df["path"]
            .str.strip("[]")
            .str.split(",")
            .apply(lambda x: [s.strip() for s in x])
        )
        # Explode the path column so each edge becomes one row and aggregate within the batch
        out_df = (
            df.explode("path", ignore_index=True)
            .groupby(["path", "origin_node", "destination_node"], as_index=False)[
                "flow"
            ]
            .sum()
        )

        # Merge with shares DataFrame to broadcast purpose shares for each OD
        merged = out_df.merge(
            shares_df, on=["origin_node", "destination_node"], how="left"
        )
        # drop rows without shares
        merged = merged.dropna(subset=purposes, how="all")
        if merged.empty:
            continue

        # compute purpose-specific flows: multiply flow by each purpose share
        for p in purposes:
            merged[p] = merged[p].fillna(0).astype(float)
            merged[p + "_flow"] = merged["flow"] * merged[p]

        # melt purpose flows into long format
        flow_cols = [p + "_flow" for p in purposes]
        melted = merged.melt(
            id_vars=["path", "origin_node", "destination_node"],
            value_vars=flow_cols,
            var_name="purpose",
            value_name="flow",
        )
        # normalize purpose names (remove _flow suffix)
        melted["purpose"] = melted["purpose"].str.replace("_flow", "", regex=False)
        # drop zero or missing flows
        melted = melted[melted["flow"] > 0]

        if melted.empty:
            continue

        # Accumulate per-batch purpose flows into the global accumulator
        for r in melted.itertuples(index=False):
            acc[(r.path, r.purpose)] += float(r.flow)

    # Build final DataFrame aggregated across all batches (path, purpose, flow)
    if len(acc) == 0:
        logging.info("No data to write for od=%s", od)
        return

    final_df = pd.DataFrame(
        [(k[0], k[1], v) for k, v in acc.items()], columns=["path", "purpose", "flow"]
    )
    final_df = final_df.sort_values(by=["path", "purpose"]).reset_index(drop=True)

    # ensure output directory exists and write single parquet file
    out_file = out_path / f"odpfc_{od}_expanded.parquet"
    out_path.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(final_df, preserve_index=False)
    pq.write_table(table, out_file)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("od", help="OD dataset identifier (filename suffix)")
        parser.add_argument(
            "shares_file",
            help="Shares file name located under tables/ (json/csv/parquet)",
        )
        parser.add_argument("--batch-size", type=int, default=100_000)
        args = parser.parse_args()
        main(
            args.od,
            batch_size=args.batch_size,
            shares_file=args.shares_file,
        )
    except (IndexError, NameError):
        logging.info("Provide input parameters!")
