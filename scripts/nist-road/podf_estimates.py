# %%
import sys
import pandas as pd

from pathlib import Path
from nird.utils import load_config
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import logging
import warnings

warnings.simplefilter("ignore")
pd.set_option("display.max_columns", None)


def get_paths():
    cfg = load_config()
    nist_path = Path(cfg["paths"]["NIST"])  # path to NIST folder
    out_path = nist_path.parent / "outputs"
    return nist_path, out_path


def load_odpfc(od: str):
    _, out_path = get_paths()
    odpfc_out_path = out_path / f"odpfc_{od}.pq"  # od_agg / od_agg_2050
    if not odpfc_out_path.exists():
        raise FileNotFoundError(f"{odpfc_out_path} does not exist.")
    return ds.dataset(odpfc_out_path, format="parquet")


def main(od: str, batch_size: int = 100_000):
    dataset = load_odpfc(od)
    scanner = dataset.scanner(
        columns=["origin_node", "destination_node", "path", "flow"],
        batch_size=batch_size,
    )
    from collections import defaultdict

    # Accumulate aggregated flows across all batches in a dict
    acc = defaultdict(float)

    for batch in scanner.to_batches():
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

        # Update global accumulator
        for row in out_df.itertuples(index=False):
            acc[(row.path, row.origin_node, row.destination_node)] += row.flow

    # Build final DataFrame aggregated across all batches
    if len(acc) > 0:
        final_df = pd.DataFrame(
            [(k[0], k[1], k[2], v) for k, v in acc.items()],
            columns=["path", "origin_node", "destination_node", "flow"],
        )
        final_df = final_df.sort_values(
            by=["path", "origin_node", "destination_node"]
        ).reset_index(drop=True)

        # ensure output directory exists and write single parquet file
        _, out_path = get_paths()
        out_path.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(final_df, preserve_index=False)
        pq.write_table(table, out_path / f"odpfc_{od}_expanded.parquet")
    else:
        logging.info("No data to write for od=%s", od)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        od = sys.argv[1]
        main(od)
    except (IndexError, NameError):
        logging.info("Provide input parameters!")
