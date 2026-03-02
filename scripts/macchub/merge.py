from pathlib import Path
from nird.utils import load_config
import sys
import pyarrow as pa
import pyarrow.parquet as pq

macc_path = Path(load_config()["paths"]["MACCHUB"])

def merge_parquet_parts(part_files, out_file):
    """
    Stream-merge a list of parquet part file paths (pyarrow/parquet readable)
    into one parquet file using a ParquetWriter. Does not load all data at once.
    """
    part_files = [Path(p) for p in part_files]
    if not part_files:
        raise ValueError("No part files provided")

    # Use the schema of the first file
    first_table = pq.read_table(part_files[0])
    writer = pq.ParquetWriter(out_file, first_table.schema)

    try:
        writer.write_table(first_table)
        for p in part_files[1:]:
            tbl = pq.read_table(p)  # reads into arrow table (not pandas), reasonably memory efficient
            writer.write_table(tbl)
    finally:
        writer.close()

def main(event_key):
    parts = sorted(Path(macc_path.parent / "outputs" / f"{event_key}").glob(f"odpfc_{event_key}_part*.pq"))
    merge_parquet_parts(parts, macc_path.parent / "outputs" / f"odpfc_{event_key}.pq")

if __name__ == "__main__":
    event_key = sys.argv[1]
    main(event_key)
