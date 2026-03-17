# %%
from pathlib import Path
import pandas as pd
from nird.utils import load_config
import json
import ast
from collections import defaultdict
import warnings

pd.set_option("display.max_columns", None)
warnings.simplefilter("ignore")

nist_path = Path(load_config()["paths"]["nist_path"])
nist_path = nist_path / "incoming" / "20260216 - inputs to OxfUni models"

# %%
# timetable
timetable = pd.read_parquet(nist_path / "processed_data" / "timetable_20251219.pq")
# trains
with open(nist_path / "processed_data" / "daily_train_schedule.json", "r") as f:
    trains = json.load(f)
selected_trains = trains["2025-05-19"]

# od pairs
with open(
    nist_path / "processed_data" / "origin_to_destinations_2025-05-19.json", "r"
) as f:
    orig_to_dest = json.load(f)

od = (
    pd.Series(orig_to_dest)
    .explode()
    .reset_index()
    .rename(columns={"index": "origin", 0: "destination"})
)

# %%
# explode path and stops
train_stops = timetable.assign(
    station=timetable["path"], stop=timetable["stops"]
).explode(["station", "stop"])

# keep only stopping stations
train_stops = train_stops[train_stops["stop"] == "S"]
train_stops = train_stops[["train_id", "station", "stop"]]


# %%
def parse_list_field(x, delimiter="|"):
    """Robustly parse list-like fields (list, np.array, JSON string, Python literal, or delimited string)."""
    if x is None:
        return []
    # already a list/tuple/set
    if isinstance(x, (list, tuple, set)):
        return list(x)
    # numpy array
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    # string cases
    if isinstance(x, str):
        # try JSON
        try:
            parsed = json.loads(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        # try python literal like "['A','B']"
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        # fallback to delimiter split (empty string -> [''])
        return x.split(delimiter)
    # anything else -> try to coerce to list
    try:
        return list(x)
    except Exception:
        return []


def explode_merge_chunks_get_train_ids(
    timetable: pd.DataFrame,
    od: pd.DataFrame,
    chunk_size: int = 20000,
    path_col: str = "path",
    stops_col: str = "stops",
    train_id_col: str = "train_id",
    delimiter: str = "|",
) -> pd.DataFrame:
    """
    Return od DataFrame with an additional column 'train_ids' listing train_id(s)
    that stop at both origin and destination for that OD row.

    Assumes `timetable` and `od` are already loaded into memory (but timetable is large;
    timetable is iterated in chunks via .iloc slices to reduce peak memory).
    """
    # reset/keep od index as integer index for mapping back
    od_indexed = od.reset_index(drop=False).rename(
        columns={"index": "orig_index"}
    )  # orig_index is original position
    od_indexed = (
        od_indexed.rename(columns={"index": "_drop_index"})
        if "index" in od_indexed.columns
        else od_indexed
    )
    # Build quick lookup frames for merging
    od_orig = od_indexed[["orig_index", "origin"]].rename(columns={"origin": "station"})
    od_dest = od_indexed[["orig_index", "destination"]].rename(
        columns={"destination": "station"}
    )

    # accumulator: od_index -> set(train_id)
    od_to_trains = defaultdict(set)

    total = len(timetable)
    for start in range(0, total, chunk_size):
        sub = timetable.iloc[start : start + chunk_size]
        # Build list of (train_id, station) only for S stops in this chunk
        rows = []  # list of tuples (train_id, station)
        for _, r in sub[[train_id_col, path_col, stops_col]].iterrows():
            tid = r[train_id_col]
            p = parse_list_field(r[path_col], delimiter=delimiter)
            s = parse_list_field(r[stops_col], delimiter=delimiter)
            # align lengths
            if len(p) != len(s):
                m = min(len(p), len(s))
                p = p[:m]
                s = s[:m]
            # collect only stations with stop == 'S'
            for station, stop_flag in zip(p, s):
                # explicit equality check (avoid numpy boolean confusion)
                if stop_flag == "S":
                    rows.append((tid, station))
        if not rows:
            continue

        stops_df = pd.DataFrame(
            rows, columns=[train_id_col, "station"]
        ).drop_duplicates()

        # Merge stops with od origins to get (od_idx, train_id) for origin matches
        origin_matches = stops_df.merge(
            od_orig, on="station", how="inner"
        )  # columns: train_id, station, orig_index
        if origin_matches.empty:
            # nothing in this chunk matches any origin
            continue
        origin_matches = (
            origin_matches[["orig_index", train_id_col]]
            .drop_duplicates()
            .rename(columns={"orig_index": "od_idx"})
        )

        # Merge stops with od destinations to get (od_idx, train_id) for destination matches
        dest_matches = stops_df.merge(od_dest, on="station", how="inner")
        if dest_matches.empty:
            continue
        dest_matches = (
            dest_matches[["orig_index", train_id_col]]
            .drop_duplicates()
            .rename(columns={"orig_index": "od_idx"})
        )

        # Inner join on od_idx & train_id to find trains that appear in both origin & destination for same OD row
        # (Note: origin_matches and dest_matches both use od_idx referring to the same OD rows)
        both = origin_matches.merge(
            dest_matches, on=["od_idx", train_id_col], how="inner"
        )  # columns: od_idx, train_id

        if both.empty:
            continue

        # accumulate train_ids into od_to_trains map
        for od_idx, tid in both[["od_idx", train_id_col]].itertuples(
            index=False, name=None
        ):
            od_to_trains[od_idx].add(tid)

        # optional: free memory of this chunk's temporary dfs
        del stops_df, origin_matches, dest_matches, both

    # Prepare final result: list of train_ids per original od order
    n_od = len(od_indexed)
    train_lists = [None] * n_od
    for i in range(n_od):
        s = od_to_trains.get(i, set())
        train_lists[i] = sorted(s) if s else []

    od_out = od.copy().reset_index(drop=True)  # ensure clean integer index
    od_out["train_ids"] = train_lists

    return od_out


od_with_trains = explode_merge_chunks_get_train_ids(
    timetable=timetable,
    od=od,
    chunk_size=20000,
    path_col="path",
    stops_col="stops",
    train_id_col="train_id",
    delimiter="|",
)

od_with_trains["num_of_trains"] = od_with_trains.train_ids.apply(len)
od_with_trains.to_parquet(nist_path / "processed_data" / "od_trains.pq")
