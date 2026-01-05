# %%
from pathlib import Path
import pandas as pd
import numpy as np
import os
from collections import defaultdict

base_path = Path(r"C:\Oxford\Research\DAFNI\local")
scenario_path = base_path / "papers" / "final" / "revision" / "recovery_rates"
input_path = base_path / "outputs" / "rerouting_analysis" / "revision"

# %%
scenario_design = pd.read_csv(scenario_path / "recovery design_updated.csv")
slow_rates = pd.read_csv(scenario_path / "slow_rates.csv")
fast_rates = pd.read_csv(scenario_path / "fast_rates.csv")
ave_rates = pd.read_csv(scenario_path / "ave_rates.csv")

# %%
# calculate the number of scenarios for each recovery design
cols = scenario_design.columns[1:]
slow_dict = (
    slow_rates.merge(scenario_design, on=cols.tolist(), how="left")
    .value_counts("scenario")
    .reset_index()
    .sort_values("scenario")
    .set_index("scenario")["count"]
    .to_dict()
)
slow_dict = {i: {int(k): v for k, v in slow_dict.items()}.get(i, 0) for i in range(7)}

ave_dict = (
    ave_rates.merge(scenario_design, on=cols.tolist(), how="left")
    .value_counts("scenario")
    .reset_index()
    .sort_values("scenario")
    .set_index("scenario")["count"]
    .to_dict()
)
ave_dict = {i: {int(k): v for k, v in ave_dict.items()}.get(i, 0) for i in range(7)}

fast_dict = (
    fast_rates.merge(scenario_design, on=cols.tolist(), how="left")
    .value_counts("scenario")
    .reset_index()
    .sort_values("scenario")
    .set_index("scenario")["count"]
    .to_dict()
)
fast_dict = {i: {int(k): v for k, v in fast_dict.items()}.get(i, 0) for i in range(7)}

# %%
# rerouting cost
temp_list = []
for depth_key in [15, 30, 60]:
    for event_key in range(1, 17):
        if event_key in [2]:
            continue
        in_path = (
            input_path / f"{depth_key}" / f"{event_key}" / "cost_matrix_by_scenario.csv"
        )
        df = pd.read_csv(in_path)
        df["rerouting_cost_million"] = df["rerouting_cost"] / 1e6  # million GBP
        df["count_slow"] = df["scenario"].map(slow_dict)
        df["count_fast"] = df["scenario"].map(fast_dict)
        df["count_ave"] = df["scenario"].map(ave_dict)
        total_slow = (
            df["rerouting_cost_million"].clip(lower=0) * df["count_slow"]
        ).sum()
        total_fast = (
            df["rerouting_cost_million"].clip(lower=0) * df["count_fast"]
        ).sum()
        total_ave = (df["rerouting_cost_million"].clip(lower=0) * df["count_ave"]).sum()
        print(
            f"depth {depth_key}, event {event_key}, slow {total_slow:.2f}, "
            f"fast {total_fast:.2f}, ave {total_ave:.2f}"
        )
        # inside your loop:
        temp_list.append(
            {
                "depth": depth_key,
                "event": event_key,
                "slow": total_slow,
                "fast": total_fast,
                "ave": total_ave,
            }
        )

# after loop, convert to DataFrame
temp_df = pd.DataFrame(temp_list)

# %%
# isolated trips
temp_list = []
for depth_key in [15, 30, 60]:
    for event_key in range(1, 17):
        for scenario_key in range(0, 7):
            if event_key in [2]:
                continue
            in_path = (
                input_path
                / f"{depth_key}"
                / f"{event_key}"
                / f"trip_isolations_{scenario_key}.csv"
            )
            if os.path.exists(in_path):
                df = pd.read_csv(in_path)
            else:
                in_path = (
                    input_path
                    / f"{depth_key}"
                    / f"{event_key}"
                    / f"trip_isolations_{scenario_key}.pq"
                )
                df = pd.read_parquet(in_path)
                # df.rename(columns={"flow": "Car21"}, inplace=True)

            v = df.flow.sum()  # number of isolated trips
            v_slow = v * slow_dict[scenario_key]
            v_fast = v * fast_dict[scenario_key]
            v_ave = v * ave_dict[scenario_key]
            print(
                f"depth {depth_key}, event {event_key}, scenario {scenario_key}, "
                f"slow {v_slow:.2f}, fast {v_fast:.2f}, ave {v_ave:.2f}"
            )
            temp_list.append(
                {
                    "depth": depth_key,
                    "event": event_key,
                    "scenario": scenario_key,
                    "slow": v_slow,
                    "fast": v_fast,
                    "ave": v_ave,
                }
            )

temp_df = pd.DataFrame(temp_list)
temp_df_gp = temp_df.groupby(["depth", "event"]).sum().reset_index()

# %%
slow_rates = slow_rates.merge(scenario_design, on=cols.tolist(), how="left")  # 160
fast_rates = fast_rates.merge(scenario_design, on=cols.tolist(), how="left")  # 60
ave_rates = ave_rates.merge(scenario_design, on=cols.tolist(), how="left")  # 110

# %%
# rerouting cost over time
event_rerouting_dict = defaultdict(lambda: defaultdict(list))
event_rerouting_stats = defaultdict(dict)
event_rerouting_dfs = defaultdict(dict)


def elementwise_stats_as_arrays(list_of_arrays):
    """
    Compute element-wise min/max/mean for arrays that all come
    from the SAME category (slow/fast/ave).
    They may differ in length — we keep each category independent.
    Returns numpy arrays (mins, maxs, means) of the category's natural length.
    """
    if not list_of_arrays:
        return np.array([]), np.array([]), np.array([])

    # stack using the full length of that category (all arrays MUST match in this category)
    arrays = [np.asarray(a) for a in list_of_arrays]
    stacked = np.vstack(arrays)  # shape (D, N) where N is category length

    mins = np.nanmin(stacked, axis=0).clip(min=0)
    maxs = np.nanmax(stacked, axis=0).clip(min=0)
    means = np.nanmean(stacked, axis=0).clip(min=0)

    return mins, maxs, means


def pad(arr, target=161):
    arr = np.asarray(arr)
    if arr.size >= target:
        return arr[:target]
    out = np.zeros(target, dtype=float)
    out[: arr.size] = arr
    return out


# %%
for event_key in range(1, 17):
    if event_key == 2:
        continue
    else:
        # reset list storage for this event
        event_rerouting_dict[event_key]["slow"] = []
        event_rerouting_dict[event_key]["fast"] = []
        event_rerouting_dict[event_key]["ave"] = []

        for depth_key in [15, 30, 60]:
            in_path = (
                input_path
                / f"{depth_key}"
                / f"{event_key}"
                / "cost_matrix_by_scenario.csv"
            )
            df = pd.read_csv(in_path)
            df["rerouting_cost_million"] = df["rerouting_cost"] / 1e6
            mapping = df.set_index("scenario")["rerouting_cost_million"].to_dict()

            # map into slow/fast/ave scenario orders (their lengths may differ)
            slow_arr = slow_rates["scenario"].map(mapping).to_numpy()
            fast_arr = fast_rates["scenario"].map(mapping).to_numpy()
            ave_arr = ave_rates["scenario"].map(mapping).to_numpy()

            event_rerouting_dict[event_key]["slow"].append(slow_arr)
            event_rerouting_dict[event_key]["fast"].append(fast_arr)
            event_rerouting_dict[event_key]["ave"].append(ave_arr)

            # ---- compute stats (no trimming between categories!) ----
            slow_mins, slow_maxs, slow_means = elementwise_stats_as_arrays(
                event_rerouting_dict[event_key]["slow"]
            )
            fast_mins, fast_maxs, fast_means = elementwise_stats_as_arrays(
                event_rerouting_dict[event_key]["fast"]
            )
            ave_mins, ave_maxs, ave_means = elementwise_stats_as_arrays(
                event_rerouting_dict[event_key]["ave"]
            )

        event_rerouting_stats[event_key] = {
            "slow": {"min": slow_mins, "max": slow_maxs, "mean": slow_means},
            "fast": {"min": fast_mins, "max": fast_maxs, "mean": fast_means},
            "ave": {"min": ave_mins, "max": ave_maxs, "mean": ave_means},
        }

        fast_mins = pad(fast_mins)
        fast_maxs = pad(fast_maxs)
        fast_means = pad(fast_means)

        ave_mins = pad(ave_mins)
        ave_maxs = pad(ave_maxs)
        ave_means = pad(ave_means)

        df = pd.DataFrame(
            {
                "day": np.arange(1, 162),
                "min_slow": slow_mins,
                "max_slow": slow_maxs,
                "mean_slow": slow_means,
                "min_fast": fast_mins,
                "max_fast": fast_maxs,
                "mean_fast": fast_means,
                "min_ave": ave_mins,
                "max_ave": ave_maxs,
                "mean_ave": ave_means,
            },
        )
        df.to_csv(
            scenario_path.parent
            / "results"
            / "stats"
            / "rerouting_costs"
            / f"rerouting_stats_event_{event_key}.csv",
            index=False,
        )

# %%
# trip isolations over time
for depth_key in [15, 30, 60]:
    for event_key in range(1, 17):
        if event_key in [2]:
            continue
        temp = []
        for scenario_key in range(0, 7):
            if event_key in [2]:
                continue
            in_path = (
                input_path
                / f"{depth_key}"
                / f"{event_key}"
                / f"trip_isolations_{scenario_key}.pq"
            )
            df = pd.read_parquet(in_path)
            v = df.flow.sum() / 1e6  # number of isolated trips
            temp.append(v)

        temp_df = pd.DataFrame(
            {
                "scenario": range(0, 7),
                "isolated_trips_million": temp,
            }
        )
        temp_df.to_csv(
            input_path
            / f"{depth_key}"
            / f"{event_key}"
            / "isolation_matrix_by_scenario.csv",
            index=False,
        )

# %%
event_rerouting_dict = defaultdict(lambda: defaultdict(list))
event_rerouting_stats = defaultdict(dict)
event_rerouting_dfs = defaultdict(dict)

for event_key in range(1, 17):
    if event_key == 2:
        continue
    else:
        # reset list storage for this event
        event_rerouting_dict[event_key]["slow"] = []
        event_rerouting_dict[event_key]["fast"] = []
        event_rerouting_dict[event_key]["ave"] = []

        for depth_key in [15, 30, 60]:
            in_path = (
                input_path
                / f"{depth_key}"
                / f"{event_key}"
                / "isolation_matrix_by_scenario.csv"
            )
            df = pd.read_csv(in_path)
            mapping = df.set_index("scenario")["isolated_trips_million"].to_dict()

            # map into slow/fast/ave scenario orders (their lengths may differ)
            slow_arr = slow_rates["scenario"].map(mapping).to_numpy()
            fast_arr = fast_rates["scenario"].map(mapping).to_numpy()
            ave_arr = ave_rates["scenario"].map(mapping).to_numpy()

            event_rerouting_dict[event_key]["slow"].append(slow_arr)
            event_rerouting_dict[event_key]["fast"].append(fast_arr)
            event_rerouting_dict[event_key]["ave"].append(ave_arr)

            # ---- compute stats (no trimming between categories!) ----
            slow_mins, slow_maxs, slow_means = elementwise_stats_as_arrays(
                event_rerouting_dict[event_key]["slow"]
            )
            fast_mins, fast_maxs, fast_means = elementwise_stats_as_arrays(
                event_rerouting_dict[event_key]["fast"]
            )
            ave_mins, ave_maxs, ave_means = elementwise_stats_as_arrays(
                event_rerouting_dict[event_key]["ave"]
            )

        event_rerouting_stats[event_key] = {
            "slow": {"min": slow_mins, "max": slow_maxs, "mean": slow_means},
            "fast": {"min": fast_mins, "max": fast_maxs, "mean": fast_means},
            "ave": {"min": ave_mins, "max": ave_maxs, "mean": ave_means},
        }

        fast_mins = pad(fast_mins)
        fast_maxs = pad(fast_maxs)
        fast_means = pad(fast_means)

        ave_mins = pad(ave_mins)
        ave_maxs = pad(ave_maxs)
        ave_means = pad(ave_means)

        df = pd.DataFrame(
            {
                "day": np.arange(1, 162),
                "min_slow": slow_mins,
                "max_slow": slow_maxs,
                "mean_slow": slow_means,
                "min_fast": fast_mins,
                "max_fast": fast_maxs,
                "mean_fast": fast_means,
                "min_ave": ave_mins,
                "max_ave": ave_maxs,
                "mean_ave": ave_means,
            },
        )

    df.to_csv(
        scenario_path.parent
        / "results"
        / "stats"
        / "isolations"
        / f"isolation_stats_event_{event_key}.csv",
        index=False,
    )
