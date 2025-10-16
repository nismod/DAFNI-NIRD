# %%
from pathlib import Path
import pandas as pd


base_path = Path(r"P:\mistral\DAFNI_NIRD\results\rerouting_analysis\revision")

# %%
scenario_design = pd.read_csv(
    r"C:\Oxford\Research\DAFNI\local\papers\final\revision\recovery_rates\recovery design_updated.csv"
)
slow_rates = pd.read_csv(
    r"C:\Oxford\Research\DAFNI\local\papers\final\revision\recovery_rates\slow_rates.csv"
)
fast_rates = pd.read_csv(
    r"C:\Oxford\Research\DAFNI\local\papers\final\revision\recovery_rates\fast_rates.csv"
)
ave_rates = pd.read_csv(
    r"C:\Oxford\Research\DAFNI\local\papers\final\revision\recovery_rates\ave_rates.csv"
)
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
            base_path / f"{depth_key}" / f"{event_key}" / "cost_matrix_by_scenario.csv"
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
                base_path
                / f"{depth_key}"
                / f"{event_key}"
                / f"trip_isolations_{scenario_key}.csv"
            )
            df = pd.read_csv(in_path)
            v = df.Car21.sum()  # number
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
