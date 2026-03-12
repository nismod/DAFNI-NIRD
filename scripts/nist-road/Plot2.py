# %%
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nird.utils import load_config

nist_path = Path(load_config()["paths"]["nist_path"])
nist_path = nist_path / "incoming" / "20260216 - inputs to OxfUni models"
in_path = nist_path / "outputs" / "roads" / "access_variations.csv"

# %%
# Load data and calculate ranks
df = pd.read_csv(in_path)
scenario_columns = [
    ("Baseline", "Baseline"),
    ("2030_PPP", "2030 PPP"),
    ("2030_HHH", "2030 HHH"),
    ("2050_PPP", "2050 PPP"),
    ("2050_HHH", "2050 HHH"),
]
missing_columns = [col for col, _ in scenario_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing scenario columns in input table: {missing_columns}")

panel_configs = [
    {
        "columns": ["Baseline", "2030_PPP", "2050_PPP"],
        "labels": ["Baseline", "2030 PPP", "2050 PPP"],
        "sort_by": "2050_PPP",
        "title": "Baseline vs PPP Scenarios",
    },
    {
        "columns": ["Baseline", "2030_HHH", "2050_HHH"],
        "labels": ["Baseline", "2030 HHH", "2050 HHH"],
        "sort_by": "2050_HHH",
        "title": "Baseline vs HHH Scenarios",
    },
]


def prepare_rank_data(
    dataframe: pd.DataFrame, scenario_cols: list[str], scenario_labels: list[str]
):
    ranked = dataframe[["TCITY15NM", *scenario_cols]].copy()
    rank_col_names = []
    scenario_map = {}
    for column_name, display_name in zip(scenario_cols, scenario_labels):
        rank_col = f"rank_{column_name}"
        ranked[rank_col] = ranked[column_name].rank(ascending=False, method="min")
        rank_col_names.append(rank_col)
        scenario_map[rank_col] = display_name
    rank_df = ranked[["TCITY15NM", *rank_col_names]]
    rank_melted = rank_df.melt(
        id_vars="TCITY15NM", var_name="Scenario", value_name="Rank"
    )
    rank_melted["Scenario"] = pd.Categorical(
        rank_melted["Scenario"].map(scenario_map),
        categories=scenario_labels,
        ordered=True,
    )
    return rank_melted


def plot_bump_panel(
    ax, rank_melted: pd.DataFrame, top_cities: list[str], panel_title: str
):
    bump_palette = sns.color_palette("husl", len(top_cities))
    scenario_order = rank_melted["Scenario"].cat.categories.tolist()

    for i, city in enumerate(top_cities):
        city_data = rank_melted[rank_melted["TCITY15NM"] == city]
        color = bump_palette[i]
        ax.plot(
            city_data["Scenario"],
            city_data["Rank"],
            marker="o",
            markersize=10,
            linewidth=4,
            color=color,
            alpha=0.9,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

        baseline_rank = city_data[city_data["Scenario"] == scenario_order[0]][
            "Rank"
        ].values[0]
        ax.text(
            -0.22,
            baseline_rank,
            city,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
        )

        final_rank = city_data[city_data["Scenario"] == scenario_order[-1]][
            "Rank"
        ].values[0]
        ax.text(
            len(scenario_order) - 1 + 0.06,
            final_rank,
            f"{city} ({int(final_rank)})",
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
        )

    ax.invert_yaxis()
    ax.set_title(panel_title, fontsize=18, fontweight="bold", pad=18)
    ax.set_xlabel("Scenario", fontsize=16, fontweight="bold", labelpad=12)
    ax.set_xticks(range(len(scenario_order)))
    ax.set_xticklabels(scenario_order, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="-", alpha=0.15)
    sns.despine(ax=ax, left=True, bottom=True)


fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=False)
max_rank = 0

for ax, panel in zip(axes, panel_configs):
    top_cities = (
        df.sort_values(panel["sort_by"], ascending=False).head(15)["TCITY15NM"].tolist()
    )
    subset = df[df["TCITY15NM"].isin(top_cities)].copy()
    rank_melted = prepare_rank_data(subset, panel["columns"], panel["labels"])
    max_rank = max(max_rank, int(rank_melted["Rank"].max()))
    plot_bump_panel(ax, rank_melted, top_cities, panel["title"])

axes[0].set_ylabel("Rank Position", fontsize=16, fontweight="bold", labelpad=90)
axes[1].set_ylabel("")
for ax in axes:
    ax.set_yticks(range(1, max_rank + 1, 2))
    ax.tick_params(axis="y", labelsize=12)

fig.suptitle(
    "Urban Ranking Dynamics across PPP and HHH Scenarios",
    fontsize=24,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(nist_path / "outputs" / "roads" / "rank_bump_chart_panels.png", dpi=300)
plt.show()
