import sys
from pathlib import Path
import pandas as pd
import ibis as ib
from ibis import _
import itertools
import time
import logging

from nird.utils import load_config

ib.options.interactive = True
conn = ib.connect("duckdb://", **{"memory_limit": "150GB", "threads": 30})
conn.sql("""SELECT * FROM duckdb_settings();""").execute()
conn.sql("""SELECT current_setting('threads') as threads;""").execute()

base_path = Path(load_config()["paths"]["soge_clusters"])


def update_edge_speed(
    table,
    combined_label: str,
    total_flow: float,
    initial_flow_speed: float,
    min_flow_speed: float,
    breakpoint_flow: float,
):
    vp = total_flow / 24
    return table.mutate(
        acc_speed=ib.least(
            ib.cases(
                (
                    (combined_label == "M") & (vp > breakpoint_flow),
                    initial_flow_speed - 0.033 * (vp - breakpoint_flow),
                ),
                (
                    (combined_label == "A_single") & (vp > breakpoint_flow),
                    initial_flow_speed - 0.05 * (vp - breakpoint_flow),
                ),
                (
                    (combined_label == "A_dual") & (vp > breakpoint_flow),
                    initial_flow_speed - 0.033 * (vp - breakpoint_flow),
                ),
                (
                    (combined_label == "B") & (vp > breakpoint_flow),
                    initial_flow_speed - 0.05 * (vp - breakpoint_flow),
                ),
                else_=initial_flow_speed,
            ),
            min_flow_speed,
        )
    )


def apply_recovery(
    table,
    recovery_rates_dict: dict,
    day,
):
    event_day = bool(
        int(
            recovery_rates_dict[day]
            .filter(_.damage_level == "event_day")
            .recovery_rate.execute()[0]
        )
    )
    return table.join(
        recovery_rates_dict[day],
        predicates=[_.damage_level == recovery_rates_dict[day].damage_level],
        how="left",
    ).mutate(
        acc_capacity=ib.cases(
            (_.damage_level.isin(["extensive", "severe"]) & event_day, 0),
            (_.damage_level_max == "no", _.current_capacity),
            else_=_.current_capacity * _.recovery_rate,
        )
    )


def main(
    depth_key,
    event_key,
    start_day,
    end_day,
):
    # base_path: .../preprocessed_data
    od = conn.read_parquet(
        base_path.parent
        / "results"
        / "base_scenario"
        / "revision"
        / "odpfc_grouped.pq"
    )
    recovery_rates = pd.read_csv(base_path / "tables" / "recovery design_updated.csv")
    recovery_rates_dict = {
        key: (
            conn.create_table(
                f"bridge_recovery_dict_{key}",
                obj=(
                    pd.DataFrame(
                        data=[val]
                        # ,index = ['damage_level','recovery_rate']
                    ).T.reset_index(drop=False)
                ),
                overwrite=True,
            ).rename({"damage_level": "col0", "recovery_rate": "col1"})
        )
        for key, val in recovery_rates.to_dict(orient="index").items()
    }

    scenarios = itertools.product(
        [depth_key], [event_key], range(start_day, end_day + 1)
    )
    for depth, event, day in scenarios:
        start = time.time()
        road_links = conn.read_parquet(
            base_path.parent
            / "results"
            / "disruption_analysis"
            / "revision"
            / f"{depth}"
            / "links"
            / f"road_links_{event}.gpq"
        )

        road_links = road_links.mutate(
            breakpoint_flows=_.combined_label.cases(
                ("M", 1200), ("A_dual", 1080), ("A_single", 1200), ("B", 1200)
            )
        )

        disrupted_links = (
            road_links.filter(
                ib.or_(
                    _.max_speed < _.current_speed,
                    _.damage_level_max.isin(["extensive", "severe"]),
                )
            )
            .select("e_id")
            .distinct()
            .execute()["e_id"]
            .tolist()
        )

        # Applying recovery (update road link capacities)
        cols = road_links.columns
        road_links = (
            road_links.mutate(acc_capacity=_.current_capacity)
            .mutate(damage_level=_.road_label + "_" + _.damage_level_max)
            .pipe(apply_recovery, recovery_rates_dict=recovery_rates_dict, day=day)
            .select(*cols, "acc_capacity")
        )

        recovery_links = road_links.select("e_id", "acc_capacity")

        od_disrupted = (
            od.select("path", "origin_node", "destination_node")
            .mutate(od_id=_.origin_node.concat(":", _.destination_node))
            .unnest("path")
            .group_by("path")
            .agg(od_id=_.od_id.collect())
            .join(
                recovery_links,
                predicates=[_.path == recovery_links.e_id],
                how="left",
            )
            .filter(_.path.isin(disrupted_links))
            .unnest("od_id")
            .group_by("od_id")
            .agg(
                disrupted_links=_.path.collect(), min_link_capacity=_.acc_capacity.min()
            )
            .mutate(
                origin_node=_.od_id.re_split(":")[0],
                destination_node=_.od_id.re_split(":")[1],
            )
            .select(
                "origin_node",
                "destination_node",
                "disrupted_links",
                "min_link_capacity",
            )
            .join(
                od,
                predicates=["origin_node", "destination_node"],
                how="inner",
            )
        ).cache()

        # Initialise key variables for flow simulation
        links = (
            od_disrupted.mutate(
                disrupted_flows=ib.greatest(_.flow - _.min_link_capacity, 0)
            )
            .unnest("path")
            .group_by("path")
            .agg(disrupted_flows=_.disrupted_flows.sum())
            .join(road_links, predicates=[_.path == road_links.e_id], how="right")
            .mutate(
                disrupted_flows=ib.cases(
                    (_.disrupted_flows.isnull(), 0),  # if null, link is not disrupted!
                    else_=_.disrupted_flows,
                )
            )
            .mutate(
                acc_capacity=_.acc_capacity + _.disrupted_flows,
                acc_flow=_.current_flow - _.disrupted_flows,
            )
            .pipe(
                update_edge_speed,
                combined_label=_.combined_label,
                total_flow=_.acc_flow,
                initial_flow_speed=_.initial_flow_speeds,
                min_flow_speed=_.min_flow_speeds,
                breakpoint_flow=_.breakpoint_flows,
            )
            .select(*road_links.columns, "acc_flow", "acc_speed")
        )

        # Export to parquet files
        od_disrupted.to_parquet(
            base_path.parent
            / "results"
            / "disruption_analysis"
            / "revision"
            / "ibis_results"
            / f"{depth_key}"
            / f"disrupted_od_depth{depth}_event{event}_day{day}.pq",
            compression="ZSTD",
        )
        links.to_parquet(
            base_path.parent
            / "results"
            / "disruption_analysis"
            / "revision"
            / "ibis_results"
            / f"{depth_key}"
            / f"road_links_depth{depth}_event{event}_day{day}.gpq",
            compression="ZSTD",
        )
        print(f"time(sec): {(time.time() - start)}")


if __name__ == "__main__":
    try:
        depth_key = int(sys.argv[1])
    except (IndexError, ValueError):
        logging.info(
            "Error: Please provide the flood depth for road closure "
            "(e.g., 15, 30 or 60 cm) as the first argument!"
        )
        sys.exit(1)
    try:
        event_key = int(sys.argv[2])
    except (IndexError, ValueError):
        logging.info("Error: Please provide the flood key!")
        sys.exit(1)
    try:
        start_day = int(sys.argv[3])
    except (IndexError, ValueError):
        logging.info("Error: Please provide the number of CPUs as the second argument!")
        sys.exit(1)
    try:
        end_day = int(sys.argv[4])
    except (IndexError, ValueError):
        logging.info("Error: Please provide the number of CPUs as the second argument!")
        sys.exit(1)

    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    main(depth_key, event_key, start_day, end_day)
