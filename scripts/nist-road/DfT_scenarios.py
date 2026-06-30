# %%
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

# import sqlite3

try:
    import pyodbc
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyodbc is required. Install it with: pip install pyodbc") from exc

scenario = "High"  # !!! update: (Core, High)
# -----------------------------
# User settings
# -----------------------------
BASE_DIR = Path(r"C:\Program Files (x86)\TEMPRO8\DATA")
OUT_DIR = Path(r"C:\Oxford\Research\NIST\DfT Model\processed_data")

# Access SQL filters
# PURPOSES = (1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18)
Town = (4, 5, 6, 7, 8, 14, 15, 16, 18)
Edu = (3, 13)
Emp = (2, 12)
Comm = (1, 11)
# Identify group of purposes
PURPOSES = Edu  #!!! update
# Compute average daily across all selected time periods
MODES = (3, 4)  # 3=Car Driver, 4=Car Passenger
"""TIME_PERIOD.
1-4: this is 5-day total, should be multiplied by 0.2 to get average weekday
1: 0700-1000 (morning peak time)
2: 1000-1600 (inter-peak time)
3: 1600-1900 (evening peak time)
4: 0-7/19-24 (off-peak time)
5: SAT (all times of the day)
6: SUN (all times of the day)
# AADT: annual average daily traffic
# ((1+2+3+4)+5+6)/7 -> this is average day traffic
# (1+2+3+4)/5 -> this is average weekday traffic
"""
TIME_PERIOD = tuple(range(1, 7))  # Compute average daily across all periods (1-6)
TRIP_TYPES = (1, 2)  # 1=Production, 2=Attraction
YEAR_COL = "2061"  #!!! update (2011, 2016, 2021, 2026, 2031, 2036, 2041, 2046, 2051, 2056, 2061)
TABLE_NAME = "TripEndDataByDirection"
# !!! update the outpath
OUTPUT_CSV = OUT_DIR / "high" / f"{scenario.lower()}_education_{YEAR_COL}_outflow.csv"


def access_connection(db_path: Path) -> pyodbc.Connection:
    """Open an Access MDB/ACCDB database using the installed ODBC driver."""
    conn_strs = [
        rf"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};",
        rf"DRIVER={{Microsoft Access Driver (*.mdb)}};DBQ={db_path};",
    ]
    last_err = None
    for conn_str in conn_strs:
        try:
            return pyodbc.connect(conn_str)
        except Exception as exc:  # pragma: no cover
            last_err = exc
    raise RuntimeError(f"Failed to open {db_path}. Last error: {last_err}")


def list_high_databases(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.glob(f"*_{scenario}.mdb") if p.is_file()])


def read_filtered_aggregate(db_path: Path) -> pd.DataFrame:
    purpose_list = ", ".join(str(p) for p in PURPOSES)
    mode_list = ", ".join(str(m) for m in MODES)
    trip_type_list = ", ".join(str(t) for t in TRIP_TYPES)

    # Compute average daily by summing TimePeriods 1-6 and dividing by 7 (no weekday scaling).
    sum_expr = f"SUM([{YEAR_COL}]) / 7"
    where_time = "TimePeriod IN (1,2,3,4,5,6)"

    sql = f"""
        SELECT
            ZoneID,
            TripType,
            {sum_expr} AS Total{YEAR_COL}
        FROM {TABLE_NAME}
        WHERE Purpose IN ({purpose_list})
          AND Mode IN ({mode_list})
          AND {where_time}
          AND TripType IN ({trip_type_list})
        GROUP BY ZoneID, TripType
    """

    with access_connection(db_path) as conn:
        df = pd.read_sql(sql, conn)
        zone_df = pd.read_sql(
            """
            SELECT ZoneID, ZoneName, Authority
            FROM Zones
            """,
            conn,
        )

        # Read Planning counts from Planning table: per-year columns (e.g. [2021])
        try:
            planning_sql = f"""
                SELECT ZoneID,
                    SUM(IIF(PlanningDataType IN (1,2,3), [{YEAR_COL}], 0)) AS Population_{YEAR_COL},
                    SUM(IIF(PlanningDataType = 4, [{YEAR_COL}], 0)) AS Workers_{YEAR_COL},
                    SUM(IIF(PlanningDataType = 5, [{YEAR_COL}], 0)) AS Households_{YEAR_COL},
                    SUM(IIF(PlanningDataType = 6, [{YEAR_COL}], 0)) AS Jobs_{YEAR_COL}
                FROM Planning
                GROUP BY ZoneID
            """
            planning_df = pd.read_sql(planning_sql, conn)
        except Exception:
            planning_df = pd.DataFrame(
                columns=[
                    "ZoneID",
                    f"Population_{YEAR_COL}",
                    f"Workers_{YEAR_COL}",
                    f"Households_{YEAR_COL}",
                    f"Jobs_{YEAR_COL}",
                ]
            )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "SourceDB",
                "ZoneID",
                "ZoneName",
                "Authority",
                f"Total{YEAR_COL}_Production",
                f"Total{YEAR_COL}_Attraction",
                f"Population_{YEAR_COL}",
                f"Workers_{YEAR_COL}",
                f"Households_{YEAR_COL}",
                f"Jobs_{YEAR_COL}",
            ]
        )

    df = df.pivot(index="ZoneID", columns="TripType", values=f"Total{YEAR_COL}")
    df = df.rename(
        columns={
            1: f"Total{YEAR_COL}_Production",
            2: f"Total{YEAR_COL}_Attraction",
        }
    )
    df = df.reset_index().fillna(0)

    zone_df = zone_df.drop_duplicates(subset="ZoneID")
    # Merge planning counts (if present)
    if not planning_df.empty:
        df = df.merge(planning_df, on="ZoneID", how="left")
    else:
        df[f"Population_{YEAR_COL}"] = 0
        df[f"Workers_{YEAR_COL}"] = 0
        df[f"Households_{YEAR_COL}"] = 0
        df[f"Jobs_{YEAR_COL}"] = 0

    df = df.merge(zone_df, on="ZoneID", how="left")
    df.insert(0, "SourceDB", db_path.name)
    df = df[
        [
            "SourceDB",
            "ZoneID",
            "ZoneName",
            "Authority",
            f"Total{YEAR_COL}_Production",
            f"Total{YEAR_COL}_Attraction",
            f"Population_{YEAR_COL}",
            f"Workers_{YEAR_COL}",
            f"Households_{YEAR_COL}",
            f"Jobs_{YEAR_COL}",
        ]
    ]
    return df


def main() -> None:
    if not BASE_DIR.exists():
        raise SystemExit(f"Base directory not found: {BASE_DIR}")

    db_files = list_high_databases(BASE_DIR)
    if not db_files:
        raise SystemExit(f"No *_{scenario}.mdb files found in {BASE_DIR}")

    per_db_frames = []
    for db in db_files:
        try:
            df = read_filtered_aggregate(db)
            per_db_frames.append(df)
            print(f"Processed {db.name}: {len(df)} zone rows")
        except Exception as exc:
            print(f"Skipped {db.name} due to error: {exc}")

    if not per_db_frames:
        raise SystemExit("No database could be processed successfully.")

    combined_by_db = pd.concat(per_db_frames, ignore_index=True)
    # combined_by_db.to_csv(OUTPUT_BY_DB_CSV, index=False)

    # Final summary across all High-scenario databases
    summary = combined_by_db.groupby("ZoneID", as_index=False).agg(
        **{
            f"Total{YEAR_COL}_Production": (f"Total{YEAR_COL}_Production", "sum"),
            f"Total{YEAR_COL}_Attraction": (f"Total{YEAR_COL}_Attraction", "sum"),
            f"Population_{YEAR_COL}": (f"Population_{YEAR_COL}", "sum"),
            f"Workers_{YEAR_COL}": (f"Workers_{YEAR_COL}", "sum"),
            f"Households_{YEAR_COL}": (f"Households_{YEAR_COL}", "sum"),
            f"Jobs_{YEAR_COL}": (f"Jobs_{YEAR_COL}", "sum"),
        }
    )
    summary = summary.rename(
        columns={
            f"Total{YEAR_COL}_Production": f"Total{YEAR_COL}_Production_All{scenario}DBs",
            f"Total{YEAR_COL}_Attraction": f"Total{YEAR_COL}_Attraction_All{scenario}DBs",
            f"Population_{YEAR_COL}": f"Population_{YEAR_COL}_All{scenario}DBs",
            f"Workers_{YEAR_COL}": f"Workers_{YEAR_COL}_All{scenario}DBs",
            f"Households_{YEAR_COL}": f"Households_{YEAR_COL}_All{scenario}DBs",
            f"Jobs_{YEAR_COL}": f"Jobs_{YEAR_COL}_All{scenario}DBs",
        }
    )

    zone_meta = combined_by_db.drop_duplicates(subset="ZoneID")[
        ["ZoneID", "ZoneName", "Authority"]
    ]
    summary = summary.merge(zone_meta, on="ZoneID", how="left")
    summary = summary[
        [
            "ZoneID",
            "ZoneName",
            "Authority",
            f"Total{YEAR_COL}_Production_All{scenario}DBs",
            f"Total{YEAR_COL}_Attraction_All{scenario}DBs",
            f"Population_{YEAR_COL}_All{scenario}DBs",
            f"Workers_{YEAR_COL}_All{scenario}DBs",
            f"Households_{YEAR_COL}_All{scenario}DBs",
            f"Jobs_{YEAR_COL}_All{scenario}DBs",
        ]
    ].sort_values("ZoneID")
    summary.to_csv(OUTPUT_CSV, index=False)

    print()
    print(f"Wrote combined summary table:      {OUTPUT_CSV}")
    print(f"Summary rows: {len(summary)}")


if __name__ == "__main__":
    main()

# %%
