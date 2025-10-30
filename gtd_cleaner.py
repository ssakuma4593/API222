from __future__ import annotations

import argparse
from typing import Iterable, List, Optional, Set

import numpy as np
import pandas as pd


# -----------------------------
# Data loading
# -----------------------------

def load_gtd_file(path: str) -> pd.DataFrame:
    """Load the Global Terrorism Database file (CSV or Excel) using pandas.

    - CSV: uses read_csv with low_memory=False
    - Excel (.xlsx/.xls): uses read_excel (requires openpyxl for .xlsx)
    """
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    # Fallback: try CSV first, then Excel
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_excel(path)


# -----------------------------
# Cleaning primitives
# -----------------------------

def clean_casualty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean nkill and nwound by treating -99/None/blank as missing, filling with 0, and casting to int.

    Returns a new DataFrame with cleaned columns.
    """
    result = df.copy()
    for col in ["nkill", "nwound"]:
        if col in result.columns:
            # Coerce to numeric and consider special codes as missing
            result[col] = (
                pd.to_numeric(result[col], errors="coerce")
                .replace({-99: np.nan})
                .fillna(0)
                .astype(int)
            )
    return result


def ensure_binary_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ensure specified binary indicator columns are strictly 0 or 1 (integers).

    Non-0/1 and missing values become 0.
    """
    result = df.copy()
    for col in columns:
        if col in result.columns:
            # Map common values and coerce invalid/missing to 0
            series = pd.to_numeric(result[col], errors="coerce").fillna(0)
            series = series.where(series == 1, 0)
            result[col] = series.astype(int)
    return result


def clean_geo_columns(
    df: pd.DataFrame,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    drop_missing_geo: bool = False,
) -> pd.DataFrame:
    """Replace -99 and invalid values with NaN in latitude/longitude.

    Adds a boolean flag column `geo_missing` and optionally drops rows with missing geo.
    """
    result = df.copy()
    for col in [latitude_col, longitude_col]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").replace({-99: np.nan})

    # Flag rows with missing (either lat or lon missing)
    if latitude_col in result.columns and longitude_col in result.columns:
        result["geo_missing"] = result[latitude_col].isna() | result[longitude_col].isna()
    else:
        result["geo_missing"] = True

    if drop_missing_geo:
        result = result[~result["geo_missing"]].copy()
    return result


def one_hot_encode(
    df: pd.DataFrame,
    categorical_columns: Iterable[str],
    drop_first: bool = False,
    prefix_sep: str = "__",
) -> pd.DataFrame:
    """Apply one-hot encoding to specified categorical columns.

    Uses pandas.get_dummies, does not create a column for NaN (dummy_na=False).
    """
    present_cols = [c for c in categorical_columns if c in df.columns]
    if not present_cols:
        return df.copy()
    return pd.get_dummies(df, columns=present_cols, drop_first=drop_first, dummy_na=False, prefix_sep=prefix_sep)


def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.5,
    required_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Drop columns with missing-value ratio strictly greater than threshold, except required columns.

    threshold is a fraction in [0,1].
    """
    result = df.copy()
    required: Set[str] = set(required_columns or [])
    missing_ratio = result.isna().mean()
    to_drop = [col for col, ratio in missing_ratio.items() if ratio > threshold and col not in required]
    if to_drop:
        result = result.drop(columns=to_drop)
    return result


def remove_duplicates_by_eventid(df: pd.DataFrame, eventid_col: str = "eventid") -> pd.DataFrame:
    """Drop duplicate events based on eventid, keeping the first occurrence."""
    if eventid_col in df.columns:
        return df.drop_duplicates(subset=[eventid_col], keep="first").copy()
    return df.copy()


def add_severity_feature(df: pd.DataFrame, kill_col: str = "nkill", wound_col: str = "nwound") -> pd.DataFrame:
    """Add severity feature as the sum of nkill and nwound."""
    result = df.copy()
    if kill_col in result.columns and wound_col in result.columns:
        # Ensure numeric
        kills = pd.to_numeric(result[kill_col], errors="coerce").fillna(0)
        wounds = pd.to_numeric(result[wound_col], errors="coerce").fillna(0)
        result["severity"] = kills + wounds
    else:
        result["severity"] = np.nan
    return result


# -----------------------------
# High-level cleaning pipeline
# -----------------------------

DEFAULT_CATEGORICAL_COLS: List[str] = [
    "attacktype1_txt",
    "weaptype1_txt",
    "region_txt",
    "country_txt",
    "targtype1_txt",
]

DEFAULT_BINARY_COLS: List[str] = [
    "success",
    "suicide",
]


def clean_gtd(
    csv_path: str,
    *,
    categorical_columns: Optional[Iterable[str]] = None,
    binary_columns: Optional[Iterable[str]] = None,
    required_columns_for_missing_drop: Optional[Iterable[str]] = None,
    drop_missing_geo: bool = False,
    drop_first_for_dummies: bool = False,
) -> pd.DataFrame:
    """Load and clean the GTD CSV for machine learning.

    Steps:
      1) Load CSV
      2) Clean casualties (nkill, nwound)
      3) Ensure binary columns (success, suicide)
      4) Clean geo columns (latitude, longitude); add geo_missing; optional drop
      5) Remove duplicates by eventid
      6) Drop columns with >50% missing (except required)
      7) One-hot encode selected categorical columns
      8) Add severity feature
    """
    df = load_gtd_file(csv_path)

    # 2) casualties
    df = clean_casualty_columns(df)

    # 3) binaries
    df = ensure_binary_columns(df, columns=list(binary_columns or DEFAULT_BINARY_COLS))

    # 4) geo
    df = clean_geo_columns(df, latitude_col="latitude", longitude_col="longitude", drop_missing_geo=drop_missing_geo)

    # 5) dedupe
    df = remove_duplicates_by_eventid(df, eventid_col="eventid")

    # 6) drop high-missing columns
    # Keep core analytical columns by default
    default_required = {
        "eventid",
        "iyear",
        "imonth",
        "iday",
        "country_txt",
        "region_txt",
        "attacktype1_txt",
        "targtype1_txt",
        "weaptype1_txt",
        "nkill",
        "nwound",
        "success",
        "suicide",
        "latitude",
        "longitude",
    }
    if required_columns_for_missing_drop:
        default_required.update(required_columns_for_missing_drop)
    df = drop_high_missing_columns(df, threshold=0.5, required_columns=default_required)

    # 7) one-hot encoding for selected categoricals
    df = one_hot_encode(
        df,
        categorical_columns=list(categorical_columns or DEFAULT_CATEGORICAL_COLS),
        drop_first=drop_first_for_dummies,
        prefix_sep="__",
    )

    # 8) severity
    df = add_severity_feature(df, kill_col="nkill", wound_col="nwound")

    return df


# -----------------------------
# CLI
# -----------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean GTD CSV for machine learning")
    p.add_argument("csv_path", help="Path to GTD CSV (input)")
    p.add_argument(
        "--output",
        "-o",
        help="Optional output path for cleaned CSV. If omitted, prints info only.",
        default=None,
    )
    p.add_argument(
        "--drop-missing-geo",
        action="store_true",
        help="Drop rows where latitude/longitude is missing after cleaning.",
    )
    p.add_argument(
        "--drop-first",
        action="store_true",
        help="Use drop_first=True for one-hot encoding to reduce multicollinearity.",
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cleaned = clean_gtd(
        args.csv_path,
        drop_missing_geo=bool(args.drop_missing_geo),
        drop_first_for_dummies=bool(args.drop_first),
    )

    if args.output:
        cleaned.to_csv(args.output, index=False)
        print(f"Saved cleaned CSV to: {args.output}")
    else:
        print("Cleaned DataFrame shape:", cleaned.shape)
        print("Columns (sample):", list(cleaned.columns)[:20])


if __name__ == "__main__":
    main()



