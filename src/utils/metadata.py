"""
src/utils/metadata.py
=====================
Parse farm / station / date metadata from track filenames.

One canonical anonymized filename convention is supported:

  FarmX format:
    FarmA_L1_S10_250729_093716_100005000023.csv
    Fields: farm=Farm{A|B}, location=L{N}, station=S{N}, date=YYMMDD, time=HHMMSS

Public API
----------
  parse_filename(filename)  -> dict or None
  parse_dataframe(df, filepath_col)  -> df with metadata columns added
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

_FARMX_RE = re.compile(
    r"^Farm(?P<farm_letter>[AB])_"
    r"L(?P<location>\d+)_"          # L1, L2, L3, L4 — independent locations
    r"S(?P<station>\d+)_"           # S10 → station='10'
    r"(?P<date>\d{6})_"             # YYMMDD
    r"(?P<time>\d{6})_"             # HHMMSS
    r"(?P<track_id>\d+)"
    r"\.csv$"
)

# ---------------------------------------------------------------------------
# Core parsing function
# ---------------------------------------------------------------------------

def parse_filename(filename: str | Path) -> Optional[dict]:
    """Parse a track filename and return a metadata dict, or None on no match.

    Farm identity
    -------------
    Farm A and Farm B are pseudonymous site labels.

    Returned dict keys
    ------------------
    farm_name   : str   'Farm_A' or 'Farm_B'
    farm_id     : str   anonymized site-location ID, e.g. FarmA_L1
    camera_unit : str   anonymized location-level identifier
    station_id  : str   e.g. 'S10' (as it appears in the filename)
    station_num : int   numeric station number
    date        : datetime.date
    date_str    : str   ISO date string YYYY-MM-DD
    time_str    : str   HHMMSS recording start time
    track_id    : str   raw numeric track identifier
    group_key   : str   '{farm_id}_{station_id}' — used as GroupKFold group label
    """
    name = Path(filename).name

    # --- FarmX_LY_SN canonical anonymized format ---
    m = _FARMX_RE.match(name)
    if m:
        raw_date = m.group("date")          # YYMMDD
        parsed_date = date(
            2000 + int(raw_date[:2]),
            int(raw_date[2:4]),
            int(raw_date[4:6]),
        )
        farm_letter  = m.group("farm_letter")       # 'A' or 'B'
        location_num = int(m.group("location"))      # 1, 2, 3, 4
        station_num  = int(m.group("station"))
        farm_name    = f"Farm_{farm_letter}"         # 'Farm_A' or 'Farm_B'
        farm_id      = f"Farm{farm_letter}_L{location_num}"  # e.g. 'FarmA_L1'
        station_id   = f"S{station_num}"
        return {
            "farm_name":    farm_name,
            "farm_id":      farm_id,
            "camera_unit":  farm_id,             # no separate camera unit in this format
            "location_num": location_num,
            "station_id":   station_id,
            "station_num":  station_num,
            "date":         parsed_date,
            "date_str":     parsed_date.isoformat(),
            "time_str":     m.group("time"),
            "track_id":     m.group("track_id"),
            "group_key":    f"{farm_id}_{station_id}",  # Farm+Location+Station
        }

    return None


# ---------------------------------------------------------------------------
# DataFrame helper
# ---------------------------------------------------------------------------

def parse_dataframe(df: pd.DataFrame, filepath_col: str = "filepath") -> pd.DataFrame:
    """Add metadata columns to a DataFrame that has a filepath column.

    Added columns: farm_id, station_id, station_num, date_str, time_str,
                   track_id, group_key, filename.

    Rows whose filename cannot be parsed are flagged with farm_id='UNKNOWN'
    and all other metadata fields as None. A warning is printed for each.
    """
    df = df.copy()

    meta_records = []
    for fp in df[filepath_col]:
        fname = Path(str(fp)).name
        meta = parse_filename(fname)
        if meta is None:
            import warnings
            warnings.warn(f"Could not parse metadata from filename: {fname!r}")
            meta_records.append(
                {
                    "farm_id": "UNKNOWN",
                    "station_id": None,
                    "station_num": None,
                    "date_str": None,
                    "time_str": None,
                    "track_id": None,
                    "group_key": "UNKNOWN",
                    "filename": fname,
                }
            )
        else:
            meta["filename"] = fname
            # date object → string for CSV friendliness; drop raw date object
            meta.pop("date", None)
            meta_records.append(meta)

    meta_df = pd.DataFrame(meta_records)

    # Reset index alignment before concat
    df = df.reset_index(drop=True)
    meta_df = meta_df.reset_index(drop=True)

    return pd.concat([df, meta_df], axis=1)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # Canonical anonymized format
        "FarmA_L1_S10_250101_093716_100005000023.csv",
        "FarmA_L2_S11_250101_110854_100079000000.csv",
        "FarmB_L1_S10_250201_092040_100001000000.csv",
        "unexpected_name.csv",
    ]
    for t in test_cases:
        result = parse_filename(t)
        print(f"{t}\n  → {result}\n")
