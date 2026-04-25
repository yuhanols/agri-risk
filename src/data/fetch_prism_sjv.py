"""
Fetch PRISM weather data for San Joaquin Valley only, then append to existing weather file.
"""

import json
import time
import csv
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "prism"
RAW.mkdir(parents=True, exist_ok=True)
PROC = ROOT / "data" / "processed"

LOC = {"lat": 36.7378, "lon": -119.7871, "elev": 94}
VARIABLES = ["tmax", "tmin", "ppt"]
START = "20100101"
END = "20260407"
API_URL = "https://prism.oregonstate.edu/explorer/dataexplorer/rpc.php"


def date_range_chunks(start_str, end_str, days=365):
    start = datetime.strptime(start_str, "%Y%m%d")
    end = datetime.strptime(end_str, "%Y%m%d")
    chunks = []
    while start < end:
        chunk_end = min(start + timedelta(days=days - 1), end)
        chunks.append((start.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        start = chunk_end + timedelta(days=1)
    return chunks


def fetch_prism(lat, lon, elev, variable, start, end):
    params = {
        "call": "pp/daily_timeseries", "proc": "gridserv", "spares": "4km",
        "interp": "idw", "stats": variable, "units": "si", "range": "daily",
        "start": start, "end": end, "lon": lon, "lat": lat, "elev": elev,
        "stability": "stable",
    }
    data = urlencode(params).encode()
    req = Request(API_URL, data=data, method="POST")
    resp = urlopen(req, timeout=120)
    result = json.loads(resp.read().decode())
    if result.get("errors"):
        raise ValueError(f"API error: {result['errors']}")
    values = result["result"]["data"][variable]
    s = datetime.strptime(start, "%Y%m%d")
    dates = [(s + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(values))]
    return list(zip(dates, values))


def main():
    loc_data = {}

    for var in VARIABLES:
        print(f"Fetching {var}...", end=" ", flush=True)
        chunks = date_range_chunks(START, END)
        var_rows = []
        for cs, ce in chunks:
            try:
                rows = fetch_prism(LOC["lat"], LOC["lon"], LOC["elev"], var, cs, ce)
                var_rows.extend(rows)
                print(f"[{cs[:4]}]", end=" ", flush=True)
                time.sleep(1)
            except Exception as e:
                print(f"\n  ERROR {cs}-{ce}: {e}")
                time.sleep(5)
        print(f" -> {len(var_rows)} days")

        # Save raw
        raw_path = RAW / f"prism_san_joaquin_ca_{var}.csv"
        with open(raw_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", var])
            w.writerows(var_rows)

        for date, val in var_rows:
            if date not in loc_data:
                loc_data[date] = {"location": "san_joaquin_ca", "district": "San Joaquin Valley", "date": date}
            loc_data[date][var] = val

    # Append to existing weather file
    weather_path = PROC / "weather_daily_2010_2026.csv"
    existing = []
    with open(weather_path, "r") as f:
        reader = csv.DictReader(f)
        existing = list(reader)

    new_rows = list(loc_data.values())
    all_rows = existing + new_rows
    all_rows.sort(key=lambda x: (x["location"], x["date"]))

    with open(weather_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["location", "district", "date", "tmax", "tmin", "ppt"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nAppended {len(new_rows)} rows to {weather_path}")
    print(f"Total rows now: {len(all_rows)}")


if __name__ == "__main__":
    main()
