"""
Fetch PRISM daily weather data for lettuce production regions via Explorer RPC API.
Variables: tmax, tmin, ppt (daily)
Period: 2010-01-01 to 2026-04-07
Locations: Yuma AZ, Salinas CA, Imperial Valley CA, Oxnard CA, Santa Maria CA

Output: data/raw/prism_{location}_{variable}.csv per location+variable
        data/processed/weather_daily_2010_2026.csv (merged)

Note: PRISM API returns data in JSON. We request in ~1-year chunks to stay within limits.
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

LOCATIONS = {
    "yuma_az":       {"lat": 32.6927, "lon": -114.6277, "elev": 42,  "district": "Western Arizona"},
    "salinas_ca":    {"lat": 36.6744, "lon": -121.6550, "elev": 16,  "district": "Salinas-Watsonville"},
    "imperial_ca":   {"lat": 32.8421, "lon": -115.5694, "elev": -18, "district": "Imperial Valley"},
    "oxnard_ca":     {"lat": 34.1975, "lon": -119.1771, "elev": 13,  "district": "Oxnard"},
    "santa_maria_ca":{"lat": 34.9530, "lon": -120.4357, "elev": 66,  "district": "Santa Maria"},
    "san_joaquin_ca":{"lat": 36.7378, "lon": -119.7871, "elev": 94,  "district": "San Joaquin Valley"},
}

VARIABLES = ["tmax", "tmin", "ppt"]
START = "20100101"
END = "20260407"
API_URL = "https://prism.oregonstate.edu/explorer/dataexplorer/rpc.php"


def date_range_chunks(start_str, end_str, days=365):
    """Split date range into chunks of `days` length."""
    start = datetime.strptime(start_str, "%Y%m%d")
    end = datetime.strptime(end_str, "%Y%m%d")
    chunks = []
    while start < end:
        chunk_end = min(start + timedelta(days=days - 1), end)
        chunks.append((start.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        start = chunk_end + timedelta(days=1)
    return chunks


def fetch_prism(lat, lon, elev, variable, start, end):
    """Fetch PRISM daily data for one location/variable/date range."""
    params = {
        "call": "pp/daily_timeseries",
        "proc": "gridserv",
        "spares": "4km",
        "interp": "idw",
        "stats": variable,
        "units": "si",
        "range": "daily",
        "start": start,
        "end": end,
        "lon": lon,
        "lat": lat,
        "elev": elev,
        "stability": "stable",
    }
    data = urlencode(params).encode()
    req = Request(API_URL, data=data, method="POST")
    resp = urlopen(req, timeout=120)
    result = json.loads(resp.read().decode())

    if result.get("errors"):
        raise ValueError(f"API error: {result['errors']}")

    values = result["result"]["data"][variable]

    # Generate date list
    s = datetime.strptime(start, "%Y%m%d")
    dates = [(s + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(values))]
    return list(zip(dates, values))


def main():
    all_data = []

    for loc_name, loc_info in LOCATIONS.items():
        print(f"\n=== {loc_name} ===")
        loc_data = {}  # date -> {tmax, tmin, ppt}

        for var in VARIABLES:
            print(f"  Fetching {var}...", end=" ", flush=True)
            chunks = date_range_chunks(START, END)
            var_rows = []

            for i, (cs, ce) in enumerate(chunks):
                try:
                    rows = fetch_prism(loc_info["lat"], loc_info["lon"], loc_info["elev"], var, cs, ce)
                    var_rows.extend(rows)
                    print(f"[{cs[:4]}]", end=" ", flush=True)
                    time.sleep(1)  # be polite
                except Exception as e:
                    print(f"\n    ERROR chunk {cs}-{ce}: {e}")
                    time.sleep(5)

            print(f"  -> {len(var_rows)} days")

            # Save raw per location+variable
            raw_path = RAW / f"prism_{loc_name}_{var}.csv"
            with open(raw_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", var])
                w.writerows(var_rows)

            for date, val in var_rows:
                if date not in loc_data:
                    loc_data[date] = {"location": loc_name, "district": loc_info["district"], "date": date}
                loc_data[date][var] = val

        for row in loc_data.values():
            all_data.append(row)

    # Save merged
    out_path = ROOT / "data" / "processed" / "weather_daily_2010_2026.csv"
    all_data.sort(key=lambda x: (x["location"], x["date"]))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["location", "district", "date", "tmax", "tmin", "ppt"])
        w.writeheader()
        w.writerows(all_data)

    print(f"\n=== Done ===")
    print(f"Saved: {out_path}")
    print(f"Total rows: {len(all_data)}")


if __name__ == "__main__":
    main()
