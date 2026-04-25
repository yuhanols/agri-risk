"""
Merge all FOB shipping point price CSVs (2010-2026) into one processed file.
Only keeps lettuce rows for AZ and CA districts.
Output: data/processed/price_lettuce_daily_2010_2026.csv
"""

import csv
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
PRICE_DIR = ROOT / "data" / "raw" / "price"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Also include the earlier downloads
EXTRA_FILES = [
    Path("data/raw/price/AMS_sc_shippingpt_daily_20260422-004557.csv"),  # 2010
    Path("data/raw/price/AMS_sc_shippingpt_daily_20260422-004531.csv"),  # 2026 Mar-Apr
    Path("data/raw/price/AMS_sc_shippingpt_daily_20260422-003353.csv"),  # 2026 Apr 16-17
]

# Districts we care about (AZ + CA)
KEEP_DISTRICTS = {
    "WESTERN ARIZONA",
    "SALINAS-WATSONVILLE CALIFORNIA",
    "SANTA MARIA CALIFORNIA",
    "SOUTH AND CENTRAL DISTRICT CALIFORNIA",
    "IMPERIAL VALLEY CALIFORNIA",
    "IMPERIAL, COACHELLA AND PALO VERDE VALLEYS CALIFORNIA",
    "IMPERIAL AND COACHELLA VALLEYS CALIFORNIA",
    "IMPERIAL AND PALO VERDE VALLEYS CALIFORNIA",
    "OXNARD DISTRICT CALIFORNIA",
    "CENTRAL SAN JOAQUIN VALLEY CALIFORNIA",
    "SAN JOAQUIN VALLEY CALIFORNIA",
    "KERN DISTRICT CALIFORNIA",
    "COACHELLA VALLEY CALIFORNIA",
}

OUT_COLS = [
    "report_date", "district", "commodity", "variety", "package", "item_size",
    "organic", "low_price", "high_price", "mostly_low_price", "mostly_high_price",
    "market_tone_comments", "supply_tone_comments", "demand_tone_comments",
    "commodity_comments",
]


def parse_date(d):
    """Parse MM/DD/YYYY to YYYY-MM-DD."""
    try:
        return datetime.strptime(d.strip(), "%m/%d/%Y").strftime("%Y-%m-%d")
    except:
        return None


def load_lettuce(filepath):
    rows = []
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if "Lettuce" not in r.get("commodity", ""):
                continue
            if r.get("district", "").strip() not in KEEP_DISTRICTS:
                continue
            date = parse_date(r.get("report_date", ""))
            if not date:
                continue
            row = {
                "report_date": date,
                "district": r["district"].strip(),
                "commodity": r["commodity"].strip(),
                "variety": r.get("variety", "N/A").strip(),
                "package": r.get("package", "").strip(),
                "item_size": r.get("item_size", "").strip(),
                "organic": r.get("organic", "N").strip(),
                "low_price": r.get("low_price", "").strip(),
                "high_price": r.get("high_price", "").strip(),
                "mostly_low_price": r.get("mostly_low_price", "").strip(),
                "mostly_high_price": r.get("mostly_high_price", "").strip(),
                "market_tone_comments": r.get("market_tone_comments", "").strip(),
                "supply_tone_comments": r.get("supply_tone_comments", "").strip(),
                "demand_tone_comments": r.get("demand_tone_comments", "").strip(),
                "commodity_comments": r.get("commodity_comments", "").strip(),
            }
            rows.append(row)
    return rows


def main():
    all_rows = []

    # Load yearly files
    for year in range(2010, 2027):
        fp = PRICE_DIR / f"{year}.csv"
        if fp.exists():
            rows = load_lettuce(fp)
            print(f"{year}: {len(rows)} lettuce rows (AZ+CA)")
            all_rows.extend(rows)

    # Load extra files
    for fp in EXTRA_FILES:
        if fp.exists():
            rows = load_lettuce(fp)
            print(f"{fp.name}: {len(rows)} lettuce rows (AZ+CA)")
            all_rows.extend(rows)

    # Deduplicate
    seen = set()
    unique_rows = []
    for r in all_rows:
        key = (r["report_date"], r["district"], r["commodity"], r["variety"],
               r["package"], r["item_size"], r["organic"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    unique_rows.sort(key=lambda x: (x["report_date"], x["district"], x["commodity"]))

    # Save
    outpath = OUT / "price_lettuce_daily_2010_2026.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(unique_rows)

    # Summary
    print(f"\n=== Saved: {outpath} ===")
    print(f"Total rows: {len(unique_rows)}")

    dates = sorted(set(r["report_date"] for r in unique_rows))
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Unique dates: {len(dates)}")

    from collections import Counter
    print(f"\nDistricts:")
    for d, c in Counter(r["district"] for r in unique_rows).most_common():
        print(f"  {d}: {c}")

    print(f"\nCommodities:")
    for d, c in Counter(r["commodity"] for r in unique_rows).most_common():
        print(f"  {d}: {c}")

    # Price summary by commodity (non-organic, mostly price)
    from collections import defaultdict
    prices = defaultdict(list)
    for r in unique_rows:
        if r["organic"] == "Y":
            continue
        ml, mh = r["mostly_low_price"], r["mostly_high_price"]
        if ml and mh:
            try:
                mid = (float(ml) + float(mh)) / 2
                prices[r["commodity"]].append((r["report_date"][:4], mid))
            except:
                pass

    print(f"\nAvg FOB price (mostly midpoint, non-organic) by commodity and period:")
    for comm in sorted(prices.keys()):
        early = [p for y, p in prices[comm] if y <= "2015"]
        late = [p for y, p in prices[comm] if y >= "2020"]
        e_avg = sum(early) / len(early) if early else 0
        l_avg = sum(late) / len(late) if late else 0
        print(f"  {comm}: 2010-15 ${e_avg:.2f} -> 2020-26 ${l_avg:.2f} ({(l_avg/e_avg-1)*100:+.0f}%)" if early and late else f"  {comm}: insufficient data")


if __name__ == "__main__":
    main()
