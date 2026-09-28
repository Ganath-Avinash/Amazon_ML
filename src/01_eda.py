"""
Phase 1 — Exploratory Data Analysis
Amazon ML Challenge 2026: Business Entity Resolution

Run from repo root:
    python src/01_eda.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import polars as pl

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("Dataset/student_resource/dataset")
TRAIN = BASE / "train"
TEST  = BASE / "test"
OUT   = Path("output")
OUT.mkdir(exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def load(path: Path) -> pl.DataFrame:
    print(f"  Loading {path.name} …")
    return pl.read_csv(path, separator="\t", infer_schema_length=0, ignore_errors=True)

def null_rate(df: pl.DataFrame) -> dict:
    total = len(df)
    return {col: round(df[col].null_count() / total * 100, 2) for col in df.columns}

def has_question_marks(df: pl.DataFrame, col: str) -> int:
    """Count rows where a field contains '???' (encoding corruption)."""
    return df.filter(pl.col(col).str.contains(r"\?\?\?")).height

def detect_encoding_issues(df: pl.DataFrame) -> dict:
    issues = {}
    for col in ["business_name", "business_address"]:
        if col in df.columns:
            issues[col] = has_question_marks(df, col)
    return issues

def top_values(series: pl.Series, n: int = 10) -> list:
    counts = series.drop_nulls().value_counts().sort("count", descending=True).head(n)
    return [(row[0], row[1]) for row in counts.iter_rows()]

def match_count_stats(gt: pl.DataFrame) -> dict:
    """Analyse the ground truth match distribution."""
    def count_matches(ids_str):
        if ids_str is None or ids_str == "":
            return 0
        return len(ids_str.split(","))

    gt = gt.with_columns(
        pl.col("matched_entity_ids")
          .map_elements(count_matches, return_dtype=pl.Int32)
          .alias("match_count")
    )
    mc = gt["match_count"]
    singletons = (mc == 0).sum()

    return {
        "total_s1_entities": len(gt),
        "singletons_count": int(singletons),
        "singletons_pct": round(float(singletons) / len(gt) * 100, 2),
        "mean_matches": round(float(mc.mean()), 3),
        "median_matches": float(mc.median()),
        "max_matches": int(mc.max()),
        "min_matches": int(mc.min()),
        "p25": float(mc.quantile(0.25)),
        "p75": float(mc.quantile(0.75)),
        "p95": float(mc.quantile(0.95)),
        "match_count_distribution": dict(
            sorted(Counter(v for v in mc.to_list() if v is not None).items())
        ),
    }

def sample_noisy_pairs(s1: pl.DataFrame, gt: pl.DataFrame,
                       s2: pl.DataFrame, s3: pl.DataFrame,
                       n: int = 10) -> list:
    """Grab N matched pairs so we can see what noise looks like."""
    # pick entities that have at least one match
    matched_gt = gt.filter(
        pl.col("matched_entity_ids").is_not_null() &
        (pl.col("matched_entity_ids") != "")
    ).head(n * 3)

    s1_lookup = {row[0]: (row[1], row[2]) for row in
                 s1.select(["entity_id","business_name","business_address"]).iter_rows()}
    s2_lookup = {row[0]: (row[1], row[2]) for row in
                 s2.select(["entity_id","business_name","business_address"]).iter_rows()}
    s3_lookup = {row[0]: (row[1], row[2]) for row in
                 s3.select(["entity_id","business_name","business_address"]).iter_rows()}

    examples = []
    for row in matched_gt.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        if s1_id not in s1_lookup:
            continue
        s1_name, s1_addr = s1_lookup[s1_id]
        matched_ids = row["matched_entity_ids"].split(",")[:2]
        for mid in matched_ids:
            mid = mid.strip()
            lookup = s2_lookup if mid.startswith("S2") else s3_lookup
            if mid in lookup:
                cand_name, cand_addr = lookup[mid]
                examples.append({
                    "s1_id": s1_id, "s1_name": s1_name, "s1_addr": s1_addr,
                    "match_id": mid, "match_name": cand_name, "match_addr": cand_addr,
                })
        if len(examples) >= n:
            break
    return examples

def address_component_analysis(df: pl.DataFrame, label: str) -> dict:
    """Check how many records have ZIP codes, states, cities detectable."""
    zip_pattern = r"\b\d{5}(?:-\d{4})?\b|\b\d{6}\b"  # US ZIP or India PIN
    if "business_address" not in df.columns:
        return {}
    has_zip = df["business_address"].drop_nulls().map_elements(
        lambda x: bool(re.search(zip_pattern, x)), return_dtype=pl.Boolean
    ).sum()
    total_with_addr = df["business_address"].drop_nulls().len()
    return {
        "total_records": len(df),
        "records_with_address": int(total_with_addr),
        "address_null_pct": round((len(df) - total_with_addr) / len(df) * 100, 2),
        "records_with_zip_or_pin": int(has_zip),
        "zip_coverage_pct": round(float(has_zip) / total_with_addr * 100, 2) if total_with_addr else 0,
    }

# ─── Main EDA ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Amazon ML Challenge 2026 — Phase 1 EDA")
    print("=" * 60)

    report = {}

    # ── Load all files ─────────────────────────────────────────────
    print("\n[1/6] Loading datasets …")
    s1 = load(TRAIN / "train_source1.tsv")
    s2 = load(TRAIN / "train_source2.tsv")
    s3 = load(TRAIN / "train_source3.tsv")
    gt = load(TRAIN / "train_ground_truth.tsv")
    ts1 = load(TEST / "test_source1.tsv")
    ts2 = load(TEST / "test_source2.tsv")
    ts3 = load(TEST / "test_source3.tsv")

    # ── Basic shape ────────────────────────────────────────────────
    print("\n[2/6] Shape & schema …")
    report["shapes"] = {
        "train_source1": {"rows": len(s1),  "cols": len(s1.columns),  "columns": s1.columns},
        "train_source2": {"rows": len(s2),  "cols": len(s2.columns),  "columns": s2.columns},
        "train_source3": {"rows": len(s3),  "cols": len(s3.columns),  "columns": s3.columns},
        "train_gt":      {"rows": len(gt),  "cols": len(gt.columns),  "columns": gt.columns},
        "test_source1":  {"rows": len(ts1), "cols": len(ts1.columns), "columns": ts1.columns},
        "test_source2":  {"rows": len(ts2), "cols": len(ts2.columns), "columns": ts2.columns},
        "test_source3":  {"rows": len(ts3), "cols": len(ts3.columns), "columns": ts3.columns},
    }

    # ── Null rates ─────────────────────────────────────────────────
    print("\n[3/6] Null rates …")
    report["null_rates"] = {
        "train_source1": null_rate(s1),
        "train_source2": null_rate(s2),
        "train_source3": null_rate(s3),
        "train_gt":      null_rate(gt),
        "test_source1":  null_rate(ts1),
    }

    # ── Country distribution ───────────────────────────────────────
    print("\n[4/6] Country distribution …")
    report["country_distribution"] = {
        "train_source1": top_values(s1["country"]),
        "train_source2": top_values(s2["country"]),
        "train_source3": top_values(s3["country"]),
        "test_source1":  top_values(ts1["country"]),
        "test_source2":  top_values(ts2["country"]),
        "test_source3":  top_values(ts3["country"]),
    }

    # ── Ground truth match distribution ───────────────────────────
    print("\n[5/6] Ground truth match statistics …")
    report["match_stats"] = match_count_stats(gt)

    # ── Encoding / noise analysis ──────────────────────────────────
    print("\n[6/6] Encoding issues & address coverage …")
    report["encoding_issues"] = {
        "train_source1": detect_encoding_issues(s1),
        "train_source2": detect_encoding_issues(s2),
        "train_source3": detect_encoding_issues(s3),
    }
    report["address_coverage"] = {
        "train_source1": address_component_analysis(s1, "S1"),
        "train_source2": address_component_analysis(s2, "S2"),
        "train_source3": address_component_analysis(s3, "S3"),
        "test_source1":  address_component_analysis(ts1, "TS1"),
    }

    # ── Sample noisy matched pairs ─────────────────────────────────
    print("\n  Sampling noisy matched pairs …")
    report["sample_matched_pairs"] = sample_noisy_pairs(s1, gt, s2, s3, n=10)

    # ── Save JSON ──────────────────────────────────────────────────
    json_path = OUT / "eda_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ EDA JSON saved → {json_path}")

    # ── Print summary to console ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print("\n📦 Dataset Shapes:")
    for k, v in report["shapes"].items():
        print(f"  {k:<20} {v['rows']:>10,} rows")

    print("\n🌍 Country Distribution (train_source1):")
    for country, cnt in report["country_distribution"]["train_source1"]:
        print(f"  {country:<20} {cnt:>10,}")

    print("\n🌍 Country Distribution (test_source1):")
    for country, cnt in report["country_distribution"]["test_source1"]:
        print(f"  {country:<20} {cnt:>10,}")

    ms = report["match_stats"]
    print(f"\n🎯 Ground Truth Match Stats:")
    print(f"  Total S1 entities  : {ms['total_s1_entities']:,}")
    print(f"  Singletons (0 match): {ms['singletons_count']:,} ({ms['singletons_pct']}%)")
    print(f"  Mean matches       : {ms['mean_matches']}")
    print(f"  Median matches     : {ms['median_matches']}")
    print(f"  Max matches        : {ms['max_matches']}")
    print(f"  p75 matches        : {ms['p75']}")
    print(f"  p95 matches        : {ms['p95']}")

    print("\n⚠️  Encoding Issues ('???' corruption):")
    for src, issues in report["encoding_issues"].items():
        for col, cnt in issues.items():
            print(f"  {src}.{col:<25} {cnt:>10,} rows affected")

    print("\n📍 Address Coverage:")
    for src, stats in report["address_coverage"].items():
        print(f"  {src:<20} addr_null={stats['address_null_pct']}%  "
              f"zip_coverage={stats['zip_coverage_pct']}%")

    print("\n📝 Sample Noisy Matched Pairs (first 3):")
    for ex in report["sample_matched_pairs"][:3]:
        print(f"\n  S1  [{ex['s1_id']}]")
        print(f"    name : {ex['s1_name']}")
        print(f"    addr : {ex['s1_addr']}")
        print(f"  ↔ [{ex['match_id']}]")
        print(f"    name : {ex['match_name']}")
        print(f"    addr : {ex['match_addr']}")

    print("\n" + "=" * 60)
    print("  EDA Complete. Full results → output/eda_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
