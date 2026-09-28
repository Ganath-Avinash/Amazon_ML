"""
Phase 3 (v3) — Blocking — FAST + HIGH RECALL
Amazon ML Challenge 2026

Key improvements over v2:
  - Keys pre-computed in batch using Polars (vectorized, not row-by-row Python)
  - Index built from pre-computed key columns — much faster
  - 9 blocking key types including address keys for Hindi/Tamil bridging
  - ZIP fix: excluded if it's the first address token

Run from repo root:
    python -X utf8 src/03_blocking_v3.py
"""

import csv
import json
import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from collections import defaultdict
from pathlib import Path

import polars as pl

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path("Dataset/student_resource/dataset")
TRAIN_NORM = BASE / "train" / "normalized"
TEST_NORM  = BASE / "test"  / "normalized"
TRAIN_RAW  = BASE / "train"
OUT        = Path("output")
OUT.mkdir(exist_ok=True)

MAX_CANDIDATES = 300
LOG_EVERY      = 500_000

# Generic tokens to exclude from address keys (too common to be useful)
SKIP_ADDR = {
    "road","street","avenue","blvd","drive","lane","court","place",
    "near","block","sector","phase","plot","null","none","na",
    "north","south","east","west","new","old","main","unit","suite",
    "india","united","states","building","floor","apartment","colony",
    "nagar","marg","vihar","enclave","park","market","complex","tower",
    "ca","ny","tx","fl","il","pa","oh","ga","nc","mi","nj","va","wa",
    "az","ma","tn","in","mo","md","wi","co","mn","sc","al","la","ky",
    "or","ok","ct","ut","ia","nv","ar","ms","ks","ne","nm","wv","id",
    "pradesh","maharashtra","karnataka","gujarat","rajasthan","bengal",
    "kerala","andhra","telangana","bihar","odisha","jharkhand","tamil",
}


# ── Key pre-computation using Polars expressions ──────────────────────────────

def add_key_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorized: compute all blocking key columns in one Polars pass.
    Much faster than row-by-row Python.
    """

    def first_n_tokens(col: str, n: int) -> pl.Expr:
        """Extract first N space-separated tokens from a string column."""
        expr = pl.col(col).str.splitn(" ", n + 1)
        return pl.concat_list([expr.list.get(i, null_on_oob=True) for i in range(n)])

    # Pre-split norm_name tokens (polars series of lists)
    df = df.with_columns([
        pl.col("norm_name").str.split(" ").alias("_ntoks"),
        pl.col("norm_address").str.split(" ").alias("_atoks"),
        pl.col("country").str.to_lowercase().str.strip_chars().alias("_cty"),
    ])

    # Name token extractions
    df = df.with_columns([
        pl.col("_ntoks").list.get(0, null_on_oob=True).alias("_n0"),
        pl.col("_ntoks").list.get(1, null_on_oob=True).alias("_n1"),
        pl.col("_ntoks").list.get(2, null_on_oob=True).alias("_n2"),
        pl.col("_ntoks").list.get(-1, null_on_oob=True).alias("_nlast"),
        pl.col("_ntoks").list.get(-2, null_on_oob=True).alias("_n2last"),
        pl.col("_ntoks").list.len().alias("_nlen"),
    ])

    # Address token extractions (positions 1,2,3,4 and -2,-3 for mid/end keys)
    df = df.with_columns([
        pl.col("_atoks").list.get(0, null_on_oob=True).alias("_a0"),   # house#
        pl.col("_atoks").list.get(1, null_on_oob=True).alias("_a1"),
        pl.col("_atoks").list.get(2, null_on_oob=True).alias("_a2"),
        pl.col("_atoks").list.get(3, null_on_oob=True).alias("_a3"),
        pl.col("_atoks").list.get(-2, null_on_oob=True).alias("_alast2"),
        pl.col("_atoks").list.get(-3, null_on_oob=True).alias("_alast3"),
        pl.col("_atoks").list.len().alias("_alen"),
    ])

    # ZIP validity: valid only if zip_pin != first address token
    df = df.with_columns([
        pl.when(
            pl.col("zip_pin").is_not_null() &
            (pl.col("zip_pin") != "") &
            (pl.col("zip_pin") != pl.col("_a0"))
        ).then(pl.col("zip_pin")).otherwise(None).alias("_zip_valid")
    ])

    return df


def collect_keys_from_row(row: dict) -> list:
    """
    Given a pre-computed row (with _n0, _n1, etc.), emit all blocking key tuples.
    This runs on the already-computed columns — much faster than computing in Python.
    """
    keys = []
    cty  = row["_cty"] or ""
    n0, n1, n2 = row["_n0"], row["_n1"], row["_n2"]
    nlast, n2last = row["_nlast"], row["_n2last"]
    nlen = row["_nlen"] or 0
    a1, a2, a3 = row["_a1"], row["_a2"], row["_a3"]
    alast2, alast3 = row["_alast2"], row["_alast3"]
    alen = row["_alen"] or 0
    zip_valid = row["_zip_valid"]

    # ── Name keys ─────────────────────────────────────────────────
    # K1: first 2 name tokens
    if n0 and n1 and len(n0) >= 2 and len(n1) >= 2:
        keys.append(("n2", cty, n0, n1))

    # K2: first token only (ALL names — critical for typo bridging)
    if n0 and len(n0) >= 2:
        keys.append(("n1all", cty, n0))

    # K3: last 2 tokens (transpositions)
    if nlen >= 3 and n2last and nlast and len(n2last) >= 2 and len(nlast) >= 2:
        keys.append(("sfx2", cty, n2last, nlast))

    # K4: first + last token (middle-word drops)
    if nlen >= 3 and n0 and nlast and len(n0) >= 2 and len(nlast) >= 2:
        keys.append(("fl", cty, n0, nlast))

    # K5: first 3 tokens (high precision)
    if n0 and n1 and n2 and len(n0) >= 2 and len(n1) >= 2 and len(n2) >= 2:
        keys.append(("n3", cty, n0, n1, n2))

    # ── ZIP key ───────────────────────────────────────────────────
    # K6: country + valid ZIP/PIN
    if zip_valid:
        keys.append(("zip", cty, zip_valid))

    # ── Address keys (critical for Hindi/Tamil ↔ English matching) ─
    # These bridge cases where name tokens differ (transliteration)
    # but address is identical.

    # K7: addr tokens [1,2] (street/locality — skip house# at [0])
    if a1 and a2 and len(a1) >= 3 and len(a2) >= 3 \
       and a1 not in SKIP_ADDR and a2 not in SKIP_ADDR:
        keys.append(("addr_12", cty, a1, a2))

    # K8: addr tokens [2,3] (deeper locality)
    if a2 and a3 and len(a2) >= 3 and len(a3) >= 3 \
       and a2 not in SKIP_ADDR and a3 not in SKIP_ADDR:
        keys.append(("addr_23", cty, a2, a3))

    # K9: end addr tokens [-3,-2] (city-area, often distinctive)
    if alast3 and alast2 and len(alast3) >= 3 and len(alast2) >= 3 \
       and alast3 not in SKIP_ADDR and alast2 not in SKIP_ADDR \
       and alen >= 5:  # only for longer addresses (avoid noise)
        keys.append(("addr_end", cty, alast3, alast2))

    return keys


# ── Build inverted index ──────────────────────────────────────────────────────

def build_index(s2_path: Path, s3_path: Path) -> defaultdict:
    index = defaultdict(list)
    grand_total = 0

    for path, label in [(s2_path, "S2"), (s3_path, "S3")]:
        print(f"\n  Indexing {label} — {path.name}")
        t0 = time.time()

        df = pl.read_csv(
            path, separator="\t", infer_schema_length=0, ignore_errors=True,
            columns=["entity_id", "country", "norm_name", "norm_address", "zip_pin"]
        )
        df = add_key_columns(df)

        # Keep only columns needed for iteration
        keep = ["entity_id", "_cty", "_n0","_n1","_n2","_nlast","_n2last","_nlen",
                "_a1","_a2","_a3","_alast2","_alast3","_alen","_zip_valid"]
        df = df.select([c for c in keep if c in df.columns])

        count = 0
        for row in df.iter_rows(named=True):
            eid  = row["entity_id"]
            keys = collect_keys_from_row(row)
            for k in keys:
                index[k].append(eid)
            count += 1
            if count % LOG_EVERY == 0:
                print(f"    {label}: {count:,} rows  ({time.time()-t0:.0f}s)")

        grand_total += count
        print(f"  {label}: {count:,} rows in {time.time()-t0:.1f}s")

    n_keys   = len(index)
    n_values = sum(len(v) for v in index.values())
    print(f"\n  Index: {n_keys:,} unique keys | {n_values:,} total entries | "
          f"{grand_total:,} records indexed")
    return index


# ── Generate candidates ───────────────────────────────────────────────────────

def generate_candidates(s1_path: Path, index: defaultdict) -> dict:
    print(f"\n  Reading S1 — {s1_path.name}")
    t0 = time.time()

    df = pl.read_csv(
        s1_path, separator="\t", infer_schema_length=0, ignore_errors=True,
        columns=["entity_id", "country", "norm_name", "norm_address", "zip_pin"]
    )
    df = add_key_columns(df)
    keep = ["entity_id", "_cty", "_n0","_n1","_n2","_nlast","_n2last","_nlen",
            "_a1","_a2","_a3","_alast2","_alast3","_alen","_zip_valid"]
    df = df.select([c for c in keep if c in df.columns])

    candidates  = {}
    count       = 0
    total_cands = 0
    zero_cand   = 0
    capped      = 0

    for row in df.iter_rows(named=True):
        s1_id = row["entity_id"]
        keys  = collect_keys_from_row(row)

        cand_set = set()
        for k in keys:
            if k in index:
                cand_set.update(index[k])
        cand_set.discard(s1_id)

        if len(cand_set) > MAX_CANDIDATES:
            cand_list = list(cand_set)[:MAX_CANDIDATES]
            capped += 1
        else:
            cand_list = list(cand_set)

        candidates[s1_id] = cand_list
        total_cands += len(cand_list)
        if not cand_list:
            zero_cand += 1
        count += 1

        if count % 200_000 == 0:
            avg = total_cands / count
            print(f"    {count:,} S1 | avg: {avg:.1f} | zero: {zero_cand:,} | "
                  f"capped: {capped:,} | {time.time()-t0:.0f}s")

    avg = total_cands / count if count else 0
    print(f"  Done: {count:,} S1 in {time.time()-t0:.1f}s | "
          f"avg {avg:.1f} cands | {zero_cand:,} zero | {capped:,} capped")
    return candidates


# ── Measure recall ────────────────────────────────────────────────────────────

def measure_recall(candidates: dict, gt_path: Path) -> dict:
    print("  Loading ground truth ...")
    gt = pl.read_csv(gt_path, separator="\t", infer_schema_length=0, ignore_errors=True)

    total_true = total_found = n_miss_ents = zero_c_miss = 0
    missed_egs = []

    for row in gt.iter_rows(named=True):
        s1_id       = row["source1_entity_id"]
        matched_str = row.get("matched_entity_ids", "") or ""
        if not matched_str.strip():
            continue

        true_set = set(m.strip() for m in matched_str.split(",") if m.strip())
        cand_set = set(candidates.get(s1_id, []))
        found    = true_set & cand_set
        missed   = true_set - cand_set

        total_true  += len(true_set)
        total_found += len(found)

        if missed:
            n_miss_ents += 1
            if not cand_set:
                zero_c_miss += 1
            if len(missed_egs) < 10:
                missed_egs.append({
                    "s1_id":    s1_id,
                    "n_true":   len(true_set),
                    "n_found":  len(found),
                    "n_missed": len(missed),
                    "n_cands":  len(cand_set),
                    "missed":   list(missed)[:3],
                })

    recall = total_found / total_true if total_true > 0 else 0.0
    return {
        "total_true_matches":    total_true,
        "found_in_candidates":   total_found,
        "missed":                total_true - total_found,
        "blocking_recall":       round(recall, 6),
        "n_entities_with_miss":  n_miss_ents,
        "zero_cand_misses":      zero_c_miss,
        "sample_misses":         missed_egs,
    }


# ── Save TSV ──────────────────────────────────────────────────────────────────

def save_candidates(candidates: dict, out_path: Path):
    t0 = time.time()
    print(f"  Writing {out_path} ...")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id, cands in candidates.items():
            writer.writerow([s1_id, ",".join(cands)])
    mb = out_path.stat().st_size / 1e6
    print(f"  Saved {mb:.1f} MB in {time.time()-t0:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Amazon ML Challenge 2026 - Phase 3 v3: Blocking (Fast)")
    print("=" * 65)
    print(f"\n  Config: MAX_CANDIDATES={MAX_CANDIDATES}")
    print(f"  Keys (9): n2, n1all, sfx2, fl, n3, zip, addr_12, addr_23, addr_end")
    print(f"  Engine: Polars vectorized key pre-computation\n")

    wall_start = time.time()

    # ═══ TRAIN ═══════════════════════════════════════════════════
    print("-" * 65)
    print("  TRAIN BLOCKING")
    print("-" * 65)

    print("\n[1/4] Building train index (S2 + S3) ...")
    train_index = build_index(
        TRAIN_NORM / "norm_source2.tsv",
        TRAIN_NORM / "norm_source3.tsv",
    )

    print("\n[2/4] Generating candidates for train S1 ...")
    train_cands = generate_candidates(TRAIN_NORM / "norm_source1.tsv", train_index)

    print("\n[3/4] Measuring blocking recall ...")
    recall = measure_recall(train_cands, TRAIN_RAW / "train_ground_truth.tsv")

    print("\n[4/4] Saving train_candidate_pairs.tsv ...")
    save_candidates(train_cands, OUT / "train_candidate_pairs.tsv")
    del train_index

    # ═══ TEST ════════════════════════════════════════════════════
    print("\n" + "-" * 65)
    print("  TEST BLOCKING")
    print("-" * 65)

    print("\n[1/2] Building test index (S2 + S3) ...")
    test_index = build_index(
        TEST_NORM / "norm_source2.tsv",
        TEST_NORM / "norm_source3.tsv",
    )

    print("\n[2/2] Generating candidates for test S1 ...")
    test_cands = generate_candidates(TEST_NORM / "norm_source1.tsv", test_index)

    print("\nSaving candidate_pairs.tsv ...")
    save_candidates(test_cands, OUT / "candidate_pairs.tsv")
    del test_index

    # ═══ SUMMARY ═════════════════════════════════════════════════
    wall_total  = time.time() - wall_start
    train_total = sum(len(v) for v in train_cands.values())
    train_avg   = train_total / len(train_cands) if train_cands else 0
    train_zero  = sum(1 for v in train_cands.values() if not v)
    test_total  = sum(len(v) for v in test_cands.values())
    test_avg    = test_total / len(test_cands) if test_cands else 0

    bkts = {"0": 0, "1-10": 0, "11-50": 0, "51-100": 0,
            "101-200": 0, "201-300": 0, "300+(cap)": 0}
    for cands in train_cands.values():
        n = len(cands)
        if   n == 0:    bkts["0"] += 1
        elif n <= 10:   bkts["1-10"] += 1
        elif n <= 50:   bkts["11-50"] += 1
        elif n <= 100:  bkts["51-100"] += 1
        elif n <= 200:  bkts["101-200"] += 1
        elif n <= 300:  bkts["201-300"] += 1
        else:           bkts["300+(cap)"] += 1

    r = recall
    print(f"""
{"="*65}
  PHASE 3 v3 COMPLETE - SUMMARY
{"="*65}

  TRAIN CANDIDATES
  -----------------------------------------------
  S1 entities              : {len(train_cands):>12,}
  Total candidate pairs    : {train_total:>12,}
  Avg candidates / entity  : {train_avg:>12.1f}
  Zero-candidate entities  : {train_zero:>12,}

  Candidate distribution:
    0              : {bkts['0']:>10,}
    1  - 10        : {bkts['1-10']:>10,}
    11 - 50        : {bkts['11-50']:>10,}
    51 - 100       : {bkts['51-100']:>10,}
    101 - 200      : {bkts['101-200']:>10,}
    201 - 300      : {bkts['201-300']:>10,}
    300+ (capped)  : {bkts['300+(cap)']:>10,}

  BLOCKING RECALL
  -----------------------------------------------
  True matches             : {r['total_true_matches']:>12,}
  Found in candidates      : {r['found_in_candidates']:>12,}
  Missed                   : {r['missed']:>12,}
  RECALL                   : {r['blocking_recall']*100:>11.2f}%
  Entities with any miss   : {r['n_entities_with_miss']:>12,}
  Zero-cand misses         : {r['zero_cand_misses']:>12,}

  TEST CANDIDATES
  -----------------------------------------------
  S1 entities              : {len(test_cands):>12,}
  Total candidate pairs    : {test_total:>12,}
  Avg candidates / entity  : {test_avg:>12.1f}

  Wall time                : {wall_total/60:>11.1f} min
""")

    status = (
        "[WARNING] Recall < 90% - BM25 boost needed!"
        if r['blocking_recall'] < 0.90 else
        "[NOTE] Recall < 95% - consider BM25 for tail misses."
        if r['blocking_recall'] < 0.95 else
        "[OK] Recall >= 95% - excellent! Proceed to Phase 4."
    )
    print(f"  {status}")

    # Save stats
    stats = {
        "version": "v3",
        "config": {"max_candidates": MAX_CANDIDATES},
        "keys": ["n2","n1all","sfx2","fl","n3","zip","addr_12","addr_23","addr_end"],
        "recall": r,
        "train": {
            "n_entities": len(train_cands),
            "total_pairs": train_total,
            "avg_per_entity": round(train_avg, 2),
            "zero_candidate": train_zero,
            "distribution": bkts,
        },
        "test": {
            "n_entities": len(test_cands),
            "total_pairs": test_total,
            "avg_per_entity": round(test_avg, 2),
        },
        "wall_time_min": round(wall_total / 60, 1),
    }
    sp = OUT / "blocking_stats_v3.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\n  Stats -> {sp}")

    if r["sample_misses"]:
        print("\n  SAMPLE MISSES:")
        for eg in r["sample_misses"][:3]:
            print(f"    {eg['s1_id']}: {eg['n_found']}/{eg['n_true']} found | "
                  f"cands={eg['n_cands']} | missed={eg['missed']}")

    print("\n  Next: Phase 4 - Feature Engineering")


if __name__ == "__main__":
    main()
