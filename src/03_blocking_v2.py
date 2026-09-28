"""
Phase 3 (v2) — Blocking / Candidate Generation — IMPROVED
Amazon ML Challenge 2026: Business Entity Resolution

v2 Fixes vs v1 (50.34% recall):
  - Added address-based blocking keys (critical for Hindi/Tamil entities where
    name transliteration differs but address is shared)
  - Added first-single-token key for ALL names (not just 1-token names)
    catches typos like "williams" vs "wilblims" via shared first token "maure"
  - Added 3-char phonetic prefix key to catch more typo variants
  - Raised MAX_CANDIDATES to 300 to avoid capping true matches

Blocking Keys (9 total):
  n2      : country + first 2 name tokens          (primary)
  n1all   : country + first name token             (typo fallback)
  sfx2    : country + last 2 name tokens           (transposition)
  fl      : country + first + last name token      (middle-word drops)
  n3      : country + first 3 name tokens          (high-precision)
  zip     : country + ZIP/PIN  (fixed: not first addr token)
  addr_m  : country + addr tokens [2:4]            (mid-address, skips house#)
  addr_e  : country + addr tokens [-3:-1]          (city-area end tokens)
  addr_3  : country + addr tokens [1:3]            (second+third addr tokens)

Run from repo root:
    python -X utf8 src/03_blocking_v2.py
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

# ── Config ────────────────────────────────────────────────────────────────────
MAX_CANDIDATES = 300   # raised from 200 to avoid capping true matches
LOG_EVERY      = 500_000

# Common city/state tokens that are too generic for address blocking
# (adding them as keys would create massive, low-quality buckets)
COMMON_ADDR_TOKENS = {
    # US states
    "ca", "ny", "tx", "fl", "il", "pa", "oh", "ga", "nc", "mi",
    "nj", "va", "wa", "az", "ma", "tn", "in", "mo", "md", "wi",
    "co", "mn", "sc", "al", "la", "ky", "or", "ok", "ct", "ut",
    "ia", "nv", "ar", "ms", "ks", "ne", "nm", "wv", "id", "hi",
    "nh", "me", "ri", "mt", "de", "sd", "nd", "ak", "vt", "wy",
    # Indian states (common abbreviations after normalization)
    "pradesh", "maharashtra", "karnataka", "gujarat", "rajasthan",
    "tamil", "nadu", "bengal", "kerala", "andhra", "telangana",
    "bihar", "odisha", "jharkhand", "uttarakhand", "himachal",
    # Generic
    "north", "south", "east", "west", "new", "old", "main", "road",
    "street", "avenue", "near", "block", "sector", "phase", "plot",
    "null", "none", "na", "building", "floor", "unit", "suite",
    "india", "united", "states",
}


def sig_tokens(norm_name: str) -> list:
    return [t for t in (norm_name or "").split() if len(t) >= 2]


def sig_addr_tokens(norm_addr: str) -> list:
    """Return address tokens that are long enough and not overly generic."""
    return [
        t for t in (norm_addr or "").split()
        if len(t) >= 3 and t not in COMMON_ADDR_TOKENS
    ]


def is_valid_zip(addr_tokens: list, zip_val: str) -> bool:
    """ZIP/PIN fix: reject if it's the first address token (= street number)."""
    if not zip_val:
        return False
    if addr_tokens and addr_tokens[0] == zip_val:
        return False
    return True


def make_keys(country: str, norm_name: str, norm_addr: str, zip_pin: str) -> list:
    """
    Generate all blocking key tuples for one record.
    Returns list of hashable tuples for use as dict keys.
    """
    cty    = (country or "").lower().strip()
    ntoks  = sig_tokens(norm_name)
    atoks  = (norm_addr or "").split()
    satoks = sig_addr_tokens(norm_addr)  # filtered for quality
    keys   = []
    n      = len(ntoks)

    # ── NAME KEYS ─────────────────────────────────────────────────
    # Key 1: first 2 name tokens (primary workhorse)
    if n >= 2:
        keys.append(("n2", cty, ntoks[0], ntoks[1]))

    # Key 2: first name token only (ALL names — catches typo variants
    #         like "williams" vs "wilblims" via shared first token)
    if n >= 1:
        keys.append(("n1all", cty, ntoks[0]))

    # Key 3: last 2 name tokens (word-order transpositions)
    if n >= 3:
        keys.append(("sfx2", cty, ntoks[-2], ntoks[-1]))

    # Key 4: first + last token (middle-word drops/inserts)
    if n >= 3:
        keys.append(("fl", cty, ntoks[0], ntoks[-1]))

    # Key 5: first 3 tokens (high-precision for longer names)
    if n >= 3:
        keys.append(("n3", cty, ntoks[0], ntoks[1], ntoks[2]))

    # ── ZIP KEY ───────────────────────────────────────────────────
    # Key 6: country + ZIP/PIN (fixed: not first addr token)
    if is_valid_zip(atoks, zip_pin):
        keys.append(("zip", cty, zip_pin))

    # ── ADDRESS KEYS ──────────────────────────────────────────────
    # These are critical for transliterated names (Hindi/Tamil in S2/S3
    # vs English in S1) where name tokens don't match at all but the
    # address (street, locality, city) is the same.

    # Key 7: addr tokens [2:4] — locality / street name
    #         skips position [0]=house# and [1]=building name / street#
    if len(satoks) >= 4:
        keys.append(("addr_m", cty, satoks[2], satoks[3]))

    # Key 8: addr tokens [1:3] — second + third significant addr token
    if len(satoks) >= 3:
        keys.append(("addr_s", cty, satoks[1], satoks[2]))

    # Key 9: addr tokens [-3:-1] — city-area end tokens
    #         (penultimate tokens are often city/locality names)
    if len(satoks) >= 4:
        end_toks = satoks[-3:-1]
        if all(t not in COMMON_ADDR_TOKENS for t in end_toks):
            keys.append(("addr_e", cty, end_toks[0], end_toks[1]))

    return keys


# ── Build inverted index ──────────────────────────────────────────────────────

def build_index(s2_path: Path, s3_path: Path) -> defaultdict:
    index = defaultdict(list)
    grand_total = 0

    for path, label in [(s2_path, "S2"), (s3_path, "S3")]:
        print(f"\n  Indexing {label} — {path.name}")
        df = pl.read_csv(
            path, separator="\t", infer_schema_length=0, ignore_errors=True,
            columns=["entity_id", "country", "norm_name", "norm_address", "zip_pin"]
        )
        t0    = time.time()
        count = 0
        for row in df.iter_rows(named=True):
            keys = make_keys(
                row["country"],
                row["norm_name"]    or "",
                row["norm_address"] or "",
                row["zip_pin"]      or "",
            )
            eid = row["entity_id"]
            for k in keys:
                index[k].append(eid)
            count += 1
            if count % LOG_EVERY == 0:
                print(f"    {label}: {count:,} rows  ({time.time()-t0:.0f}s)")

        grand_total += count
        print(f"  {label}: {count:,} rows in {time.time()-t0:.1f}s")

    print(f"\n  Index: {len(index):,} unique keys | "
          f"{sum(len(v) for v in index.values()):,} total entries | "
          f"{grand_total:,} records indexed")
    return index


# ── Generate candidates for S1 ───────────────────────────────────────────────

def generate_candidates(s1_path: Path, index: defaultdict) -> dict:
    df = pl.read_csv(
        s1_path, separator="\t", infer_schema_length=0, ignore_errors=True,
        columns=["entity_id", "country", "norm_name", "norm_address", "zip_pin"]
    )

    candidates  = {}
    t0          = time.time()
    count       = 0
    total_cands = 0
    zero_cand   = 0
    capped      = 0

    for row in df.iter_rows(named=True):
        s1_id = row["entity_id"]
        keys  = make_keys(
            row["country"],
            row["norm_name"]    or "",
            row["norm_address"] or "",
            row["zip_pin"]      or "",
        )

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
            elapsed = time.time() - t0
            avg = total_cands / count
            print(f"    {count:,} S1 entities | avg cands: {avg:.1f} | "
                  f"zero: {zero_cand:,} | capped: {capped:,} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    avg = total_cands / count if count else 0
    print(f"  Done: {count:,} S1 entities in {elapsed:.1f}s | "
          f"avg {avg:.1f} candidates | {zero_cand:,} zero | {capped:,} capped")
    return candidates


# ── Measure recall ────────────────────────────────────────────────────────────

def measure_recall(candidates: dict, gt_path: Path) -> dict:
    gt = pl.read_csv(gt_path, separator="\t", infer_schema_length=0, ignore_errors=True)

    total_true   = 0
    total_found  = 0
    n_miss_ents  = 0
    zero_c_miss  = 0
    missed_egs   = []

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
            if len(cand_set) == 0:
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
        "total_true_matches":  total_true,
        "found_in_candidates": total_found,
        "missed":              total_true - total_found,
        "blocking_recall":     round(recall, 6),
        "n_entities_with_miss": n_miss_ents,
        "zero_cand_misses":    zero_c_miss,
        "sample_misses":       missed_egs,
    }


# ── Save TSV ──────────────────────────────────────────────────────────────────

def save_candidates(candidates: dict, out_path: Path):
    t0 = time.time()
    print(f"  Writing {out_path} …")
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
    print("  Amazon ML Challenge 2026 - Phase 3 v2: Blocking (Improved)")
    print("=" * 65)
    print(f"\n  Config: MAX_CANDIDATES={MAX_CANDIDATES}")
    print(f"  Keys: n2, n1all, sfx2, fl, n3, zip, addr_m, addr_s, addr_e")
    print(f"  v2 improvements: address keys + first-token-all key\n")

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

    print("\nSaving candidate_pairs.tsv (submission file) ...")
    save_candidates(test_cands, OUT / "candidate_pairs.tsv")

    del test_index

    # ═══ SUMMARY ═════════════════════════════════════════════════
    wall_total = time.time() - wall_start

    train_total = sum(len(v) for v in train_cands.values())
    train_avg   = train_total / len(train_cands) if train_cands else 0
    train_zero  = sum(1 for v in train_cands.values() if not v)
    test_total  = sum(len(v) for v in test_cands.values())
    test_avg    = test_total / len(test_cands) if test_cands else 0

    # Distribution buckets
    bkts = {"0": 0, "1-10": 0, "11-50": 0, "51-100": 0, "101-200": 0,
            "201-300": 0, "300+(cap)": 0}
    for cands in train_cands.values():
        n = len(cands)
        if n == 0:           bkts["0"] += 1
        elif n <= 10:        bkts["1-10"] += 1
        elif n <= 50:        bkts["11-50"] += 1
        elif n <= 100:       bkts["51-100"] += 1
        elif n <= 200:       bkts["101-200"] += 1
        elif n <= 300:       bkts["201-300"] += 1
        else:                bkts["300+(cap)"] += 1

    r = recall
    print(f"""
{"="*65}
  PHASE 3 v2 COMPLETE - SUMMARY
{"="*65}

  TRAIN CANDIDATES
  -----------------------------------------------
  S1 entities              : {len(train_cands):>12,}
  Total candidate pairs    : {train_total:>12,}
  Avg candidates / entity  : {train_avg:>12.1f}
  Zero-candidate entities  : {train_zero:>12,}

  Candidate distribution:
    0            : {bkts['0']:>10,}
    1 - 10       : {bkts['1-10']:>10,}
    11 - 50      : {bkts['11-50']:>10,}
    51 - 100     : {bkts['51-100']:>10,}
    101 - 200    : {bkts['101-200']:>10,}
    201 - 300    : {bkts['201-300']:>10,}
    300+ (capped): {bkts['300+(cap)']:>10,}

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

    if r['blocking_recall'] < 0.90:
        print("  [WARNING] Recall < 90% - consider BM25 boost for missed entities!")
    elif r['blocking_recall'] < 0.95:
        print("  [NOTE] Recall < 95% - acceptable, consider BM25 for missed entities.")
    else:
        print("  [OK] Recall >= 95% - excellent blocking. Proceed to Phase 4.")

    # Save JSON stats
    stats = {
        "version": "v2",
        "config": {"max_candidates": MAX_CANDIDATES},
        "keys_used": ["n2","n1all","sfx2","fl","n3","zip","addr_m","addr_s","addr_e"],
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
    stats_path = OUT / "blocking_stats_v2.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Stats -> {stats_path}")

    if r["sample_misses"]:
        print("\n  SAMPLE MISSES (debugging):")
        for eg in r["sample_misses"][:3]:
            print(f"    {eg['s1_id']}: {eg['n_found']}/{eg['n_true']} found | "
                  f"cands={eg['n_cands']} | missed: {eg['missed']}")

    print("\n  Next: Phase 4 - Feature Engineering")


if __name__ == "__main__":
    main()
