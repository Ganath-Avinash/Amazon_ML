"""
Phase 3 — Blocking / Candidate Generation
Amazon ML Challenge 2026: Business Entity Resolution

Strategy:
  5 complementary blocking key types built from (country + name tokens + ZIP)
  ensuring >95% recall while keeping candidates manageable (<200 per S1 entity).

  KEY TYPES:
    n2   : country + first 2 name tokens          (primary)
    sfx2 : country + last 2 name tokens           (handles transpositions)
    fl   : country + first token + last token      (handles middle-word drops)
    n3   : country + first 3 tokens               (high-precision fallback)
    zip  : country + ZIP/PIN  (FIXED: not first addr token)

Output:
  output/train_candidate_pairs.tsv  (with recall stats)
  output/candidate_pairs.tsv        (TEST — submit this)

Run from repo root:
    python src/03_blocking.py
"""

import csv
import json
import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from collections import defaultdict
from pathlib import Path

import polars as pl

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path("Dataset/student_resource/dataset")
TRAIN_NORM = BASE / "train" / "normalized"
TEST_NORM  = BASE / "test"  / "normalized"
TRAIN_RAW  = BASE / "train"
OUT        = Path("output")
OUT.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_CANDIDATES = 200   # cap per S1 entity (prevents runaway common-name blocks)
LOG_EVERY      = 500_000

# ── Helpers ────────────────────────────────────────────────────────────────────

def sig_tokens(norm_name: str) -> list:
    """Return tokens from norm_name that are >=2 chars (filters noise)."""
    return [t for t in (norm_name or "").split() if len(t) >= 2]


def is_valid_zip(addr_tokens: list, zip_val: str) -> bool:
    """
    ZIP/PIN fix: reject if it matches the FIRST address token.
    e.g. '17560 Ellis Road' → addr_tokens[0]='17560' → NOT a valid ZIP.
    A real ZIP/PIN appears in the middle or end of the address string.
    """
    if not zip_val:
        return False
    if addr_tokens and addr_tokens[0] == zip_val:
        return False  # It's a street number, not a postal code
    return True


def make_keys(country: str, norm_name: str, norm_addr: str, zip_pin: str) -> list:
    """
    Generate all blocking key tuples for a single record.
    Each tuple is hashable and used as dict key in the inverted index.
    """
    cty    = (country or "").lower().strip()
    tokens = sig_tokens(norm_name)
    atoks  = (norm_addr or "").split()
    keys   = []

    n = len(tokens)

    # KEY 1 — first 2 name tokens  (primary workhorse)
    if n >= 2:
        keys.append(("n2", cty, tokens[0], tokens[1]))

    # KEY 2 — single token (short business names: "Subway", "Nike")
    if n == 1:
        keys.append(("n1", cty, tokens[0]))

    # KEY 3 — last 2 name tokens  (handles word-order transpositions)
    if n >= 3:
        keys.append(("sfx2", cty, tokens[-2], tokens[-1]))

    # KEY 4 — first + last token  (handles middle-word insertions/drops)
    if n >= 3:
        keys.append(("fl", cty, tokens[0], tokens[-1]))

    # KEY 5 — first 3 tokens  (tighter match, high precision for long names)
    if n >= 3:
        keys.append(("n3", cty, tokens[0], tokens[1], tokens[2]))

    # KEY 6 — ZIP / PIN  (only valid postal codes, not street numbers)
    if is_valid_zip(atoks, zip_pin):
        keys.append(("zip", cty, zip_pin))

    return keys


# ── Stage 1: Build inverted index ─────────────────────────────────────────────

def build_index(s2_path: Path, s3_path: Path) -> defaultdict:
    """
    Build multi-key inverted index over all S2 + S3 records.
    Returns: {key_tuple: [entity_id, ...]}
    """
    index = defaultdict(list)
    grand_total = 0

    for path, label in [(s2_path, "S2"), (s3_path, "S3")]:
        print(f"\n  Indexing {label} — {path.name}")
        df = pl.read_csv(
            path, separator="\t", infer_schema_length=0, ignore_errors=True,
            columns=["entity_id", "country", "norm_name", "norm_address", "zip_pin"]
        )
        t0 = time.time()
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
        print(f"  {label} done: {count:,} rows in {time.time()-t0:.1f}s")

    n_keys   = len(index)
    n_values = sum(len(v) for v in index.values())
    print(f"\n  Index: {n_keys:,} unique keys | {n_values:,} total entries | "
          f"{grand_total:,} records indexed")
    return index


# ── Stage 2: Query index for every S1 entity ──────────────────────────────────

def generate_candidates(s1_path: Path, index: defaultdict) -> dict:
    """
    For each S1 entity, collect all S2/S3 entities sharing any blocking key.
    Returns: {s1_entity_id: [candidate_ids, ...]}
    """
    print(f"\n  Reading S1 — {s1_path.name}")
    df = pl.read_csv(
        s1_path, separator="\t", infer_schema_length=0, ignore_errors=True,
        columns=["entity_id", "country", "norm_name", "norm_address", "zip_pin"]
    )

    candidates = {}
    t0 = time.time()
    count = 0
    total_cands = 0
    zero_cand   = 0

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
        cand_set.discard(s1_id)   # remove self just in case

        # Cap to avoid runaway blocks (common business names)
        if len(cand_set) > MAX_CANDIDATES:
            cand_list = list(cand_set)[:MAX_CANDIDATES]
        else:
            cand_list = list(cand_set)

        candidates[s1_id] = cand_list
        total_cands += len(cand_list)
        if len(cand_list) == 0:
            zero_cand += 1

        count += 1
        if count % 200_000 == 0:
            elapsed = time.time() - t0
            avg = total_cands / count
            print(f"    {count:,} S1 entities | avg candidates: {avg:.1f} | "
                  f"zero-candidate: {zero_cand:,} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    avg = total_cands / count if count else 0
    print(f"  Done: {count:,} S1 entities in {elapsed:.1f}s | "
          f"avg {avg:.1f} candidates | {zero_cand:,} with zero candidates")
    return candidates


# ── Stage 3: Measure blocking recall ──────────────────────────────────────────

def measure_recall(candidates: dict, gt_path: Path) -> dict:
    """
    Check what % of true ground-truth matches were captured by blocking.
    This is the RECALL CEILING for the downstream matching model.
    """
    print("\n  Loading ground truth for recall measurement …")
    gt = pl.read_csv(
        gt_path, separator="\t", infer_schema_length=0, ignore_errors=True
    )

    total_true  = 0
    total_found = 0
    missed_egs  = []
    zero_cand_misses = 0

    for row in gt.iter_rows(named=True):
        s1_id       = row["source1_entity_id"]
        matched_str = row.get("matched_entity_ids", "") or ""
        if not matched_str.strip():
            continue  # singleton — no true matches to recall

        true_set = set(m.strip() for m in matched_str.split(",") if m.strip())
        cand_set = set(candidates.get(s1_id, []))

        found  = true_set & cand_set
        missed = true_set - cand_set

        total_true  += len(true_set)
        total_found += len(found)

        if missed:
            if len(cand_set) == 0:
                zero_cand_misses += 1
            if len(missed_egs) < 10:
                missed_egs.append({
                    "s1_id":         s1_id,
                    "n_true":        len(true_set),
                    "n_found":       len(found),
                    "n_missed":      len(missed),
                    "missed_sample": list(missed)[:3],
                    "n_candidates":  len(cand_set),
                })

    recall = total_found / total_true if total_true > 0 else 0.0
    return {
        "total_true_matches":    total_true,
        "found_in_candidates":   total_found,
        "missed":                total_true - total_found,
        "blocking_recall":       round(recall, 6),
        "n_entities_with_miss":  len(missed_egs),
        "zero_cand_misses":      zero_cand_misses,
        "sample_misses":         missed_egs,
    }


# ── Stage 4: Save output TSV ───────────────────────────────────────────────────

def save_candidates(candidates: dict, out_path: Path):
    t0 = time.time()
    print(f"\n  Writing {out_path} …")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id, cands in candidates.items():
            writer.writerow([s1_id, ",".join(cands)])
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved {size_mb:.1f} MB in {time.time()-t0:.1f}s")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Amazon ML Challenge 2026 — Phase 3: Blocking")
    print("=" * 65)
    print(f"\n  Config: MAX_CANDIDATES={MAX_CANDIDATES}")
    print(f"  Keys: n2, n1, sfx2, fl, n3, zip(fixed)\n")

    wall_start = time.time()

    # ══════════════════════════════════════════════════════════════
    # TRAIN SPLIT
    # ══════════════════════════════════════════════════════════════
    print("━" * 65)
    print("  TRAIN BLOCKING")
    print("━" * 65)

    print("\n[1/4] Building index from train S2 + S3 …")
    train_index = build_index(
        TRAIN_NORM / "norm_source2.tsv",
        TRAIN_NORM / "norm_source3.tsv",
    )

    print("\n[2/4] Generating candidates for train S1 …")
    train_cands = generate_candidates(TRAIN_NORM / "norm_source1.tsv", train_index)

    print("\n[3/4] Measuring blocking recall on train GT …")
    recall = measure_recall(train_cands, TRAIN_RAW / "train_ground_truth.tsv")

    print("\n[4/4] Saving train candidate pairs …")
    save_candidates(train_cands, OUT / "train_candidate_pairs.tsv")

    # Free train index — no longer needed
    del train_index

    # ══════════════════════════════════════════════════════════════
    # TEST SPLIT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "━" * 65)
    print("  TEST BLOCKING")
    print("━" * 65)

    print("\n[1/2] Building index from test S2 + S3 …")
    test_index = build_index(
        TEST_NORM / "norm_source2.tsv",
        TEST_NORM / "norm_source3.tsv",
    )

    print("\n[2/2] Generating candidates for test S1 …")
    test_cands = generate_candidates(TEST_NORM / "norm_source1.tsv", test_index)

    print("\nSaving test candidate pairs (final submission file) …")
    save_candidates(test_cands, OUT / "candidate_pairs.tsv")

    del test_index

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    wall_total = time.time() - wall_start

    # Candidate stats
    train_total = sum(len(v) for v in train_cands.values())
    train_avg   = train_total / len(train_cands) if train_cands else 0
    train_zero  = sum(1 for v in train_cands.values() if not v)
    test_total  = sum(len(v) for v in test_cands.values())
    test_avg    = test_total / len(test_cands) if test_cands else 0

    # Candidate size distribution
    buckets = {"0": 0, "1-10": 0, "11-50": 0, "51-100": 0, "101-200": 0, "200+": 0}
    for cands in train_cands.values():
        n = len(cands)
        if n == 0:          buckets["0"] += 1
        elif n <= 10:       buckets["1-10"] += 1
        elif n <= 50:       buckets["11-50"] += 1
        elif n <= 100:      buckets["51-100"] += 1
        elif n <= 200:      buckets["101-200"] += 1
        else:               buckets["200+"] += 1

    print("\n" + "=" * 65)
    print("  PHASE 3 COMPLETE — SUMMARY")
    print("=" * 65)

    print(f"""
  TRAIN CANDIDATES
  ─────────────────────────────────────────────
  S1 entities              : {len(train_cands):>12,}
  Total candidate pairs    : {train_total:>12,}
  Avg candidates / entity  : {train_avg:>12.1f}
  Zero-candidate entities  : {train_zero:>12,}

  Candidate size distribution:
    0          : {buckets['0']:>10,}
    1 – 10     : {buckets['1-10']:>10,}
    11 – 50    : {buckets['11-50']:>10,}
    51 – 100   : {buckets['51-100']:>10,}
    101 – 200  : {buckets['101-200']:>10,}
    >200 (cap) : {buckets['200+']:>10,}

  BLOCKING RECALL (train)
  ─────────────────────────────────────────────
  True matches             : {recall['total_true_matches']:>12,}
  Found in candidates      : {recall['found_in_candidates']:>12,}
  Missed                   : {recall['missed']:>12,}
  RECALL                   : {recall['blocking_recall']*100:>11.2f}%
  Entities with any miss   : {recall['n_entities_with_miss']:>12,}
  Zero-cand misses         : {recall['zero_cand_misses']:>12,}

  TEST CANDIDATES
  ─────────────────────────────────────────────
  S1 entities              : {len(test_cands):>12,}
  Total candidate pairs    : {test_total:>12,}
  Avg candidates / entity  : {test_avg:>12.1f}

  Wall time                : {wall_total/60:>11.1f} min

  OUTPUT FILES
  ─────────────────────────────────────────────
  output/train_candidate_pairs.tsv   (train, with recall stats)
  output/candidate_pairs.tsv         (TEST — use for submission)
""")

    if recall['blocking_recall'] < 0.90:
        print("  [WARNING] Recall < 90% — consider adding more blocking keys!")
    elif recall['blocking_recall'] < 0.95:
        print("  [NOTE] Recall < 95% — consider BM25 boost for missed entities.")
    else:
        print("  [OK] Recall >= 95% — blocking is solid. Proceed to Phase 4.")

    # Save recall stats to JSON for reporting
    stats_path = OUT / "blocking_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "recall": recall,
            "train_candidates": {
                "n_entities": len(train_cands),
                "total_pairs": train_total,
                "avg_per_entity": round(train_avg, 2),
                "zero_candidate": train_zero,
                "distribution": buckets,
            },
            "test_candidates": {
                "n_entities": len(test_cands),
                "total_pairs": test_total,
                "avg_per_entity": round(test_avg, 2),
            }
        }, f, indent=2, default=str)
    print(f"  Stats saved to {stats_path}")

    # Show sample misses for debugging
    if recall['sample_misses']:
        print("\n  SAMPLE MISSED ENTITIES (for debugging blocking gaps):")
        for eg in recall['sample_misses'][:3]:
            print(f"    {eg['s1_id']}: {eg['n_found']}/{eg['n_true']} found, "
                  f"missed {eg['n_missed']}, candidates={eg['n_candidates']}")
            print(f"      missed IDs: {eg['missed_sample']}")

    print("\n  Next: Phase 4 — Feature Engineering")


if __name__ == "__main__":
    main()
