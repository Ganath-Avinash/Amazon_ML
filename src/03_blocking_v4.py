"""
Phase 3 (v4) — Blocking — FAST, LOW-MEMORY, HIGH RECALL
Amazon ML Challenge 2026

Root cause of v2/v3 slowness:
  str.split() creates LIST columns (huge RAM — each row stores a Python list of strings).
  For 10M rows this creates 3-5 GB of intermediate data, causing swap/thrash.

v4 fix:
  Use regex str.extract() to pull individual tokens as SCALAR string columns.
  Regex columns = same memory as a normal string column (~50 MB for 5M rows).
  Then use .to_list() column-by-column for iteration (5-10x faster than iter_rows).

Blocking Keys (9 types):
  n2      : country + first 2 name tokens
  n1all   : country + first name token  (typo fallback)
  sfx2    : country + last 2 name tokens (transposition)
  fl      : country + first + last token (middle drops)
  n3      : country + first 3 name tokens (high precision)
  zip     : country + valid ZIP/PIN
  addr_12 : country + addr tokens [1,2]  (locality, skips house#)
  addr_23 : country + addr tokens [2,3]  (deeper locality)
  addr_e  : country + addr tokens [-3,-2] (city-area end)

Run from repo root:
    python -X utf8 src/03_blocking_v4.py
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

# Tokens too common to use as address blocking keys
SKIP_ADDR = {
    "road","street","avenue","blvd","drive","lane","court","place",
    "near","block","sector","phase","plot","null","none",
    "north","south","east","west","new","old","main",
    "unit","suite","building","floor","apartment","colony",
    "nagar","marg","vihar","enclave","park","market","complex","tower",
    "india","united","states",
    # US state abbreviations (2-char — filtered by len>=3 anyway)
    # Indian state words
    "pradesh","maharashtra","karnataka","gujarat","rajasthan","bengal",
    "kerala","andhra","telangana","bihar","odisha","jharkhand","tamil",
}


# ── Fast key-column extraction using regex (scalar columns, not list columns) ─

def add_key_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract token columns using regex — produces SCALAR string columns,
    not list columns. Regex is vectorized in Polars and uses ~10x less RAM
    than str.split() + list.get().
    """
    df = df.with_columns([
        # Country normalised
        pl.col("country").str.to_lowercase().str.strip_chars().alias("_cty"),

        # Name tokens (regex: \S+ = non-whitespace token)
        pl.col("norm_name").str.extract(r"^(\S+)", 1).alias("_n0"),
        pl.col("norm_name").str.extract(r"^\S+\s+(\S+)", 1).alias("_n1"),
        pl.col("norm_name").str.extract(r"^\S+\s+\S+\s+(\S+)", 1).alias("_n2"),
        pl.col("norm_name").str.extract(r"(\S+)\s*$", 1).alias("_nlast"),
        pl.col("norm_name").str.extract(r"(\S+)\s+\S+\s*$", 1).alias("_n2last"),
        pl.col("norm_name").str.count_matches(r"\S+").cast(pl.Int32).alias("_nlen"),

        # Address tokens (position 0 = house#, 1,2,3 = street/locality)
        pl.col("norm_address").str.extract(r"^(\S+)", 1).alias("_a0"),
        pl.col("norm_address").str.extract(r"^\S+\s+(\S+)", 1).alias("_a1"),
        pl.col("norm_address").str.extract(r"^\S+\s+\S+\s+(\S+)", 1).alias("_a2"),
        pl.col("norm_address").str.extract(r"^\S+\s+\S+\s+\S+\s+(\S+)", 1).alias("_a3"),
        pl.col("norm_address").str.extract(r"(\S+)\s+\S+\s+\S+\s*$", 1).alias("_ae3"),
        pl.col("norm_address").str.extract(r"(\S+)\s+\S+\s*$", 1).alias("_ae2"),
        pl.col("norm_address").str.count_matches(r"\S+").cast(pl.Int32).alias("_alen"),
    ])

    # ZIP validity fix: reject if zip_pin == first address token (= street number)
    df = df.with_columns([
        pl.when(
            pl.col("zip_pin").is_not_null() &
            (pl.col("zip_pin") != "") &
            (pl.col("zip_pin") != pl.col("_a0"))
        ).then(pl.col("zip_pin")).otherwise(None).alias("_zip_valid")
    ])

    return df


# ── Build index using column-wise .to_list() (fast, avoids iter_rows overhead) ─

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
        print(f"    Loaded {len(df):,} rows. Computing key columns ...")
        df = add_key_columns(df)

        # Extract all columns as Python lists at once (fast bulk operation)
        eids   = df["entity_id"].to_list()
        cty    = df["_cty"].to_list()
        n0     = df["_n0"].to_list()
        n1     = df["_n1"].to_list()
        n2     = df["_n2"].to_list()
        nlast  = df["_nlast"].to_list()
        n2last = df["_n2last"].to_list()
        nlen   = df["_nlen"].to_list()
        a1     = df["_a1"].to_list()
        a2     = df["_a2"].to_list()
        a3     = df["_a3"].to_list()
        ae3    = df["_ae3"].to_list()
        ae2    = df["_ae2"].to_list()
        alen   = df["_alen"].to_list()
        zipv   = df["_zip_valid"].to_list()
        del df  # free memory immediately

        print(f"    Columns extracted. Building index ...")
        count = 0
        for i in range(len(eids)):
            eid = eids[i]
            c   = cty[i] or ""
            keys = _make_keys(
                c,
                n0[i], n1[i], n2[i], nlast[i], n2last[i], nlen[i] or 0,
                a1[i], a2[i], a3[i], ae3[i], ae2[i], alen[i] or 0,
                zipv[i],
            )
            for k in keys:
                index[k].append(eid)
            count += 1
            if count % LOG_EVERY == 0:
                print(f"    {label}: {count:,} rows  ({time.time()-t0:.0f}s)")

        grand_total += count
        print(f"  {label}: {count:,} rows in {time.time()-t0:.1f}s")

    n_keys   = len(index)
    n_values = sum(len(v) for v in index.values())
    print(f"\n  Index: {n_keys:,} unique keys | {n_values:,} entries | "
          f"{grand_total:,} records indexed")
    return index


def _make_keys(c, n0, n1, n2, nlast, n2last, nlen,
               a1, a2, a3, ae3, ae2, alen, zipv) -> list:
    """Emit all blocking key tuples for a single record (called from tight loop)."""
    keys = []

    # ── Name keys ─────────────────────────────────────────────────────────────
    if n0 and len(n0) >= 2:
        # K1: first 2 tokens
        if n1 and len(n1) >= 2:
            keys.append(("n2", c, n0, n1))
        # K2: first token alone (ALL names — catches typos via shared first word)
        keys.append(("n1all", c, n0))

    if nlen >= 3:
        # K3: last 2 tokens (handles transpositions)
        if n2last and nlast and len(n2last) >= 2 and len(nlast) >= 2:
            keys.append(("sfx2", c, n2last, nlast))
        # K4: first + last (handles middle drops/inserts)
        if n0 and nlast and len(n0) >= 2 and len(nlast) >= 2:
            keys.append(("fl", c, n0, nlast))
        # K5: first 3 tokens (high precision)
        if n0 and n1 and n2 and len(n0) >= 2 and len(n1) >= 2 and len(n2) >= 2:
            keys.append(("n3", c, n0, n1, n2))

    # ── ZIP key ───────────────────────────────────────────────────────────────
    if zipv:
        keys.append(("zip", c, zipv))

    # ── Address keys (bridge Hindi/Tamil ↔ English via shared address) ────────
    # K7: addr tokens [1,2] — street/locality, skips house# at [0]
    if a1 and a2 and len(a1) >= 3 and len(a2) >= 3 \
            and a1 not in SKIP_ADDR and a2 not in SKIP_ADDR:
        keys.append(("addr_12", c, a1, a2))

    # K8: addr tokens [2,3] — deeper locality
    if a2 and a3 and len(a2) >= 3 and len(a3) >= 3 \
            and a2 not in SKIP_ADDR and a3 not in SKIP_ADDR:
        keys.append(("addr_23", c, a2, a3))

    # K9: addr end tokens [-3,-2] — city/area (only for long enough addresses)
    if ae3 and ae2 and len(ae3) >= 3 and len(ae2) >= 3 \
            and ae3 not in SKIP_ADDR and ae2 not in SKIP_ADDR \
            and alen >= 5:
        keys.append(("addr_e", c, ae3, ae2))

    return keys


# ── Generate candidates ───────────────────────────────────────────────────────

def generate_candidates(s1_path: Path, index: defaultdict) -> dict:
    print(f"\n  Reading S1 — {s1_path.name}")
    t0 = time.time()

    df = pl.read_csv(
        s1_path, separator="\t", infer_schema_length=0, ignore_errors=True,
        columns=["entity_id", "country", "norm_name", "norm_address", "zip_pin"]
    )
    df = add_key_columns(df)

    eids   = df["entity_id"].to_list()
    cty    = df["_cty"].to_list()
    n0     = df["_n0"].to_list()
    n1     = df["_n1"].to_list()
    n2     = df["_n2"].to_list()
    nlast  = df["_nlast"].to_list()
    n2last = df["_n2last"].to_list()
    nlen   = df["_nlen"].to_list()
    a1     = df["_a1"].to_list()
    a2     = df["_a2"].to_list()
    a3     = df["_a3"].to_list()
    ae3    = df["_ae3"].to_list()
    ae2    = df["_ae2"].to_list()
    alen   = df["_alen"].to_list()
    zipv   = df["_zip_valid"].to_list()
    del df

    candidates  = {}
    count       = 0
    total_cands = 0
    zero_cand   = 0
    capped      = 0

    for i in range(len(eids)):
        s1_id = eids[i]
        keys  = _make_keys(
            cty[i] or "", n0[i], n1[i], n2[i], nlast[i], n2last[i], nlen[i] or 0,
            a1[i], a2[i], a3[i], ae3[i], ae2[i], alen[i] or 0, zipv[i],
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
            avg = total_cands / count
            print(f"    {count:,} S1 | avg: {avg:.1f} | zero: {zero_cand:,} | "
                  f"capped: {capped:,} | {time.time()-t0:.0f}s")

    avg = total_cands / count if count else 0
    print(f"  Done: {count:,} S1 in {time.time()-t0:.1f}s | "
          f"avg {avg:.1f} | {zero_cand:,} zero | {capped:,} capped")
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
                    "s1_id": s1_id, "n_true": len(true_set),
                    "n_found": len(found), "n_missed": len(missed),
                    "n_cands": len(cand_set), "missed": list(missed)[:3],
                })

    recall = total_found / total_true if total_true > 0 else 0.0
    return {
        "total_true_matches":   total_true,
        "found_in_candidates":  total_found,
        "missed":               total_true - total_found,
        "blocking_recall":      round(recall, 6),
        "n_entities_with_miss": n_miss_ents,
        "zero_cand_misses":     zero_c_miss,
        "sample_misses":        missed_egs,
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
    print(f"  Saved {out_path.stat().st_size/1e6:.1f} MB in {time.time()-t0:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Amazon ML Challenge 2026 - Phase 3 v4: Blocking")
    print("  (Regex extraction, no list columns, column-wise iteration)")
    print("=" * 65)
    print(f"\n  MAX_CANDIDATES = {MAX_CANDIDATES}")
    print(f"  Keys: n2, n1all, sfx2, fl, n3, zip, addr_12, addr_23, addr_e\n")

    wall_start = time.time()

    # ═══ TRAIN ═══════════════════════════════════════════════════
    print("-" * 65)
    print("  TRAIN BLOCKING")
    print("-" * 65)
    print("\n[1/4] Building train index ...")
    train_index = build_index(TRAIN_NORM / "norm_source2.tsv",
                               TRAIN_NORM / "norm_source3.tsv")

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
    print("\n[1/2] Building test index ...")
    test_index = build_index(TEST_NORM / "norm_source2.tsv",
                              TEST_NORM / "norm_source3.tsv")

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

    bkts = {"0":0,"1-10":0,"11-50":0,"51-100":0,"101-200":0,"201-300":0,"300+":0}
    for cands in train_cands.values():
        n = len(cands)
        if   n == 0:    bkts["0"] += 1
        elif n <= 10:   bkts["1-10"] += 1
        elif n <= 50:   bkts["11-50"] += 1
        elif n <= 100:  bkts["51-100"] += 1
        elif n <= 200:  bkts["101-200"] += 1
        elif n <= 300:  bkts["201-300"] += 1
        else:           bkts["300+"] += 1

    r = recall
    print(f"""
{"="*65}
  PHASE 3 v4 COMPLETE
{"="*65}

  TRAIN
  S1 entities           : {len(train_cands):>12,}
  Total candidate pairs : {train_total:>12,}
  Avg candidates/entity : {train_avg:>12.1f}
  Zero-candidate        : {train_zero:>12,}

  Distribution:
    0          : {bkts['0']:>10,}
    1-10       : {bkts['1-10']:>10,}
    11-50      : {bkts['11-50']:>10,}
    51-100     : {bkts['51-100']:>10,}
    101-200    : {bkts['101-200']:>10,}
    201-300    : {bkts['201-300']:>10,}
    300+(cap)  : {bkts['300+']:>10,}

  BLOCKING RECALL
  True matches          : {r['total_true_matches']:>12,}
  Found in candidates   : {r['found_in_candidates']:>12,}
  Missed                : {r['missed']:>12,}
  RECALL                : {r['blocking_recall']*100:>11.2f}%
  Entities with miss    : {r['n_entities_with_miss']:>12,}
  Zero-cand misses      : {r['zero_cand_misses']:>12,}

  TEST
  S1 entities           : {len(test_cands):>12,}
  Total candidate pairs : {test_total:>12,}
  Avg candidates/entity : {test_avg:>12.1f}

  Wall time             : {wall_total/60:>11.1f} min
""")

    status = (
        "[WARNING] Recall < 90%"  if r['blocking_recall'] < 0.90 else
        "[NOTE] Recall < 95%"     if r['blocking_recall'] < 0.95 else
        "[OK] Recall >= 95% — proceed to Phase 4!"
    )
    print(f"  {status}")

    stats = {
        "version": "v4",
        "config": {"max_candidates": MAX_CANDIDATES},
        "keys": ["n2","n1all","sfx2","fl","n3","zip","addr_12","addr_23","addr_e"],
        "recall": r,
        "train": {"n_entities": len(train_cands), "total_pairs": train_total,
                  "avg": round(train_avg,2), "zero": train_zero, "dist": bkts},
        "test":  {"n_entities": len(test_cands),  "total_pairs": test_total,
                  "avg": round(test_avg,2)},
        "wall_time_min": round(wall_total/60, 1),
    }
    sp = OUT / "blocking_stats_v4.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Stats -> {sp}")

    if r["sample_misses"]:
        print("\n  SAMPLE MISSES:")
        for eg in r["sample_misses"][:3]:
            print(f"    {eg['s1_id']}: {eg['n_found']}/{eg['n_true']} found | "
                  f"cands={eg['n_cands']} | missed={eg['missed']}")

    print("\n  Next: Phase 4 - Feature Engineering")


if __name__ == "__main__":
    main()
