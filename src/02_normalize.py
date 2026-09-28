"""
Phase 2 — Text Normalization Pipeline
Amazon ML Challenge 2026: Business Entity Resolution

Reads all 7 raw TSVs, applies cleaning, writes normalized versions to:
    Dataset/student_resource/dataset/train/normalized/
    Dataset/student_resource/dataset/test/normalized/

Run from repo root:
    python src/02_normalize.py
"""

import re
import sys
import time
from pathlib import Path

import polars as pl
from unidecode import unidecode

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE  = Path("Dataset/student_resource/dataset")
TRAIN = BASE / "train"
TEST  = BASE / "test"
TRAIN_OUT = TRAIN / "normalized"
TEST_OUT  = TEST  / "normalized"
TRAIN_OUT.mkdir(exist_ok=True)
TEST_OUT.mkdir(exist_ok=True)

# ─── Normalization dictionaries ───────────────────────────────────────────────

# Legal suffixes to strip from business names (order matters — longest first)
LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt. ltd", "pvt ltd.",
    "public limited", "pub ltd",
    "limited liability partnership", "llp",
    "limited liability company", "llc",
    "incorporated", "incorporation",
    "corporation", "corp.",  "corp",
    "limited", "ltd.", "ltd",
    "company", "co.",
    "partners", "partnership",
    "associates", "associate",
    "enterprises", "enterprise",
    "solutions", "solution",
    "services", "service",
    "group", "holdings", "holding",
    "industries", "industry",
    "international", "intl",
    "brothers", "bros.",  "bros",
    "& sons", "and sons",
]

# Common abbreviation expansions for NAMES
NAME_ABBREV = {
    "&":    "and",
    "st.":  "saint",   # for names like "St. Mary" - careful with street
    "dr.":  "doctor",
    "mr.":  "mister",
    "mrs.": "missus",
    "prof.":"professor",
    "natl": "national",
    "intl": "international",
    "assn": "association",
    "mgmt": "management",
    "mgr":  "manager",
    "dept": "department",
    "univ": "university",
    "tech": "technology",
    "sys":  "systems",
    "svc":  "service",
    "svcs": "services",
    "mfg":  "manufacturing",
    "dist": "distribution",
    "dev":  "development",
    "res":  "resources",
    "prop": "properties",
    "inv":  "investments",
    "fin":  "financial",
    "ins":  "insurance",
    "med":  "medical",
    "hosp": "hospital",
    "educ": "education",
    "comm": "communications",
}

# Common abbreviation expansions for ADDRESSES
ADDR_ABBREV = {
    # Street types
    "st":    "street",
    "st.":   "street",
    "ave":   "avenue",
    "ave.":  "avenue",
    "blvd":  "boulevard",
    "blvd.": "boulevard",
    "rd":    "road",
    "rd.":   "road",
    "dr":    "drive",
    "dr.":   "drive",
    "ln":    "lane",
    "ln.":   "lane",
    "ct":    "court",
    "ct.":   "court",
    "pl":    "place",
    "pl.":   "place",
    "sq":    "square",
    "sq.":   "square",
    "fwy":   "freeway",
    "hwy":   "highway",
    "pkwy":  "parkway",
    "expy":  "expressway",
    "trl":   "trail",
    "trce":  "trace",
    "ter":   "terrace",
    "ter.":  "terrace",
    "terr":  "terrace",
    # Directionals
    "n":     "north",
    "s":     "south",
    "e":     "east",
    "w":     "west",
    "ne":    "northeast",
    "nw":    "northwest",
    "se":    "southeast",
    "sw":    "southwest",
    # Building types
    "apt":   "apartment",
    "apt.":  "apartment",
    "ste":   "suite",
    "ste.":  "suite",
    "fl":    "floor",
    "bldg":  "building",
    "bldg.": "building",
    # Indian address
    "nagar": "nagar",  # keep but normalize
    "marg":  "marg",
    "rd":    "road",
    "ph":    "phase",
}

# Stopwords to remove from names (not addresses — addresses need location words)
NAME_STOPWORDS = {"the", "a", "an", "of", "in", "at", "by", "for", "and", "or"}

# Punctuation to normalize
PUNCT_RE = re.compile(r"[^\w\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

# Phone number pattern (remove from addresses — they're noise)
PHONE_RE = re.compile(r"\b(ph\.?\s*)?\d{6,}\b", re.IGNORECASE)

# ─── Core normalization functions ─────────────────────────────────────────────

def transliterate_to_ascii(text: str) -> str:
    """
    Convert non-ASCII characters to closest ASCII equivalent.
    'Énterprises' → 'Enterprises'
    Preserves native script words as their romanized form.
    """
    try:
        return unidecode(text)
    except Exception:
        return text

def strip_legal_suffixes(name: str) -> str:
    """Remove legal entity suffixes from the END of a business name."""
    name = name.strip()
    for suffix in LEGAL_SUFFIXES:
        # Match at end of string, with optional punctuation
        pattern = re.compile(
            r"[,\s]*\b" + re.escape(suffix) + r"\.?\s*$",
            re.IGNORECASE
        )
        name = pattern.sub("", name).strip()
    return name.strip(",. ")

def normalize_name(raw: str) -> str:
    """
    Full normalization pipeline for business names:
    1. Transliterate to ASCII (handles accents + native scripts)
    2. Lowercase
    3. Strip legal suffixes
    4. Expand abbreviations
    5. Remove punctuation
    6. Remove stopwords
    7. Collapse whitespace
    """
    if not raw or not isinstance(raw, str):
        return ""

    text = transliterate_to_ascii(raw)
    text = text.lower().strip()

    # Strip legal suffixes before tokenizing
    text = strip_legal_suffixes(text)

    # Remove punctuation (keep alphanumeric + spaces)
    text = PUNCT_RE.sub(" ", text)

    # Expand abbreviations token by token
    tokens = text.split()
    expanded = []
    for tok in tokens:
        tok_clean = tok.strip(".")
        expanded.append(NAME_ABBREV.get(tok_clean, NAME_ABBREV.get(tok, tok)))

    # Remove stopwords and empty tokens
    tokens = [t for t in expanded if t and t not in NAME_STOPWORDS]

    return MULTI_SPACE_RE.sub(" ", " ".join(tokens)).strip()


def normalize_address(raw: str) -> str:
    """
    Full normalization pipeline for addresses:
    1. Transliterate to ASCII
    2. Lowercase
    3. Remove phone numbers (noise in Indian addresses)
    4. Remove punctuation
    5. Expand street type abbreviations
    6. Collapse whitespace
    NOTE: We do NOT remove stopwords from addresses (location words matter)
    NOTE: We do NOT sort tokens — we preserve rough order for context
    """
    if not raw or not isinstance(raw, str):
        return ""

    text = transliterate_to_ascii(raw)
    text = text.lower().strip()

    # Remove phone numbers (common noise in Indian addresses like "Ph. 989, 9487203")
    text = PHONE_RE.sub(" ", text)

    # Remove punctuation
    text = PUNCT_RE.sub(" ", text)

    # Expand address abbreviations token by token
    tokens = text.split()
    expanded = []
    for tok in tokens:
        tok_clean = tok.strip(".")
        expanded.append(ADDR_ABBREV.get(tok_clean, ADDR_ABBREV.get(tok, tok)))

    # Filter empty tokens and 1-char tokens that aren't digits
    tokens = [t for t in expanded if t and (len(t) > 1 or t.isdigit())]

    return MULTI_SPACE_RE.sub(" ", " ".join(tokens)).strip()


def extract_zip_pin(address: str) -> str:
    """Extract US ZIP (5 digit) or India PIN (6 digit) from normalized address."""
    if not address:
        return ""
    # US ZIP: 5 digits optionally followed by -4 digits
    us_zip = re.search(r"\b(\d{5})(?:-\d{4})?\b", address)
    if us_zip:
        return us_zip.group(1)
    # India PIN: 6 digits
    india_pin = re.search(r"\b(\d{6})\b", address)
    if india_pin:
        return india_pin.group(1)
    return ""


def extract_name_tokens(norm_name: str, n: int = 3) -> str:
    """Return first N significant tokens of normalized name for blocking."""
    tokens = norm_name.split()
    return " ".join(tokens[:n])


def extract_name_suffix_tokens(norm_name: str, n: int = 2) -> str:
    """Return last N significant tokens of normalized name (for transposition blocking)."""
    tokens = norm_name.split()
    return " ".join(tokens[-n:]) if len(tokens) >= n else " ".join(tokens)


# ─── Process a single DataFrame ──────────────────────────────────────────────

def process_df(df: pl.DataFrame, label: str) -> pl.DataFrame:
    """Apply normalization to a source DataFrame, add derived columns."""
    print(f"  Processing {label} ({len(df):,} rows) …")
    t0 = time.time()

    # Convert to list for row-wise processing (faster than polars map for complex logic)
    names    = df["business_name"].to_list()
    addrs    = df["business_address"].to_list()
    countries = df["country"].to_list()

    norm_names  = []
    norm_addrs  = []
    zip_pins    = []
    name_prefix = []
    name_suffix = []

    for name, addr, country in zip(names, addrs, countries):
        n = normalize_name(name or "")
        a = normalize_address(addr or "")
        z = extract_zip_pin(a)
        norm_names.append(n)
        norm_addrs.append(a)
        zip_pins.append(z)
        name_prefix.append(extract_name_tokens(n, n=3))
        name_suffix.append(extract_name_suffix_tokens(n, n=2))

    result = df.with_columns([
        pl.Series("norm_name",    norm_names),
        pl.Series("norm_address", norm_addrs),
        pl.Series("zip_pin",      zip_pins),
        pl.Series("name_prefix",  name_prefix),
        pl.Series("name_suffix",  name_suffix),
    ])

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Amazon ML Challenge 2026 — Phase 2: Normalization")
    print("=" * 65)

    files = [
        (TRAIN / "train_source1.tsv", TRAIN_OUT / "norm_source1.tsv", "train_source1"),
        (TRAIN / "train_source2.tsv", TRAIN_OUT / "norm_source2.tsv", "train_source2"),
        (TRAIN / "train_source3.tsv", TRAIN_OUT / "norm_source3.tsv", "train_source3"),
        (TEST  / "test_source1.tsv",  TEST_OUT  / "norm_source1.tsv", "test_source1"),
        (TEST  / "test_source2.tsv",  TEST_OUT  / "norm_source2.tsv", "test_source2"),
        (TEST  / "test_source3.tsv",  TEST_OUT  / "norm_source3.tsv", "test_source3"),
    ]

    stats = []

    for src_path, dst_path, label in files:
        print(f"\n[{label}]")
        df = pl.read_csv(src_path, separator="\t", infer_schema_length=0, ignore_errors=True)
        norm_df = process_df(df, label)

        # Save normalized TSV
        norm_df.write_csv(dst_path, separator="\t")
        print(f"  Saved -> {dst_path}")

        # Spot-check stats
        zip_count = sum(1 for z in norm_df["zip_pin"].to_list() if z)
        addr_null = norm_df.filter(pl.col("business_address").is_null()).height
        stats.append({
            "file": label,
            "rows": len(norm_df),
            "zip_extracted": zip_count,
            "zip_pct": round(zip_count / len(norm_df) * 100, 2),
            "addr_null": addr_null,
        })

    # ── Print sample normalized pairs ─────────────────────────────
    print("\n" + "=" * 65)
    print("  SAMPLE NORMALIZATION OUTPUT")
    print("=" * 65)

    sample_src = pl.read_csv(
        TRAIN / "train_source1.tsv",
        separator="\t", infer_schema_length=0, ignore_errors=True
    ).head(5)
    sample_norm = pl.read_csv(
        TRAIN_OUT / "norm_source1.tsv",
        separator="\t", infer_schema_length=0, ignore_errors=True
    ).head(5)

    for i, (raw_row, norm_row) in enumerate(
        zip(sample_src.iter_rows(named=True), sample_norm.iter_rows(named=True))
    ):
        print(f"\n  [{i+1}] {raw_row['entity_id']}")
        print(f"    RAW  name : {raw_row['business_name']}")
        print(f"    NORM name : {norm_row['norm_name']}")
        print(f"    RAW  addr : {raw_row['business_address']}")
        print(f"    NORM addr : {norm_row['norm_address']}")
        print(f"    ZIP/PIN   : {norm_row['zip_pin'] or '(none)'}")
        print(f"    Prefix    : {norm_row['name_prefix']}")

    # ── Final stats ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  NORMALIZATION SUMMARY")
    print("=" * 65)
    print(f"\n  {'File':<20} {'Rows':>10} {'ZIP extracted':>15} {'ZIP%':>8} {'Addr null':>10}")
    print("  " + "-" * 68)
    for s in stats:
        print(f"  {s['file']:<20} {s['rows']:>10,} {s['zip_extracted']:>15,} "
              f"{s['zip_pct']:>7.2f}% {s['addr_null']:>10,}")

    print("\n  All normalized files saved.")
    print("  Next: Phase 3 -- Blocking / Candidate Generation")


if __name__ == "__main__":
    main()
