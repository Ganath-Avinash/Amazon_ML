# 🏆 Amazon ML Challenge 2026 — Complete Analysis & Approach Guide

## 📌 Problem: Business Entity Resolution (ER)

> **Core Task:** Given business records from 3 noisy, independent data sources with no shared identifiers, determine which records across sources refer to the same **real-world business entity**.

---

## 📊 Dataset Snapshot (from EDA)

| File | Rows | Notes |
|---|---|---|
| `train_source1.tsv` | **2,206,821** | Reference source (deduplicated) |
| `train_source2.tsv` | **5,034,616** | Noisy, ~169K missing addresses |
| `train_source3.tsv` | **5,285,603** | Noisy, ~176K missing addresses |
| `train_ground_truth.tsv` | 2,206,821 | One row per S1 entity |
| `test_source1.tsv` | ~1.75M | Generate matches for ALL rows |
| `test_source2.tsv` | ~5M | — |
| `test_source3.tsv` | ~5M | — |

### 📈 Ground Truth Statistics
| Stat | Value |
|---|---|
| Countries in train | `US`, `India` |
| Countries in test | `US`, `India`, **`France`** (unseen!) |
| Average matches per S1 entity | **3.46** (min 0, max 11) |
| Singletons (0 matches) | **123,247** entities |
| Match count (p50) | 3 |
| Match count (p75) | 5 |

### ⚠️ Key Data Observations
- Source2 & Source3 have **Hindi/Indic characters** rendered as `???` (encoding issue)
- Addresses missing in 3–4% of S2/S3 records
- Address field order is **inconsistent** (e.g., `19 1/2 STARDUST TRAIL, GREENSBORO, NC` vs `GREENSBORO, NC, 19 1/2 STARDUST TRAIL`)
- Business names may include URLs (`wilfordhancock.com`), abbreviations, DBA names, transpositions

---

## 🎯 Metric: F₀.₅ Score (Precision-Heavy)

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- **Precision is weighted 2× over Recall**
- Computed as **macro-average** across ALL Source 1 entities
- **Singletons score 1.0 when correctly predicted empty, 0.0 when you predict any match**
- → False merges hurt more than missed links → **be conservative**

---

## 🏗️ Solution Architecture (Two-Stage Pipeline)

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: BLOCKING / CANDIDATE GENERATION               │
│  → Reduce ~(2.2M × 10M) = 22 TRILLION comparisons      │
│    to a manageable candidate set per S1 entity          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Stage 2: MATCHING MODEL                                │
│  → Score each (S1, S2/S3) candidate pair                │
│  → Threshold to produce final matches                   │
└─────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> `candidate_pairs.tsv` is **reviewed for ranking** beyond the leaderboard score. A **smaller, high-recall** candidate set = better ranking. Blocking efficiency matters!

---

## 🧱 Stage 1: Blocking Strategies

The goal: high recall (don't miss true matches), maximum reduction.

### Strategy A — Multi-Key Inverted Index (Fast & Effective)
```
blocking_key = normalize(country) + top_tokens(business_name) + zip/PIN
```

**Steps:**
1. **Text normalization** on business_name and business_address:
   - Lowercase, strip punctuation
   - Expand abbreviations: `rd→road`, `st→street`, `corp→corporation`, `pvt→private`, `ltd→limited`
   - Remove legal suffixes: `llc`, `inc`, `pvt ltd`, etc.
   - Remove stopwords: `the`, `a`, `and`
2. **Extract blocking keys (multi-pass):**
   - **Key 1:** Country + first two tokens of normalized name
   - **Key 2:** Country + last 2 tokens of normalized name (handles transpositions)
   - **Key 3:** Country + 5-digit ZIP or 6-digit PIN code
   - **Key 4:** Phonetic encoding (Metaphone/Soundex) of first name token
   - **Key 5:** TF-IDF top-3 name tokens (character n-gram)
3. Union all S2/S3 records sharing **any blocking key** with a given S1 record

### Strategy B — BM25 / TF-IDF ANN (Scalable Approximate Search)
- Build TF-IDF (char 2–4 grams) index over all S2+S3 business names
- For each S1 entity, retrieve top-K candidates by BM25 similarity
- Tool: `rank_bm25`, `sklearn TfidfVectorizer + sparse matrix`, or `faiss`

### Strategy C — Embedding-Based Recall Boost (Optional)
- Encode `business_name + " " + business_address` with a lightweight sentence encoder (e.g., `intfloat/e5-small`, `sentence-transformers/paraphrase-multilingual-MiniLM`)
- FAISS `IndexFlatIP` or HNSW for approximate nearest neighbor retrieval
- **Must be ≤8B parameters** (fine-tunable MIT/Apache 2.0 model)

> [!TIP]
> Combine A + B for maximum recall with reasonable candidate set size. Target < 50–100 candidates per S1 entity.

---

## 🤖 Stage 2: Matching Model

### Feature Engineering (Per Candidate Pair)

**Name Features:**
| Feature | Method |
|---|---|
| Character Jaccard | Jaccard on char 3-grams |
| Token Overlap | |intersection| / |union| of name tokens |
| Levenshtein ratio | `rapidfuzz.ratio()` |
| Jaro-Winkler | For prefix-heavy matches |
| Phonetic Match | Metaphone, Soundex encoding |
| Token Sort Ratio | Handle word-order transpositions |
| TF-IDF cosine | Between name strings |
| Same prefix tokens | How many first-N tokens match |

**Address Features:**
| Feature | Method |
|---|---|
| ZIP/PIN match | Exact match of numeric postal code |
| City token match | City name token overlap |
| State match | State abbreviation match |
| Street number match | Numeric house/street number extracted |
| Full address edit distance | Levenshtein on normalized full address |
| Address token Jaccard | Token overlap of full address |
| Shared rare tokens | Rare words (low IDF) in common |

**Meta Features:**
| Feature | Method |
|---|---|
| Country match | Exact string match |
| Name length ratio | `min/max` of name char lengths |
| Address availability | Boolean: both have address? |

### Model Options

**Option 1 — Gradient Boosted Trees (XGBoost/LightGBM) ✅ Recommended**
- Train on positive pairs from ground truth + negative pairs sampled from candidates
- Negative sampling ratio: ~10:1 (negatives:positives)
- Threshold tuned for F_0.5 on held-out validation fold
- **Pros:** Fast, interpretable, no GPU needed

**Option 2 — Siamese/Cross-Encoder Neural Network**
- Input: `[CLS] name1 [SEP] addr1 [SEP] name2 [SEP] addr2`
- Use `xlm-roberta-base` (multilingual, MIT-like license, 280M params) or `intfloat/multilingual-e5-base`
- Fine-tune as binary classifier on candidate pairs
- **Pros:** Better at semantic similarity, handles multilingual content

**Option 3 — Hybrid (Recommended for best score)**
1. XGBoost on string features for speed
2. Reranking with cross-encoder on top candidates
3. Ensemble via soft voting

---

## 🔄 Complete Pipeline

```python
# Pseudocode outline
# --- BLOCKING ---
s1 = load_tsv("train_source1.tsv")
s2 = load_tsv("train_source2.tsv")
s3 = load_tsv("train_source3.tsv")

s2_s3 = concat(s2, s3)
s2_s3_normalized = normalize_names_and_addresses(s2_s3)
blocking_index = build_inverted_index(s2_s3_normalized, keys=["country+name_tokens", "zip", "phonetic"])

candidates = {}
for entity in s1:
    keys = extract_blocking_keys(entity)
    candidates[entity.id] = blocking_index.lookup(keys)

save_tsv(candidates, "candidate_pairs.tsv")

# --- FEATURE EXTRACTION ---
pairs = [(s1_row, s2s3_row) for s1_id, cand_ids in candidates.items()
                             for s2s3_row in lookup(cand_ids)]

features = [extract_features(p) for p in pairs]

# --- TRAINING ---
positives = load_ground_truth()
labels = [1 if pair in positives else 0 for pair in pairs]

model = XGBClassifier().fit(features, labels)

# --- INFERENCE ---
scores = model.predict_proba(features)[:, 1]
threshold = tune_threshold_for_f05(scores, labels_val)

matches = {s1_id: [cand for cand, score in zip(cands, scores) if score > threshold]
           for s1_id, cands in candidates.items()}

save_tsv(matches, "matching_results.tsv")
```

---

## ⚡ Scaling Considerations

| Challenge | Solution |
|---|---|
| 2.2M × 10M = 22T comparisons | Blocking reduces to ~50–100 candidates per S1 |
| Large file sizes (200–500 MB each) | Read in chunks, use Polars or Dask |
| Multilingual names (Hindi, French) | Use `unidecode` or `ftfy` for normalization; multilingual models |
| Unseen country (France in test) | Country-agnostic features; don't one-hot encode |
| Missing addresses | Fall back to name-only blocking for those records |

---

## 📋 Step-by-Step Action Plan

```
Week 1:
  ✅ EDA — understand noise patterns, distribution
  ✅ Build normalization pipeline (abbrevs, punctuation, stopwords)
  ✅ Implement multi-key inverted index blocking
  ✅ Measure blocking recall on train GT

Week 2:
  ✅ Feature engineering (name + address similarity features)
  ✅ Train XGBoost / LightGBM classifier
  ✅ Tune threshold on validation split for F_0.5
  ✅ Validate output format with validate_submission.py

Week 3:
  ✅ Add BM25 / TF-IDF ANN blocking for recall boost
  ✅ Optionally: fine-tune multilingual sentence encoder
  ✅ Ensemble or cross-encoder reranking
  ✅ Optimize candidate set size (fewer candidates → better ranking)
  ✅ Final submission + documentation
```

---

## ⚠️ Critical Constraints

| Constraint | Details |
|---|---|
| **Model license** | MIT or Apache 2.0 only |
| **Model size** | ≤ 8 Billion parameters |
| **No external data** | No geocoding APIs, business registries, internet lookup |
| **Singletons** | Must predict empty list for no-match entities |
| **Output format** | TSV, exact column names, no duplicates |
| **Submission files** | Both `matching_results.tsv` AND `candidate_pairs.tsv` |

---

## 💡 Key Insights & Winning Tips

1. **Blocking is the bottleneck** — If a true match isn't in your candidate set, you *can't* recover it. Target >95% recall in blocking.
2. **F_0.5 penalizes false merges** → When in doubt, **don't match**. A conservative threshold wins.
3. **Singletons are free points** — 123K entities with no matches. Predict empty → 1.0 per entity.
4. **Country is not enough for blocking** — Use multi-key blocking; a single key will miss too many true matches.
5. **Address component extraction** — Separately extract: house number, street, city, state, ZIP. Even partial overlap is a strong signal.
6. **Unseen France** — Use language-agnostic features (edit distances, n-gram overlap). Don't rely on language-specific rules.
7. **Character n-grams beat word tokens** for fuzzy name matching (handles abbreviations, typos).

---

## 🧰 Recommended Stack

```
pandas / polars       → Data loading and manipulation
rapidfuzz             → Fast Levenshtein, Jaro-Winkler, token sort ratio
jellyfish             → Soundex, Metaphone phonetic encoding
scikit-learn          → TF-IDF vectorizer, cosine similarity
xgboost / lightgbm   → Matching classifier
faiss-cpu             → ANN search for embedding-based blocking
sentence-transformers → Multilingual embeddings (optional)
rank_bm25             → BM25 blocking
unidecode / ftfy      → Unicode normalization
```
