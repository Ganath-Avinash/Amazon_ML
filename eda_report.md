# Phase 1 EDA Report — Amazon ML Challenge 2026
## Business Entity Resolution

---

## 1. Dataset Shapes

| File | Rows | Cols |
|---|---|---|
| `train_source1.tsv` | **2,206,821** | 4 |
| `train_source2.tsv` | **5,034,616** | 4 |
| `train_source3.tsv` | **5,285,603** | 4 |
| `train_ground_truth.tsv` | **2,206,821** | 2 |
| `test_source1.tsv` | **1,732,544** | 4 |
| `test_source2.tsv` | **4,887,273** | 4 |
| `test_source3.tsv` | **5,082,316** | 4 |

**Total records across all files: ~24.4 million**

All source files share the same 4 columns: `entity_id`, `business_name`, `business_address`, `country`.

---

## 2. Null Rates

| Source | `entity_id` | `business_name` | `business_address` | `country` |
|---|---|---|---|---|
| train_source1 | 0% | 0% | **0%** | 0% |
| train_source2 | 0% | 0% | **3.36%** | 0% |
| train_source3 | 0% | 0% | **3.33%** | 0% |
| test_source1 | 0% | 0% | **0%** | 0% |

> [!NOTE]
> Source 1 (reference) is clean — no nulls anywhere. Source 2 and 3 each have ~3.3% missing addresses (~169K and ~176K rows respectively). For these records, blocking must fall back to **name-only keys**.

**Ground truth null rate:** `matched_entity_ids` is null in **5.58%** of rows — these are the true singletons (entities in S1 with no match in S2/S3).

---

## 3. Country Distribution

### Train Set
| Country | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| US | 1,323,633 (59.97%) | 3,016,817 (59.92%) | 3,170,056 (59.96%) |
| India | 883,188 (40.03%) | 2,017,799 (40.08%) | 2,115,547 (40.04%) |

### Test Set
| Country | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| India | 809,986 (46.75%) | 2,312,565 (47.32%) | 2,405,000 (47.32%) |
| US | 663,106 (38.27%) | 1,871,330 (38.29%) | 1,945,701 (38.29%) |
| **France** | **259,452 (14.97%)** | **703,378 (14.39%)** | **731,615 (14.39%)** |

> [!IMPORTANT]
> **France is ~15% of the test set** (~259K S1 entities, ~703K S2, ~731K S3) — a large unseen country. Pipeline must be completely language-agnostic. Do NOT one-hot encode country or use language-specific rules.

---

## 4. Ground Truth Match Statistics

| Metric | Value |
|---|---|
| Total S1 entities (train) | 2,206,821 |
| **Singletons (0 matches)** | **0 (!!)** |
| Mean matches per S1 | **3.666** |
| Median matches | **4** |
| Min matches | 1 |
| Max matches | 11 |
| p25 | 3 |
| p75 | 5 |
| p95 | 6 |

> [!IMPORTANT]
> **There are ZERO singletons in the training set.** Every S1 entity in the training ground truth has at least 1 match. This means the 5.58% null rate in `matched_entity_ids` was from blank strings, not true singletons. However, the test set will contain singletons — the analysis doc says ~123K. Singleton handling must be implemented carefully.

### Match Count Distribution

| Matches | Count | % of S1 |
|---|---|---|
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.93% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10 | 534 | 0.02% |
| 11 | 37 | <0.01% |

The distribution peaks at **3 matches** and is right-skewed. The vast majority (83%) of S1 entities have 2–5 matches.

---

## 5. Encoding Issues (`???` Corruption)

| Source | `business_name` | `business_address` |
|---|---|---|
| train_source1 | 0 rows | 3 rows |
| train_source2 | 0 rows | 6 rows |
| train_source3 | 0 rows | 6 rows |

> [!NOTE]
> The `???` literal corruption is extremely rare (near-zero). The actual multilingual noise comes in a different form — **native Unicode scripts** (Tamil, Hindi/Devanagari) are preserved in the data. The real challenge is matching Latin-script S1 names against native-script S2/S3 names. See sample pairs below.

---

## 6. Address & ZIP/PIN Coverage

| Source | Total Records | Address Null % | Has ZIP/PIN | ZIP Coverage % |
|---|---|---|---|---|
| train_source1 | 2,206,821 | 0.0% | 147,257 | **6.67%** |
| train_source2 | 5,034,616 | 3.36% | 369,140 | **7.59%** |
| train_source3 | 5,285,603 | 3.33% | 385,727 | **7.55%** |
| test_source1 | 1,732,544 | 0.0% | 75,634 | **4.37%** |

> [!WARNING]
> **ZIP/PIN codes are present in only ~6–8% of records.** This is too sparse to use as a primary blocking key. It works well as a **secondary signal** to boost precision when present, but the main blocking strategy must rely on **name tokens**. Address-based blocking alone will miss ~93% of records.

---

## 7. Sample Noisy Matched Pairs

These are real matched pairs from the training ground truth — revealing exactly what noise the model must handle:

### Case 1 — Typo in name, missing address in S2
| | ID | Business Name | Address |
|---|---|---|---|
| S1 | S1-965667 | Maure **Williams** Colombier Inc | 85 Wayne Avenue, Ticonderoga, NY |
| S2 | S2-681193310 | Maure **Wilblims** Colombier Inc | *(null)* |
| S2 | S2-743505751 | Maure Williams Colombier *(no Inc)* | *(null)* |

**Noise type:** Character transposition typo + missing address + dropped legal suffix.

---

### Case 2 — Native script (Tamil) in S2, same address
| | ID | Business Name | Address |
|---|---|---|---|
| S1 | S1-55344266 | Raj Investments LLP | 6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu |
| S2 | S2-249013014 | ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி *(Tamil)* | 6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu |
| S2 | S2-197070651 | Raj Investments LLP | 6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu |

**Noise type:** Native script transliteration. Name is in Tamil Unicode, address is similar but in ALLCAPS. Address overlap is the key signal here.

---

### Case 3 — Truncated name + reordered address components
| | ID | Business Name | Address |
|---|---|---|---|
| S1 | S1-343815751 | Dahlia Power Reliable **Scientific LLC** | 630 45th Terrace, Kansas City, MO |
| S2 | S2-790675320 | Dahlia Power Reliable *(truncated)* | **KANSAS CITY, MO, 630 45ND TERRACE**, null |
| S2 | S2-479876582 | Dahlia Power Reliable Scientific *(no LLC)* | **45ND TERRACE, null, KANSAS CITY, MO** |

**Noise type:** Truncated name, dropped legal suffix, address component order scrambled, street number suffix changed (45th → 45ND).

---

### Case 4 — Hindi script + reordered address
| | ID | Business Name | Address |
|---|---|---|---|
| S1 | S1-656753428 | Ss Food Private Limited | Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh |
| S2 | S2-153058913 | एसएस फूड प्राइवेट लिमिटेड *(Hindi)* | AF-0684, NANDGRAM NEAR MOTHER INDIA PUBLIC SCHOOL. PH. 989, GHAZIABAD, 9487203, उत्तर प्रदेश |
| S2 | S2-24659151 | एसएस फूड प्राइवेट लिमिटेड *(Hindi)* | AF-0684, Uttar Pradesh, GHAZIABAD, 9487203 |

**Noise type:** Hindi transliteration + address components scrambled + partial address in S2.

---

### Case 5 — Accent characters + address abbreviation
| | ID | Business Name | Address |
|---|---|---|---|
| S1 | S1-102811957 | Payne Enterprises | 3315 Fremont Street, Peoria, IL |
| S2 | S2-478959098 | Payne **É**nterprises *(accent)* | 3315 FREMONT **ST**, PEORIA, IL |
| S2 | S2-553508714 | Payne **Enterpires** *(typo)* | 3315 FREMONT **ST**, PEORIA, IL |

**Noise type:** Accent diacritic substitution, street type abbreviation (Street → ST), character swap typo.

---

## 8. Key Takeaways for the Pipeline

| Finding | Implication |
|---|---|
| **3.3% missing addresses in S2/S3** | Blocking must work name-only when address is null |
| **ZIP/PIN present only in ~7% of records** | Don't rely on ZIP as primary blocking key |
| **0 singletons in train, but ~5.58% null GT** | GT nulls = blank strings; test will have real singletons — handle both |
| **France = 15% of test, absent from train** | No language-specific rules; use char n-gram features only |
| **Tamil & Hindi scripts in S2/S3** | Address is the best matching signal when scripts differ; use Unicode normalization carefully |
| **Address component order varies widely** | Use token-set/bag-of-words features, not sequential edit distance alone |
| **Legal suffixes dropped frequently** | Strip LLC, Inc, LLP, Pvt Ltd, etc. during normalization |
| **Typos & character swaps common** | Levenshtein + Jaro-Winkler are essential features |
| **Mean 3.67 matches per entity** | Each S1 entity should generate ~50–100 candidates in blocking |
