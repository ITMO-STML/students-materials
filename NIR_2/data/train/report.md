# Train Data Build — Ablation Grid

- Cap negatives per record: **3**
- Train queries: **99**, test queries: **199**, unused: 1798
- Corpus size: **1987** docs
- Test set written: **199** queries

## Split distribution (по primary tag)

| Tag | Train | Test |
|---|---:|---:|
| 0-hop | 1 | 1 |
| 1-hop | 72 | 145 |
| duration | 0 | 1 |
| exclusion | 1 | 1 |
| multi-constraint | 13 | 26 |
| multi-hop | 2 | 5 |
| no-answer | 9 | 17 |
| qualifier-constraint | 0 | 1 |
| ranking | 0 | 1 |
| reverse | 1 | 1 |

## Raw inputs

- CN passed candidates loaded: **138**
- BM25 passed candidates loaded: **500**
- GP passed candidates loaded: **50**

## Coverage по train queries (на 500)

- Train qids с CN negatives: **76** (76.8%)
- Train qids с BM25 negatives: **96** (97.0%)
- Train qids с GP positives: **19** (19.2%)
- Train qids с CN ∩ GP: **18** (18.2%)

## Conditions

| Condition | Records | Unique qids | qids with neg | Mean neg/rec | Mean variants/qid |
|---|---:|---:|---:|---:|---:|
| A | 99 | 99 | 0 (0.0%) | 0 | 1 |
| B | 99 | 99 | 96 (97.0%) | 2.91 | 1 |
| C | 99 | 99 | 76 (76.8%) | 1.29 | 1 |
| D | 146 | 99 | 0 (0.0%) | 0 | 1.47 |
| E | 146 | 99 | 76 (76.8%) | 1.33 | 1.47 |

## Missing qids per condition (must be empty)

- (все condition files покрывают все train qids ✓)

---
## Примеры записей (по одной из каждого condition)

### test.jsonl (первая запись)

- qid: `rubq_123`, gold_qid: `rubq_123`
- query: На каком острове жила древнегреческая поэтесса Сапфо?
- d⁺: Сапфо́ (тж. Сафо́, Са́фо, Сафо Митиленская; аттич. др.-греч. Σαπφώ (произносится — /sapːʰɔː/), эолийск. др.-греч. Ψάπφω (произносится — /psapːʰɔː/); около 630 г. до н. э., о-в Лесб…
