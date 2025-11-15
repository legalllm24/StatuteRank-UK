# StatuteRank-UK — Reproducing Results

This folder intentionally does not contain raw TSV result files.

All reranking metrics reported in the paper (Tables 2 and 3) can be
fully reproduced using the scripts in `scripts/` with the provided
corpus and the 100-query evaluation benchmark.

---

## 1. Dense-Pool Reranking (Table 2)

### Step 1 — Build dense candidate pool
python scripts/build_dense_candidates.py \
  --data_json data/corpus_uklegislation.json \
  --eval_jsonl data/eval_statutory_100queries_approved.jsonl \
  --out_jsonl candidates_dense_mpnet.jsonl \
  --k 100

### Step 2 — Run any reranker
Example:
python scripts/eval_reranker.py \
  --model_type hf-seqcls \
  --model_name amal1994/distilled-voyage-modernbert \
  --candidates candidates_dense_mpnet.jsonl \
  --out_tsv results_dense_distilledmodernbert.tsv

---

## 2. BM25-Pool Reranking (Table 3)

### Step 1 — Build BM25 candidate pool:
python scripts/build_bm25_candidates_py.py \
  --data_json data/corpus_uklegislation.json \
  --eval_jsonl data/eval_statutory_100queries_approved.jsonl \
  --out_jsonl candidates_bm25_py.jsonl \
  --k 100

### Step 2 — Run rerankers (same as above)


