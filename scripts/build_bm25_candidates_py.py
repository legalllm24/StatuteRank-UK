import json
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi


def simple_tokenize(text: str):
    # Simple, reproducible tokenizer: lowercase + split on whitespace.
    # You can replace this with something fancier later if you want.
    return text.lower().split()


def load_corpus(data_json: Path):
    print(f"Loading corpus from {data_json}")
    with data_json.open("r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} chunks")

    chunk_ids = []
    docs = []
    for item in corpus:
        meta = item.get("metadata", {})
        chunk_id = meta.get("chunk_id") or meta.get("doc_id")
        text = item.get("content", "").strip()
        if not chunk_id or not text:
            continue
        chunk_ids.append(chunk_id)
        docs.append(text)

    print(f"Using {len(docs)} chunks after filtering")
    return chunk_ids, docs


def load_eval_queries(eval_jsonl: Path):
    print(f"Loading eval queries from {eval_jsonl}")
    eval_examples = []
    with eval_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            eval_examples.append(ex)
    print(f"Loaded {len(eval_examples)} eval queries")
    return eval_examples


def extract_relevant(ex):
    # If your eval JSONL already has "relevant", just return it.
    if "relevant" in ex:
        return ex["relevant"]

    # Otherwise convert "answers" -> "relevant"
    rels = []
    for ans in ex.get("answers", []):
        cid = ans.get("chunk_id")
        rel = ans.get("rel", 1)
        if cid is None:
            continue
        rels.append({"chunk_id": cid, "rel": rel})
    return rels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_json",
        type=str,
        default="/home/amal/Desktop/Legal RAG content/Data.json",
        help="Path to Data.json corpus",
    )
    parser.add_argument(
        "--eval_jsonl",
        type=str,
        default="/home/amal/Desktop/eval_build_outputs/queries_enriched_20251014-131057_APPROVED.jsonl",
        help="Eval queries JSONL",
    )
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default="candidates_bm25_py.jsonl",
        help="Output candidates JSONL",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of BM25 candidates per query",
    )
    args = parser.parse_args()

    data_path = Path(args.data_json)
    eval_path = Path(args.eval_jsonl)
    out_path = Path(args.out_jsonl)

    # 1) Load corpus and build BM25 index in memory
    chunk_ids, docs = load_corpus(data_path)
    print("Tokenizing corpus for BM25...")
    tokenized_corpus = [simple_tokenize(d) for d in tqdm(docs, desc="Tokenizing")]
    print("Building BM25Okapi index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # 2) Load eval queries
    eval_examples = load_eval_queries(eval_path)

    # 3) For each query, get top-k BM25 docs
    print(f"Retrieving top-{args.k} BM25 candidates per query...")
    with out_path.open("w", encoding="utf-8") as out_f:
        for ex in tqdm(eval_examples, desc="BM25 retrieval"):
            qid = ex["qid"]
            query = ex["query"]
            relevant = extract_relevant(ex)

            q_tokens = simple_tokenize(query)
            scores = bm25.get_scores(q_tokens)  # numpy array of len = num_docs

            # Get indices of top-k scores
            if args.k >= len(scores):
                top_idx = np.argsort(scores)[::-1]
            else:
                top_idx = np.argpartition(scores, -args.k)[-args.k:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

            candidates = [
                {
                    "chunk_id": chunk_ids[i],
                    "score": float(scores[i]),
                }
                for i in top_idx
            ]

            out_ex = {
                "qid": qid,
                "query": query,
                "candidates": candidates,
                "relevant": relevant,
            }
            out_f.write(json.dumps(out_ex) + "\n")

    print(f"Wrote BM25 candidates to {out_path}")


if __name__ == "__main__":
    main()
