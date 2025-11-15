import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ===================== Metrics =====================

def compute_metrics_at_ks(
    examples: List[Dict],
    ks=(5, 10, 25, 50, 100),
) -> Dict[str, float]:
    ks = sorted(set(ks))
    n = len(examples)
    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"MRR@{k}": 0.0 for k in ks})
    sums.update({f"nDCG@{k}": 0.0 for k in ks})

    for ex in examples:
        rel_map = {r["chunk_id"]: r.get("rel", 1) for r in ex["relevant"]}
        ranked_ids = [cid for cid, _ in ex["reranked"]]
        relevant_ids = set(rel_map.keys())

        # Ideal gains for nDCG
        ideal_rels_sorted = sorted(rel_map.values(), reverse=True)
        ideal_prefix_gains = {}
        for k in ks:
            ideal = ideal_rels_sorted[:k]
            if not ideal:
                ideal_prefix_gains[k] = 0.0
            else:
                idcg = 0.0
                for i, rel in enumerate(ideal, start=1):
                    idcg += (2**rel - 1) / math.log2(i + 1)
                ideal_prefix_gains[k] = idcg

        for k in ks:
            top_ids = ranked_ids[:k]

            # Recall@k
            if relevant_ids:
                hits = [cid for cid in top_ids if cid in relevant_ids]
                recall = len(hits) / len(relevant_ids)
            else:
                recall = 0.0
            sums[f"Recall@{k}"] += recall

            # MRR@k
            rr = 0.0
            for rank, cid in enumerate(top_ids, start=1):
                if cid in relevant_ids:
                    rr = 1.0 / rank
                    break
            sums[f"MRR@{k}"] += rr

            # nDCG@k
            dcg = 0.0
            for rank, cid in enumerate(top_ids, start=1):
                rel = rel_map.get(cid, 0)
                if rel > 0:
                    dcg += (2**rel - 1) / math.log2(rank + 1)
            idcg = ideal_prefix_gains[k]
            ndcg = dcg / idcg if idcg > 0 else 0.0
            sums[f"nDCG@{k}"] += ndcg

    metrics = {}
    for name, total in sums.items():
        metrics[name] = total / n
    return metrics


# ===================== RankZephyr reranker =====================

class RankZephyrReranker:
    """
    Listwise reranker using castorini/rank_zephyr_7b_v1_full.
    For each query, we give the model:
      - the query
      - a numbered list of candidate passages (truncated)
    and ask it to return a JSON list of indices in ranked order.
    """

    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading RankZephyr model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    def _build_prompt(self, query: str, docs: List[str]) -> str:
        # Truncate each document text to avoid over-long prompts
        truncated_docs = []
        for d in docs:
            # truncate by characters; you can tighten if needed
            if len(d) > 600:
                truncated_docs.append(d[:600])
            else:
                truncated_docs.append(d)

        doc_block_lines = []
        for i, d in enumerate(truncated_docs, start=1):
            doc_block_lines.append(f"[{i}] {d}")

        doc_block = "\n".join(doc_block_lines)

        prompt = (
            "You are a legal information retrieval assistant. "
            "Your task is to rerank candidate law passages for a user query by relevance.\n\n"
            f"Query:\n{query}\n\n"
            "Candidate passages:\n"
            f"{doc_block}\n\n"
            "Please sort these passages from most relevant to least relevant to the query.\n"
            "Return ONLY a JSON array of the indices (1-based) in the new order, like:\n"
            "[3, 1, 2]\n\n"
            "Answer:\n"
        )
        return prompt

    def _parse_order(self, text: str, num_docs: int) -> List[int]:
        """
        Parse a JSON-like list of indices from the model output.
        We expect something like: [3, 1, 2, ...]
        Fallback: use regex and keep indices in-range and unique.
        """
        # Try to find the first [...] block
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            content = m.group(0)
            try:
                arr = json.loads(content)
                if isinstance(arr, list):
                    indices = [int(x) for x in arr]
                else:
                    indices = []
            except Exception:
                indices = []
        else:
            indices = []

        if not indices:
            # Fallback: extract all integers
            nums = re.findall(r'\d+', text)
            indices = [int(x) for x in nums]

        # Clean: keep within [1, num_docs], unique, preserve order
        seen = set()
        cleaned = []
        for i in indices:
            if 1 <= i <= num_docs and i not in seen:
                seen.add(i)
                cleaned.append(i)

        # If still missing some docs, append remaining in original order
        remaining = [i for i in range(1, num_docs + 1) if i not in seen]
        full_order = cleaned + remaining
        return full_order

    def rerank(self, query: str, docs: List[str]) -> List[int]:
        """
        Returns a permutation of indices [1..len(docs)] in ranked order.
        """
        prompt = self._build_prompt(query, docs)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Generated text after the prompt
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Keep only the part after the prompt to avoid confusion
        answer = generated[len(prompt):]
        order = self._parse_order(answer, len(docs))
        return order


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_json",
        type=str,
        default="/home/amal/Desktop/Legal RAG content/Data.json",
        help="Path to Data.json corpus",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Candidates JSONL (dense or BM25).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="castorini/rank_zephyr_7b_v1_full",
        help="RankZephyr model name from Hugging Face.",
    )
    parser.add_argument(
        "--out_tsv",
        type=str,
        default="results_rankzephyr.tsv",
        help="Output TSV with per-query rankings.",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=100,
        help="Max candidates per query to rerank (e.g., 100).",
    )
    args = parser.parse_args()

    data_path = Path(args.data_json)
    cand_path = Path(args.candidates)

    # 1) Load corpus
    print(f"Loading corpus from {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)
    id2text = {item["metadata"]["chunk_id"]: item["content"] for item in corpus}
    print(f"Corpus chunks: {len(id2text)}")

    # 2) Load candidates
    print(f"Loading candidates from {cand_path}")
    cand_examples = []
    with cand_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            cand_examples.append(ex)
    print(f"Loaded {len(cand_examples)} eval queries")

    # 3) Load RankZephyr
    reranker = RankZephyrReranker(args.model_name)

    eval_for_metrics: List[Dict] = []

    # 4) Rerank and write TSV
    with open(args.out_tsv, "w", encoding="utf-8") as out_f:
        out_f.write("qid\tmodel\tchunk_id\trank\tscore\trel\n")

        for ex in tqdm(cand_examples, desc="Reranking (RankZephyr)"):
            qid = ex["qid"]
            query = ex["query"]
            candidates = ex["candidates"][: args.max_docs]
            rel_map = {r["chunk_id"]: r.get("rel", 1) for r in ex["relevant"]}

            cand_ids = [c["chunk_id"] for c in candidates]
            docs = [id2text[cid] for cid in cand_ids]

            order = reranker.rerank(query, docs)  # 1-based indices
            # Convert to permutation over cand_ids
            # order is e.g. [3,1,2,...], so:
            ordered_ids = [cand_ids[i - 1] for i in order]

            # Assign scores: higher score = more relevant
            # We can use descending ranks as scores, e.g. score = -rank
            scored: List[Tuple[str, float]] = []
            for rank_idx, cid in enumerate(ordered_ids, start=1):
                score = float(len(ordered_ids) - rank_idx + 1)
                scored.append((cid, score))

            eval_for_metrics.append(
                {
                    "qid": qid,
                    "relevant": ex["relevant"],
                    "reranked": scored,
                }
            )

            for rank_idx, (cid, s) in enumerate(scored, start=1):
                rel = rel_map.get(cid, 0)
                out_f.write(
                    f"{qid}\t{args.model_name}\t{cid}\t{rank_idx}\t{s}\t{rel}\n"
                )

    # 5) Metrics
    metrics = compute_metrics_at_ks(
        eval_for_metrics,
        ks=(5, 10, 25, 50, 100),
    )
    print("=== Metrics (RankZephyr) ===")
    for name in sorted(metrics.keys()):
        print(f"{name}: {metrics[name]:.4f}")


if __name__ == "__main__":
    main()
