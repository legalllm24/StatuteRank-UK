import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ===================== Metrics =====================

def compute_metrics_at_ks(
    examples: List[Dict],
    ks=(5, 10, 25, 50, 100),
) -> Dict[str, float]:
    """
    examples: each has
      - 'relevant': list of {chunk_id, rel}
      - 'reranked': list of (chunk_id, score), sorted desc
    """
    ks = sorted(set(ks))
    n = len(examples)
    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"MRR@{k}": 0.0 for k in ks})
    sums.update({f"nDCG@{k}": 0.0 for k in ks})

    for ex in examples:
        rel_map = {r["chunk_id"]: r.get("rel", 1) for r in ex["relevant"]}
        ranked_ids = [cid for cid, _ in ex["reranked"]]
        relevant_ids = set(rel_map.keys())

        # Precompute ideal gains per k for nDCG
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

            # Recall@k (how many relevant docs retrieved / total relevant)
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


# ===================== ColBERT-style scorer =====================

class ColBERTScorer:
    """
    ColBERT-style late interaction scorer:
      - encodes query and docs separately
      - L2-normalizes token embeddings
      - score(q, d) = sum_q max_d (q_t · d_t')

    This uses the HF model 'colbert-ir/colbertv2.0' (or similar).
    It is faithful to the ColBERT scoring idea, but does not build a full index:
    we just use it as a reranker over a fixed candidate set.
    """

    def __init__(self, model_name: str, device: str = None, max_length: int = 256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        print(f"Loading ColBERT model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a list of texts into token embeddings + attention mask.
        Returns:
          - embeddings: (B, L, H)
          - attention_mask: (B, L)
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**enc)
            # last_hidden_state: (B, L, H)
            reps = outputs.last_hidden_state

        # Mask out padding tokens
        attn = enc["attention_mask"]  # (B, L)
        return reps, attn

    def score_query_docs(self, query: str, docs: List[str], batch_size: int = 16) -> List[float]:
        """
        Score a single query against a list of docs.
        """
        if not docs:
            return []

        # Encode query once
        q_emb, q_attn = self._encode([query])  # (1, Lq, H), (1, Lq)
        q_emb = F.normalize(q_emb, p=2, dim=-1)  # L2 norm
        q_mask = q_attn.bool()  # (1, Lq)

        scores: List[float] = []

        # Process docs in batches
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            d_emb, d_attn = self._encode(batch_docs)  # (B, Ld, H), (B, Ld)
            d_emb = F.normalize(d_emb, p=2, dim=-1)   # L2 norm
            d_mask = d_attn.bool()                    # (B, Ld)

            # q_emb: (1, Lq, H)
            # d_emb: (B, Ld, H)
            B, Ld, H = d_emb.shape
            _, Lq, _ = q_emb.shape

            # Expand query embeddings to match batch dimension
            q_expanded = q_emb.expand(B, Lq, H)      # (B, Lq, H)
            q_mask_exp = q_mask.expand(B, Lq)        # (B, Lq)

            # Compute similarity: for each doc, a matrix (Lq, Ld)
            # (B, Lq, H) @ (B, H, Ld) -> (B, Lq, Ld)
            sim = torch.matmul(q_expanded, d_emb.transpose(1, 2))  # (B, Lq, Ld)

            # Mask doc paddings: set similarities to very negative for padded tokens
            # d_mask: (B, Ld)
            d_mask_exp = d_mask.unsqueeze(1).expand(B, Lq, Ld)  # (B, Lq, Ld)
            sim = sim.masked_fill(~d_mask_exp, -1e9)

            # For each query token, max over doc tokens -> (B, Lq)
            max_per_q, _ = sim.max(dim=2)  # (B, Lq)

            # Mask padded query tokens
            max_per_q = max_per_q.masked_fill(~q_mask_exp, 0.0)

            # Sum over query tokens -> (B,)
            batch_scores = max_per_q.sum(dim=1)  # (B,)

            scores.extend(batch_scores.detach().cpu().tolist())

        return scores


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="JSONL with candidates (dense or BM25), same format as others",
    )
    parser.add_argument(
        "--data_json",
        type=str,
        default="/home/amal/Desktop/Legal RAG content/Data.json",
        help="Path to Data.json corpus",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="colbert-ir/colbertv2.0",
        help="ColBERT model name from Hugging Face",
    )
    parser.add_argument(
        "--out_tsv",
        type=str,
        default="results_colbert.tsv",
        help="Output TSV file with per-query rankings",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=100,
        help="Max candidates per query to rerank",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max sequence length for ColBERT encoder",
    )
    args = parser.parse_args()

    data_path = Path(args.data_json)
    cand_path = Path(args.candidates)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load corpus: chunk_id -> text
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

    # 3) Load ColBERT model
    scorer = ColBERTScorer(args.model_name, device=device, max_length=args.max_length)

    # 4) Rerank and write TSV
    eval_for_metrics: List[Dict] = []

    with open(args.out_tsv, "w", encoding="utf-8") as out_f:
        out_f.write("qid\tmodel\tchunk_id\trank\tscore\trel\n")

        for ex in tqdm(cand_examples, desc="Reranking (ColBERT)"):
            qid = ex["qid"]
            query = ex["query"]
            candidates = ex["candidates"][: args.max_docs]
            rel_map = {r["chunk_id"]: r.get("rel", 1) for r in ex["relevant"]}

            cand_ids = [c["chunk_id"] for c in candidates]
            docs = [id2text[cid] for cid in cand_ids]

            scores = scorer.score_query_docs(query, docs, batch_size=16)
            scored: List[Tuple[str, float]] = list(zip(cand_ids, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            eval_for_metrics.append(
                {
                    "qid": qid,
                    "relevant": ex["relevant"],
                    "reranked": scored,
                }
            )

            for rank, (cid, s) in enumerate(scored, start=1):
                rel = rel_map.get(cid, 0)
                out_f.write(
                    f"{qid}\t{args.model_name}\t{cid}\t{rank}\t{s}\t{rel}\n"
                )

    # 5) Compute metrics
    metrics = compute_metrics_at_ks(
        eval_for_metrics,
        ks=(5, 10, 25, 50, 100),
    )
    print("=== Metrics (ColBERT) ===")
    for name in sorted(metrics.keys()):
        print(f"{name}: {metrics[name]:.4f}")


if __name__ == "__main__":
    main()
