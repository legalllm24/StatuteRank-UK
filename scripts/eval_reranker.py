import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm

from sentence_transformers import CrossEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)


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

            # Recall@k (graded as “at least one relevant doc in top k” fraction)
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


# ===================== monoT5 scorer =====================

class MonoT5Scorer:
    """
    monoT5 reranker:
      Input: "Query: {q} Document: {d} Relevant:"
      Score: P(first token == 'true')
    """
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading monoT5 model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # token id for "true"
        true_ids = self.tokenizer.encode("true", add_special_tokens=False)
        assert len(true_ids) == 1, "Expected 'true' to map to a single token"
        self.true_id = true_ids[0]

    def score_pairs(
        self,
        queries: List[str],
        docs: List[str],
        batch_size: int = 8,
    ) -> List[float]:
        assert len(queries) == len(docs)
        scores = []
        for i in range(0, len(queries), batch_size):
            batch_q = queries[i:i + batch_size]
            batch_d = docs[i:i + batch_size]

            inputs = [
                f"Query: {q} Document: {d} Relevant:"
                for q, d in zip(batch_q, batch_d)
            ]
            enc = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            decoder_input_ids = torch.full(
                (enc.input_ids.size(0), 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )

            with torch.no_grad():
                outputs = self.model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                logits = outputs.logits[:, 0, :]  # first generated token
                probs = torch.softmax(logits, dim=-1)
                true_probs = probs[:, self.true_id]
                scores.extend(true_probs.detach().cpu().tolist())
        return scores


# ===================== HF seq-class scorer (ModernBERT etc.) =====================

class HFSeqClsScorer:
    """
    Generic scorer for sequence-classification rerankers:
    (query, passage) -> scalar logit score.

    Designed for amal1994/distilled-voyage-modernbert but works for
    any AutoModelForSequenceClassification-based reranker.
    """
    def __init__(self, model_name: str, device: str = None, max_length: int = 2048):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        print(f"Loading HF seq-class model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(
        self,
        queries: List[str],
        docs: List[str],
        batch_size: int = 8,
    ) -> List[float]:
        assert len(queries) == len(docs)
        scores: List[float] = []
        for i in range(0, len(queries), batch_size):
            batch_q = queries[i:i + batch_size]
            batch_d = docs[i:i + batch_size]

            inputs = self.tokenizer(
                batch_q,
                batch_d,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # (batch, 1) or (batch, num_labels)

                if logits.ndim == 2 and logits.size(1) > 1:
                    # e.g., [not relevant, relevant] -> take relevant logit
                    batch_scores = logits[:, -1].detach().cpu().tolist()
                else:
                    # regression or single-logit classifier
                    batch_scores = logits.squeeze(-1).detach().cpu().tolist()

            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)

        return scores


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=str,
        default="candidates_dense_mpnet.jsonl",
        help="JSONL with candidates from dense retriever",
    )
    parser.add_argument(
        "--data_json",
        type=str,
        default="/home/amal/Desktop/Legal RAG content/Data.json",
        help="Path to Data.json corpus",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["cross-encoder", "monot5", "hf-seqcls", "cohere", "voyage"],
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=(
            "Model name, depends on type:\n"
            "- cross-encoder: e.g. cross-encoder/ms-marco-MiniLM-L-6-v2\n"
            "- monot5: e.g. castorini/monot5-base-msmarco\n"
            "- hf-seqcls: e.g. amal1994/distilled-voyage-modernbert\n"
            "- cohere: e.g. rerank-v3.5\n"
            "- voyage: e.g. rerank-2.5 or rerank-2.5-lite"
        ),
    )
    parser.add_argument(
        "--out_tsv",
        type=str,
        default="results.tsv",
        help="Output TSV file with per-query rankings",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=100,
        help="How many candidates per query to rerank (prefix of candidate list)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_json)
    cand_path = Path(args.candidates)

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

    # 3) Load reranker model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # We unify everything as: score_query_docs(query: str, docs: List[str]) -> List[float]
    if args.model_type == "cross-encoder":
        print(f"Loading CrossEncoder: {args.model_name} on {device}")
        model = CrossEncoder(args.model_name, device=device)

        def score_query_docs(query: str, docs: List[str]) -> List[float]:
            pairs = list(zip([query] * len(docs), docs))
            return model.predict(pairs).tolist()

    elif args.model_type == "monot5":
        scorer = MonoT5Scorer(args.model_name, device=device)

        def score_query_docs(query: str, docs: List[str]) -> List[float]:
            return scorer.score_pairs([query] * len(docs), docs)

    elif args.model_type == "hf-seqcls":
        scorer = HFSeqClsScorer(args.model_name, device=device, max_length=2048)

        def score_query_docs(query: str, docs: List[str]) -> List[float]:
            return scorer.score_pairs([query] * len(docs), docs)

    elif args.model_type == "cohere":
        try:
            import cohere  # type: ignore
        except ImportError:
            raise SystemExit(
                "cohere package not installed. Run: pip install cohere"
            )
        print(f"Loading Cohere client (model={args.model_name})")
        co = cohere.ClientV2()  # uses COHERE_API_KEY env var

        def score_query_docs(query: str, docs: List[str]) -> List[float]:
            resp = co.rerank(
                model=args.model_name,
                query=query,
                documents=docs,
                top_n=len(docs),
            )
            scores = [0.0] * len(docs)
            for r in resp.results:
                scores[r.index] = r.relevance_score

            # Trial key: 10 calls/min → sleep generously
            time.sleep(12.0)

            return scores

    elif args.model_type == "voyage":
        try:
            import voyageai  # type: ignore
        except ImportError:
            raise SystemExit(
                "voyageai package not installed. Run: pip install voyageai"
            )
        print(f"Loading Voyage client (model={args.model_name})")
        vo = voyageai.Client()  # uses VOYAGE_API_KEY env var

        def score_query_docs(query: str, docs: List[str]) -> List[float]:
            reranking = vo.rerank(
                query=query,
                documents=docs,
                model=args.model_name,
                top_k=None,
            )
            scores = [0.0] * len(docs)
            for r in reranking.results:
                scores[r.index] = r.relevance_score

            # Avoid TPM rate limits → sleep per query
            time.sleep(10.0)

            return scores

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # 4) Rerank and write TSV
    eval_for_metrics: List[Dict] = []

    with open(args.out_tsv, "w", encoding="utf-8") as out_f:
        out_f.write("qid\tmodel\tchunk_id\trank\tscore\trel\n")

        for ex in tqdm(cand_examples, desc="Reranking"):
            qid = ex["qid"]
            query = ex["query"]
            candidates = ex["candidates"][: args.max_docs]
            rel_map = {r["chunk_id"]: r.get("rel", 1) for r in ex["relevant"]}

            cand_ids = [c["chunk_id"] for c in candidates]
            docs = [id2text[cid] for cid in cand_ids]

            scores = score_query_docs(query, docs)
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

    # 5) Compute metrics at multiple ks
    metrics = compute_metrics_at_ks(
        eval_for_metrics,
        ks=(5, 10, 25, 50, 100),
    )
    print("=== Metrics ===")
    for name in sorted(metrics.keys()):
        print(f"{name}: {metrics[name]:.4f}")


if __name__ == "__main__":
    main()
