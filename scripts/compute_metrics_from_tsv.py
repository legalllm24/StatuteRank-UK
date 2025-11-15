import argparse
import math
from collections import defaultdict

def compute_metrics(qid_to_ranked, ks=(5, 10, 25, 50, 100)):
    ks = sorted(set(ks))
    n = len(qid_to_ranked)

    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"MRR@{k}": 0.0 for k in ks})
    sums.update({f"nDCG@{k}": 0.0 for k in ks})

    for qid, data in qid_to_ranked.items():
        ranked_ids = data["ranked_ids"]
        rel_map = data["rel_map"]
        relevant_ids = {cid for cid, rel in rel_map.items() if rel > 0}

        # ideal gains for nDCG
        ideal_rels_sorted = sorted(
            [rel for cid, rel in rel_map.items() if rel > 0],
            reverse=True,
        )
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

            # Recall@k (fraction of relevant docs retrieved)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv",
        type=str,
        required=True,
        help="TSV with columns: qid, model, chunk_id, rank, score, rel",
    )
    args = parser.parse_args()

    qid_to_ranked = defaultdict(lambda: {"ranked_ids": [], "rel_map": {}})

    with open(args.tsv, "r", encoding="utf-8") as f:
        header = next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, model, chunk_id, rank_str, score_str, rel_str = line.split("\t")
            rank = int(rank_str)
            rel = int(rel_str)

            entry = qid_to_ranked[qid]
            entry["ranked_ids"].append(chunk_id)
            # store rel if >0; if multiple lines for same chunk_id, last one wins (shouldn't happen)
            if rel > 0:
                entry["rel_map"][chunk_id] = rel
            elif chunk_id not in entry["rel_map"]:
                entry["rel_map"][chunk_id] = 0

    print(f"Loaded {len(qid_to_ranked)} queries from {args.tsv}")

    metrics = compute_metrics(qid_to_ranked, ks=(5, 10, 25, 50, 100))

    print("=== Metrics from TSV ===")
    for name in sorted(metrics.keys()):
        print(f"{name}: {metrics[name]:.4f}")


if __name__ == "__main__":
    main()
