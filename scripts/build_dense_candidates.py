import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ====== CONFIG ======
DATA_PATH = Path("/home/amal/Desktop/Legal RAG content/Data.json")
EVAL_PATH = Path("/home/amal/Desktop/eval_build_outputs/queries_enriched_20251014-131057_APPROVED.jsonl")
OUT_PATH = Path("candidates_dense_mpnet.jsonl")
TOP_K = 100
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
# =====================


def load_corpus(path):
    print(f"Loading corpus from {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunk_ids = []
    texts = []
    id2text = {}

    for item in data:
        meta = item["metadata"]
        cid = meta["chunk_id"]
        text = item["content"]
        chunk_ids.append(cid)
        texts.append(text)
        id2text[cid] = text

    print(f"Loaded {len(chunk_ids)} chunks")
    return chunk_ids, texts, id2text


def build_index(texts, model_name):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    batch_size = 64
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding corpus"):
        batch = texts[i:i + batch_size]
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embs.append(emb)

    embs = np.vstack(embs).astype("float32")
    dim = embs.shape[1]
    print("Embeddings shape:", embs.shape)

    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    print("FAISS index built with", index.ntotal, "vectors")
    return index, embs


def load_eval(path):
    print(f"Loading eval set from {path} (JSONL)")
    eval_examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            qid = ex["qid"]
            query = ex["query"]
            answers = ex["answers"]
            relevant = [{"chunk_id": a["chunk_id"], "rel": a.get("rel", 1)} for a in answers]
            eval_examples.append({"qid": qid, "query": query, "relevant": relevant})
    print(f"Loaded {len(eval_examples)} eval queries")
    return eval_examples


def main():
    chunk_ids, texts, id2text = load_corpus(DATA_PATH)
    eval_examples = load_eval(EVAL_PATH)

    index, _ = build_index(texts, EMB_MODEL)
    model = SentenceTransformer(EMB_MODEL)

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for ex in tqdm(eval_examples, desc="Building candidates"):
            qid = ex["qid"]
            query = ex["query"]
            relevant = ex["relevant"]

            q_emb = model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

            scores, idxs = index.search(q_emb, TOP_K)
            scores = scores[0]
            idxs = idxs[0]

            candidates = []
            for s, i in zip(scores, idxs):
                cid = chunk_ids[i]
                candidates.append({"chunk_id": cid, "score": float(s)})

            record = {
                "qid": qid,
                "query": query,
                "candidates": candidates,
                "relevant": relevant,
            }
            out_f.write(json.dumps(record) + "\n")

    print(f"Wrote candidates to {OUT_PATH}")


if __name__ == "__main__":
    main()
