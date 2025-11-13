#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# ----------------- I/O helpers -----------------
def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

# ----------------- Hashing projection -----------------
def signed_hash_from_ids(term_ids, seed) -> np.ndarray:
    ids = np.asarray(term_ids, dtype=np.uint64)
    s   = np.uint64(seed)
    C1  = np.uint64(0x9E3779B97F4A7C15)
    C2  = np.uint64(0xBF58476D1CE4E5B9)
    C3  = np.uint64(0x94D049BB133111EB)

    x = ids ^ (s * C1)
    x ^= (x >> np.uint64(30)); x *= C2
    x ^= (x >> np.uint64(27)); x *= C3
    x ^= (x >> np.uint64(31))

    return np.where((x & np.uint64(1)) == np.uint64(0), 1.0, -1.0).astype(np.float32)


def project_row_hash(indices: np.ndarray,
                     data: np.ndarray,
                     H: int,
                     seed: int,
                     use_signed: bool) -> np.ndarray:
    if indices.size == 0:
        return np.zeros(H, dtype=np.float32)
    buckets = indices % H
    weights = data.astype(np.float32)
    if use_signed:
        signs = signed_hash_from_ids(indices, seed)
        weights = weights * signs
    out = np.bincount(buckets, weights=weights, minlength=H).astype(np.float32)
    return out


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


# ----------------- Metrics -----------------
def ndcg_at_k_single(qid: str, ranked_ids: List[str],
                     qrels: Dict[str, Dict[str, int]], k: int = 10) -> float:
    rel = qrels.get(qid, {})
    gains = [1 if rel.get(d, 0) > 0 else 0 for d in ranked_ids[:k]]

    def dcg(xs):
        return sum((x / math.log2(i + 2)) for i, x in enumerate(xs))

    idcg = dcg(sorted(gains, reverse=True))
    return (dcg(gains) / idcg) if idcg > 0 else 0.0


def compute_ndcg_at_k(run: Dict[str, List[Tuple[str, float]]],
                      qrels: Dict[str, Dict[str, int]], k: int = 10) -> float:
    vals = [
        ndcg_at_k_single(qid, [d for d, _ in ranked], qrels, k)
        for qid, ranked in run.items()
    ]
    return float(np.mean(vals)) if vals else 0.0


def compute_mrr_at_k(run: Dict[str, List[Tuple[str, float]]],
                     qrels: Dict[str, Dict[str, int]], k: int = 10) -> float:
    mrrs = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        rr = 0.0
        for i, (docid, _) in enumerate(ranked[:k], start=1):
            if rel.get(docid, 0) > 0:
                rr = 1.0 / i
                break
        mrrs.append(rr)
    return float(np.mean(mrrs)) if mrrs else 0.0


def topk_pairs(scores: np.ndarray, ids: List[str], K: int) -> List[Tuple[str, float]]:
    K = min(K, scores.size)
    idx = np.argpartition(-scores, K - 1)[:K]
    idx = idx[np.argsort(-scores[idx])]
    return [(ids[i], float(scores[i])) for i in idx]


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Hybrid Hadamard fusion: dense ⊙ (1 + λ_s * projected_sparse).")
    ap.add_argument("--subset_dir", type=str, required=True,
                    help="Subset dir with corpus.jsonl, queries.jsonl, qrels.jsonl")
    ap.add_argument("--vec_dirname", type=str, default="vectors_local",
                    help="Dir inside subset_dir where make_vectors.py outputs live")
    ap.add_argument("--hash_seed", type=int, default=17)
    ap.add_argument("--use_signed_hash", action="store_true")
    ap.add_argument("--lambda_sparse", type=float, default=0.5,
                    help="Strength of sparse gating (try 0.1–1.0)")
    ap.add_argument("--normalize_parts", action="store_true",
                    help="L2-normalize dense & sparse projections before Hadamard")
    ap.add_argument("--write_run", action="store_true")
    ap.add_argument("--topk", type=int, default=1000)
    args = ap.parse_args()

    SUB = Path(args.subset_dir)
    VEC = SUB / args.vec_dirname

    dense_docs = np.load(VEC / "dense_docs.npy")      # (N_docs, D)
    dense_q    = np.load(VEC / "dense_queries.npy")   # (N_q, D)
    N_docs, D = dense_docs.shape
    N_q = dense_q.shape[0]

    docs = list(read_jsonl(SUB / "corpus.jsonl"))
    queries = list(read_jsonl(SUB / "queries.jsonl"))
    doc_ids = [d["doc_id"] for d in docs]
    qids    = [q["qid"] for q in queries]

    bm25_docs_entries = {}
    for row in read_jsonl(VEC / "bm25_docs.jsonl"):
        did = row["doc_id"]
        s = row["sparse"]
        bm25_docs_entries[did] = {
            "indices": np.array(s["indices"], dtype=np.int64),
            "values": np.array(s["values"], dtype=np.float32)
        }

    bm25_q_entries = {}
    for row in read_jsonl(VEC / "bm25_queries.jsonl"):
        qid = row["qid"]
        s = row["sparse"]
        bm25_q_entries[qid] = {
            "indices": np.array(s["indices"], dtype=np.int64),
            "values": np.array(s["values"], dtype=np.float32)
        }

    # ---- Project BM25 into dim D ----
    print(f"Projecting BM25 to dim={D} (Hadamard fusion), signed={args.use_signed_hash} ...")
    proj_docs = np.zeros((N_docs, D), dtype=np.float32)
    for i, did in enumerate(tqdm(doc_ids, desc="bm25-docs->proj")):
        entry = bm25_docs_entries.get(did)
        if entry is None:
            continue
        proj_docs[i] = project_row_hash(
            entry["indices"], entry["values"], D,
            args.hash_seed, args.use_signed_hash
        )

    proj_q = np.zeros((N_q, D), dtype=np.float32)
    for i, qid in enumerate(tqdm(qids, desc="bm25-queries->proj")):
        entry = bm25_q_entries.get(qid)
        if entry is None:
            continue
        proj_q[i] = project_row_hash(
            entry["indices"], entry["values"], D,
            args.hash_seed, args.use_signed_hash
        )

    # ---- Optional: normalize parts ----
    if args.normalize_parts:
        dense_docs = l2_normalize(dense_docs)
        dense_q    = l2_normalize(dense_q)
        proj_docs  = l2_normalize(proj_docs)
        proj_q     = l2_normalize(proj_q)

    # ---- Build multiplicative gate: scale = 1 + λ_s * proj_sparse ----
    # We slightly squash sparse projection to keep things stable:
    gate_docs = 1.0 + args.lambda_sparse * np.tanh(proj_docs)
    gate_q    = 1.0 + args.lambda_sparse * np.tanh(proj_q)

    hyb_docs = (dense_docs * gate_docs).astype(np.float32)
    hyb_q    = (dense_q    * gate_q).astype(np.float32)

    out_docs = VEC / "hybrid_hadamard_docs.npy"
    out_q    = VEC / "hybrid_hadamard_queries.npy"
    np.save(out_docs, hyb_docs)
    np.save(out_q,    hyb_q)

    meta = {
        "type": "dense⊙(1+λ_s*tanh(projected_sparse))",
        "dim": int(D),
        "lambda_sparse": float(args.lambda_sparse),
        "normalize_parts": bool(args.normalize_parts),
        "hash_seed": int(args.hash_seed),
        "use_signed_hash": bool(args.use_signed_hash),
    }
    with (VEC / "hybrid_hadamard_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print(" -", out_docs)
    print(" -", out_q)
    print(" -", VEC / "hybrid_hadamard_meta.json")

    # ---- Optional: retrieval + evaluation ----
    if args.write_run:
        print("Scoring with Hadamard fusion hybrid vectors...")
        run: Dict[str, List[Tuple[str, float]]] = {}
        docs_T = hyb_docs.T  # (D, N_docs)
        for qi in tqdm(range(N_q), desc="retrieve(hybrid_hadamard)"):
            scores = hyb_q[qi] @ docs_T
            run[qids[qi]] = topk_pairs(scores, doc_ids, args.topk)

        qrels = {}
        for r in read_jsonl(SUB / "qrels.jsonl"):
            qrels.setdefault(str(r["qid"]), {})[str(r["doc_id"])] = int(r["rel"])

        nd = compute_ndcg_at_k(run, qrels, k=10)
        mr = compute_mrr_at_k(run, qrels, k=10)
        print(f"\nHybrid-Hadamard  nDCG@10={nd:.4f}  MRR@10={mr:.4f}")

        out_run = SUB / "run_hybrid_hadamard.trec"
        with out_run.open("w", encoding="utf-8") as f:
            for qid, ranked in run.items():
                for rank, (docid, score) in enumerate(ranked, start=1):
                    f.write(f"{qid} Q0 {docid} {rank} {score:.6f} hybrid_hadamard\n")
        print("Wrote:", out_run)


if __name__ == "__main__":
    main()

