#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ---------- IO ----------
def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def read_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str,int]] = {}
    for r in read_jsonl(path):
        qid = str(r["qid"])
        did = str(r["doc_id"])
        rel = int(r["rel"])
        qrels.setdefault(qid, {})[did] = rel
    return qrels

def read_run(path: Path) -> Dict[str, List[Tuple[str, float]]]:
    per_q: Dict[str, List[Tuple[str,float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _q0, docid, _rank, score, _tag = parts[:6]
            per_q.setdefault(qid, []).append((docid, float(score)))
    # sort by score desc
    for qid in per_q:
        per_q[qid].sort(key=lambda x: x[1], reverse=True)
    return per_q

# ---------- metrics ----------
def ndcg_at_k_single(qid: str, ranked_ids: List[str],
                     qrels: Dict[str, Dict[str,int]], k: int = 10) -> float:
    rel = qrels.get(qid, {})
    gains = [1 if rel.get(d, 0) > 0 else 0 for d in ranked_ids[:k]]

    def dcg(xs):
        return sum((x / math.log2(i + 2)) for i, x in enumerate(xs))

    idcg = dcg(sorted(gains, reverse=True))
    return (dcg(gains) / idcg) if idcg > 0 else 0.0

def compute_ndcg_at_k(run: Dict[str, List[Tuple[str,float]]],
                      qrels: Dict[str, Dict[str,int]], k: int = 10) -> float:
    vals = [
        ndcg_at_k_single(qid, [d for d,_ in ranked], qrels, k)
        for qid, ranked in run.items()
    ]
    return float(np.mean(vals)) if vals else 0.0

def compute_mrr_at_k(run: Dict[str, List[Tuple[str,float]]],
                     qrels: Dict[str, Dict[str,int]], k: int = 10) -> float:
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

# ---------- main ----------
def main():
    subset_dir = Path("./work/subsets/beir_trec-covid/subset_1")
    qrels_path = subset_dir / "qrels.jsonl"
    qrels = read_qrels(qrels_path)

    # map: nice name -> run file
    runs = {
    "BM25"               : subset_dir / "run_bm25.trec",
    "Dense"              : subset_dir / "run_dense.trec",
    "Hybrid_Concat"      : subset_dir / "run_hybrid_concat.trec",
    "Hybrid_SUM"         : subset_dir / "run_hybrid_sum.trec",
    "Hybrid_Hadamard"    : subset_dir / "run_hybrid_hadamard.trec",
    "Cross_Attn"         : subset_dir / "run_cross_attention_fusion.trec",

    # ===== Your new vector-fusion scores (from hybrid_no_alpha.py) =====
    "RRF"                : subset_dir / "run_rrf.trec",
    "CombSUM_Z"          : subset_dir / "run_combsum_z.trec",
    "CombMNZ_Z"          : subset_dir / "run_combmnz_z.trec",
    "Linear_LSQ"         : subset_dir / "run_linear_lsq.trec",

    # Optional: these are just raw dense/BM25 recomputed
    "BM25_vec"           : subset_dir / "run_bm25_vec.trec",
    "Dense_vec"          : subset_dir / "run_dense_vec.trec",
}


    print(f"Using qrels: {qrels_path}")
    print(f"{'System':20s} {'nDCG@10':>10s} {'MRR@10':>10s}")

    for name, path in runs.items():
        if not path.exists():
            print(f"{name:20s}  (missing file: {path.name})")
            continue
        run = read_run(path)
        nd = compute_ndcg_at_k(run, qrels, k=10)
        mr = compute_mrr_at_k(run, qrels, k=10)
        print(f"{name:20s} {nd:10.4f} {mr:10.4f}")

if __name__ == "__main__":
    main()

