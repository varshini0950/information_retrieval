#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------- IO ----------
def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def read_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    for r in read_jsonl(path):
        qid = str(r["qid"])
        did = str(r["doc_id"])
        rel = int(r["rel"])
        qrels.setdefault(qid, {})[did] = rel
    return qrels

def read_run(path: Path) -> Dict[str, List[Tuple[str, float]]]:
    per_q: Dict[str, List[Tuple[str, float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _q0, docid, _rank, score, _tag = parts[:6]
            per_q.setdefault(qid, []).append((docid, float(score)))
    # sort by score desc (just to be safe)
    for qid in per_q:
        per_q[qid].sort(key=lambda x: x[1], reverse=True)
    return per_q

# ---------- metrics ----------
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

def compute_precision_at_k(run: Dict[str, List[Tuple[str, float]]],
                           qrels: Dict[str, Dict[str, int]], k: int = 10) -> float:
    precisions = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        top = ranked[:k]
        if not top:
            precisions.append(0.0)
            continue
        rel_hits = sum(1 for docid, _ in top if rel.get(docid, 0) > 0)
        precisions.append(rel_hits / float(k))
    return float(np.mean(precisions)) if precisions else 0.0

def compute_recall_at_k(run: Dict[str, List[Tuple[str, float]]],
                        qrels: Dict[str, Dict[str, int]], k: int = 10) -> float:
    recalls = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        R = sum(1 for r in rel.values() if r > 0)
        if R == 0:
            recalls.append(0.0)
            continue
        top = ranked[:k]
        rel_hits = sum(1 for docid, _ in top if rel.get(docid, 0) > 0)
        recalls.append(rel_hits / float(R))
    return float(np.mean(recalls)) if recalls else 0.0

def compute_map_at_k(run: Dict[str, List[Tuple[str, float]]],
                     qrels: Dict[str, Dict[str, int]], k: int = 100) -> float:
    aps = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        R = sum(1 for r in rel.values() if r > 0)
        if R == 0:
            aps.append(0.0)
            continue
        num_rel_so_far = 0
        sum_precisions = 0.0
        for i, (docid, _) in enumerate(ranked[:k], start=1):
            if rel.get(docid, 0) > 0:
                num_rel_so_far += 1
                sum_precisions += num_rel_so_far / float(i)
        if num_rel_so_far == 0:
            aps.append(0.0)
        else:
            aps.append(sum_precisions / float(R))
    return float(np.mean(aps)) if aps else 0.0

# ---------- main ----------
def main():
    subset_dir = Path("./work/subsets/beir_trec-covid/subset_1")
    qrels_path = subset_dir / "qrels.jsonl"
    qrels = read_qrels(qrels_path)

    # map: nice name -> run file
    runs: Dict[str, Path] = {
        "BM25"            : subset_dir / "run_bm25.trec",
        "Dense"           : subset_dir / "run_dense.trec",
        "Hybrid_Concat"   : subset_dir / "run_hybrid_concat.trec",
        "Hybrid_SUM"      : subset_dir / "run_hybrid_sum.trec",
        "Hybrid_Hadamard" : subset_dir / "run_hybrid_hadamard.trec",
        "Cross_Attn"      : subset_dir / "run_cross_attention_fusion.trec",

        # from hybrid_no_alpha.py
        "RRF"             : subset_dir / "run_rrf.trec",
        "CombSUM_Z"       : subset_dir / "run_combsum_z.trec",
        "CombMNZ_Z"       : subset_dir / "run_combmnz_z.trec",
        "Linear_LSQ"      : subset_dir / "run_linear_lsq.trec",

        # optional raw ones
        "BM25_vec"        : subset_dir / "run_bm25_vec.trec",
        "Dense_vec"       : subset_dir / "run_dense_vec.trec",
    }

    print(f"Using qrels: {qrels_path}\n")

    # collect metrics for plotting
    names   = []
    ndcgs   = []
    mrrs    = []
    p10s    = []
    r10s    = []
    map100s = []

    header = f"{'System':20s} {'nDCG@10':>9s} {'MRR@10':>9s} {'P@10':>9s} {'R@10':>9s} {'MAP@100':>9s}"
    print(header)
    print("-" * len(header))

    for name, path in runs.items():
        if not path.exists():
            print(f"{name:20s}  (missing file: {path.name})")
            continue

        run = read_run(path)
        nd   = compute_ndcg_at_k(run,   qrels, k=10)
        mr   = compute_mrr_at_k(run,    qrels, k=10)
        p10  = compute_precision_at_k(run, qrels, k=10)
        r10  = compute_recall_at_k(run,    qrels, k=10)
        map100 = compute_map_at_k(run,     qrels, k=100)

        print(f"{name:20s} {nd:9.4f} {mr:9.4f} {p10:9.4f} {r10:9.4f} {map100:9.4f}")

        names.append(name)
        ndcgs.append(nd)
        mrrs.append(mr)
        p10s.append(p10)
        r10s.append(r10)
        map100s.append(map100)

    # ---------- Plotting ----------
    if not names:
        print("\nNo runs found, skipping plotting.")
        return

    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Panel 1: nDCG@10
    axes[0].bar(x, ndcgs)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].set_ylabel("nDCG@10")
    axes[0].set_title("nDCG@10 per system")

    # Panel 2: MRR@10
    axes[1].bar(x, mrrs)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].set_ylabel("MRR@10")
    axes[1].set_title("MRR@10 per system")

    # Panel 3: P@10, R@10, MAP@100 overlayed as grouped bars
    width = 0.25
    axes[2].bar(x - width, p10s, width=width, label="P@10")
    axes[2].bar(x,         r10s, width=width, label="R@10")
    axes[2].bar(x + width, map100s, width=width, label="MAP@100")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right")
    axes[2].set_title("P@10 / R@10 / MAP@100")
    axes[2].legend()

    out_fig = subset_dir / "metrics_summary.png"
    plt.savefig(out_fig, dpi=200)
    print(f"\nSaved metrics plot to: {out_fig}")

if __name__ == "__main__":
    main()

