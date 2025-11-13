#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, math, argparse, numpy as np, re
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
from scipy.sparse import load_npz, csr_matrix

# ---------- IO ----------
def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def trec_write(path: Path, per_q: Dict[str, List[Tuple[str, float]]], sysname: str):
    with path.open("w", encoding="utf-8") as f:
        for qid, ranked in per_q.items():
            for r, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docid} {r} {score:.6f} {sysname}\n")

# ---------- metrics ----------
def ndcg_at_k_single(qid: str, ranked_ids: List[str], qrels: Dict[str, Dict[str,int]], k:int=10)->float:
    rel = qrels.get(qid, {})
    gains = [1 if rel.get(d,0)>0 else 0 for d in ranked_ids[:k]]
    def dcg(xs): return sum((x/ math.log2(i+2)) for i,x in enumerate(xs))
    idcg = dcg(sorted(gains, reverse=True))
    return (dcg(gains)/idcg) if idcg>0 else 0.0

def compute_ndcg_at_k(run: Dict[str, List[Tuple[str,float]]], qrels, k=10)->float:
    vals=[ndcg_at_k_single(qid,[d for d,_ in ranked], qrels, k) for qid,ranked in run.items()]
    return float(np.mean(vals)) if vals else 0.0

def compute_mrr_at_k(run: Dict[str, List[Tuple[str,float]]], qrels, k=10)->float:
    m=[]
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        rr=0.0
        for i,(d,_) in enumerate(ranked[:k], start=1):
            if rel.get(d,0)>0:
                rr=1.0/i
                break
        m.append(rr)
    return float(np.mean(m)) if m else 0.0

def compute_precision_at_k(run: Dict[str, List[Tuple[str,float]]], qrels, k=10)->float:
    vals = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        topk = ranked[:k]
        if not topk:
            vals.append(0.0)
            continue
        num_rel = sum(1 for d,_ in topk if rel.get(d,0) > 0)
        vals.append(num_rel / float(k))
    return float(np.mean(vals)) if vals else 0.0

def compute_recall_at_k(run: Dict[str, List[Tuple[str,float]]], qrels, k=10)->float:
    vals = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        total_rel = sum(1 for _d,r in rel.items() if r > 0)
        if total_rel == 0:
            # standard choice: skip or count as 0; we count as 0
            vals.append(0.0)
            continue
        topk = ranked[:k]
        num_rel = sum(1 for d,_ in topk if rel.get(d,0) > 0)
        vals.append(num_rel / float(total_rel))
    return float(np.mean(vals)) if vals else 0.0

def compute_map_at_k(run: Dict[str, List[Tuple[str,float]]], qrels, k=100)->float:
    """Mean Average Precision at cutoff k."""
    aps = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        hits = 0
        precisions = []
        for i, (d, _) in enumerate(ranked[:k], start=1):
            if rel.get(d, 0) > 0:
                hits += 1
                precisions.append(hits / float(i))
        if not precisions:
            aps.append(0.0)
        else:
            aps.append(sum(precisions) / len(precisions))
    return float(np.mean(aps)) if aps else 0.0

# ---------- BM25 query builder (matches make_vectors.py) ----------
TOKEN_RE = re.compile(r"[a-z0-9]+")
def tokenize(s): return TOKEN_RE.findall(s.lower())
def bm25_idf(N, df_i): return math.log((N - df_i + 0.5) / (df_i + 0.5) + 1.0)

def bm25_query_row(qtext: str, meta, df_arr: np.ndarray, V: int) -> csr_matrix:
    toks = tokenize(qtext)
    if not toks: return csr_matrix((1,V), dtype=np.float32)
    from collections import Counter
    tf = Counter(toks); dlq=len(toks)
    k1, b = meta["k1"], meta["b"]; Ndocs=meta["N"]; avgdl=meta["avgdl"]; vocab=meta["vocab"]
    norm = k1*(1-b + b*(dlq/avgdl)) if avgdl>0 else k1
    idx,val=[],[]
    for t,c in tf.items():
        if t in vocab:
            tid=vocab[t]; idfv=bm25_idf(Ndocs, int(df_arr[tid]))
            w = idfv * ((c*(k1+1.0)) / (c+norm if (c+norm)>0 else 1.0))
            idx.append(tid); val.append(w)
    if not idx: return csr_matrix((1,V), dtype=np.float32)
    return csr_matrix((np.array(val, np.float32),
                       (np.zeros(len(idx), np.int32), np.array(idx, np.int32))),
                      shape=(1,V), dtype=np.float32)

# ---------- utilities ----------
def topk_pairs(scores: np.ndarray, ids: List[str], K:int)->List[Tuple[str,float]]:
    K=min(K, scores.size)
    idx = np.argpartition(-scores, K-1)[:K]
    idx = idx[np.argsort(-scores[idx])]
    return [(ids[i], float(scores[i])) for i in idx]

def ranks_from_scores(scores: np.ndarray)->np.ndarray:
    # rank 1 = highest score
    order = scores.argsort(kind="mergesort")[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, scores.size+1, dtype=np.int32)
    return ranks

def zscore(x: np.ndarray)->np.ndarray:
    mu = x.mean()
    sd = x.std()
    if sd<=0: return np.zeros_like(x)
    return (x - mu) / sd

# ---------- fusion alternatives ----------
def rrf_fusion(dense_s: np.ndarray, bm25_s: np.ndarray, k: int = 60)->np.ndarray:
    r1 = ranks_from_scores(dense_s)
    r2 = ranks_from_scores(bm25_s)
    return (1.0 / (k + r1)) + (1.0 / (k + r2))

def combsum_z(b1: np.ndarray, b2: np.ndarray)->np.ndarray:
    # z-normalize each, then sum (no alpha)
    return zscore(b1) + zscore(b2)

def combmnz_z(b1: np.ndarray, b2: np.ndarray)->np.ndarray:
    s1, s2 = zscore(b1), zscore(b2)
    # multiply by number of non-zero contributions (MNZ); since all are real, count via thresholds
    nz = (np.abs(s1)>1e-12).astype(np.float32) + (np.abs(s2)>1e-12).astype(np.float32)
    return (s1 + s2) * nz

def learned_linear(scores_list: List[np.ndarray], y_rel: Dict[str, Dict[str,int]], qid: str, doc_ids: List[str], w: np.ndarray, b: float)->np.ndarray:
    # scores_list = [dense_scores, bm25_scores, ...]; w has same length
    # Unconstrained linear comb: s = sum_j w_j * s_j + b
    S = np.vstack(scores_list)  # (m, N)
    return (w.reshape(-1,1) * S).sum(axis=0) + b

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Hybrid alternatives (no-alpha): RRF, CombSUM/CombMNZ-z, learned linear.")
    ap.add_argument("--subset_dir", required=True, type=str)
    ap.add_argument("--vec_dirname", default="vectors_local", type=str)
    ap.add_argument("--topk", default=1000, type=int)
    ap.add_argument("--metric_k", default=10, type=int, help="cutoff for nDCG/MRR/P/R")
    ap.add_argument("--map_k", default=100, type=int, help="cutoff for MAP")
    ap.add_argument("--dev_ratio", default=0.6, type=float)
    ap.add_argument("--seed", default=13, type=int)
    ap.add_argument("--rrf_k", default=60, type=int, help="RRF constant k (typical 60)")
    args = ap.parse_args()

    SUB = Path(args.subset_dir); VEC = SUB / args.vec_dirname

    # load data
    docs = list(read_jsonl(SUB / "corpus.jsonl"))
    queries = list(read_jsonl(SUB / "queries.jsonl"))
    qrels = {}
    for r in read_jsonl(SUB / "qrels.jsonl"):
        qrels.setdefault(str(r["qid"]), {})[str(r["doc_id"])] = int(r["rel"])

    doc_ids = [d["doc_id"] for d in docs]
    qids    = [q["qid"] for q in queries]
    qtexts  = [q["text"] for q in queries]

    xb = np.load(VEC / "dense_docs.npy")         # (N,d), L2-normalized if you ran --dense_normalize
    xq = np.load(VEC / "dense_queries.npy")      # (Q,d)
    bm25_docs = load_npz(VEC / "bm25_docs_csr.npz").tocsr()  # (N,V)
    with (VEC / "bm25_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    df_arr = np.load(VEC / "bm25_df.npy")
    V = bm25_docs.shape[1]; N = xb.shape[0]

    # precompute scores
    dense_scores_all=[]; bm25_scores_all=[]
    for qi in tqdm(range(len(qids)), desc="Dense dots"):
        dense_scores_all.append(xb @ xq[qi])  # (N,)
    for qi in tqdm(range(len(qids)), desc="BM25 dots"):
        qrow = bm25_query_row(qtexts[qi], meta, df_arr, V)
        bm25_scores_all.append((qrow @ bm25_docs.T).toarray().ravel())

    # split dev/test (used only to fit linear weights)
    rng = np.random.RandomState(args.seed)
    order = np.arange(len(qids)); rng.shuffle(order)
    dev_n = max(1, int(len(qids)*args.dev_ratio))
    dev_idx = order[:dev_n]; test_idx = order[dev_n:]

    # ----- baselines -----
    run_dense={}; run_bm25={}
    for qi in range(len(qids)):
        run_dense[qids[qi]] = topk_pairs(dense_scores_all[qi], doc_ids, args.topk)
        run_bm25[qids[qi]]  = topk_pairs(bm25_scores_all[qi],  doc_ids, args.topk)

    # baseline metrics
    def all_metrics(run, name:str):
        nd = compute_ndcg_at_k(run, qrels, args.metric_k)
        mr = compute_mrr_at_k(run,  qrels, args.metric_k)
        p  = compute_precision_at_k(run, qrels, args.metric_k)
        r  = compute_recall_at_k(run,    qrels, args.metric_k)
        m  = compute_map_at_k(run,       qrels, args.map_k)
        print(f"{name:12s} : nDCG@{args.metric_k}={nd:.4f}  "
              f"MRR@{args.metric_k}={mr:.4f}  "
              f"P@{args.metric_k}={p:.4f}  "
              f"R@{args.metric_k}={r:.4f}  "
              f"MAP@{args.map_k}={m:.4f}")
        return nd, mr, p, r, m

    print("\n=== RESULTS (all queries) ===")
    base_bm25 = all_metrics(run_bm25,  "BM25")
    base_dense= all_metrics(run_dense, "Dense")

    # ----- 1) RRF -----
    run_rrf={}
    for qi in range(len(qids)):
        h = rrf_fusion(dense_scores_all[qi], bm25_scores_all[qi], k=args.rrf_k)
        run_rrf[qids[qi]] = topk_pairs(h, doc_ids, args.topk)
    rrf_metrics = all_metrics(run_rrf, f"RRF(k={args.rrf_k})")

    # ----- 2) CombSUM/CombMNZ with z-score -----
    run_combsum={}; run_combmnz={}
    for qi in range(len(qids)):
        cs = combsum_z(dense_scores_all[qi], bm25_scores_all[qi])
        cm = combmnz_z(dense_scores_all[qi], bm25_scores_all[qi])
        run_combsum[qids[qi]] = topk_pairs(cs, doc_ids, args.topk)
        run_combmnz[qids[qi]] = topk_pairs(cm, doc_ids, args.topk)
    cs_metrics = all_metrics(run_combsum, "CombSUM-z")
    cm_metrics = all_metrics(run_combmnz, "CombMNZ-z")

    # ----- 3) Learned linear (no α constraint) -----
    # Train w,b on DEV to separate relevant vs non-relevant with least squares on [dense, bm25, 1]
    M = 200  # fit on top-200 union to stay light
    feats=[]; targets=[]
    for qi in dev_idx:
        d = dense_scores_all[qi]; b = bm25_scores_all[qi]
        # candidate pool = union of top-M from both
        di = np.argpartition(-d, M-1)[:M]; bi = np.argpartition(-b, M-1)[:M]
        pool = np.array(sorted(set(di.tolist()) | set(bi.tolist())), dtype=int)
        # X = [dense, bm25, 1], y = rel label
        Xq = np.stack([d[pool], b[pool], np.ones_like(pool, dtype=np.float32)], axis=1)
        yq = np.array([1 if qrels.get(qids[qi],{}).get(doc_ids[j],0)>0 else 0 for j in pool], dtype=np.float32)
        feats.append(Xq); targets.append(yq)
    if feats:
        X = np.vstack(feats); y = np.concatenate(targets)
        # least squares
        w_full, *_ = np.linalg.lstsq(X, y, rcond=None)  # shape (3,)
    else:
        w_full = np.array([0.5, 0.5, 0.0], dtype=np.float32)  # fallback

    run_lin={}
    for qi in range(len(qids)):
        d = dense_scores_all[qi]; b = bm25_scores_all[qi]
        s = w_full[0]*d + w_full[1]*b + w_full[2]
        run_lin[qids[qi]] = topk_pairs(s, doc_ids, args.topk)
    lin_metrics = all_metrics(run_lin, "Linear-LSQ")
    print(f"Linear weights w=[{w_full[0]:.3f}, {w_full[1]:.3f}], b={w_full[2]:.3f}")

    # ----- (Optional) “new vector” interpretation without α -----
    # Standardize channels per-query and sum inner-products on scores:
    #   s = < z(dense_q), z(dense_docs) > + < z(bm25_q), z(bm25_docs) >
    # CombSUM-z already realizes the same idea on the score level.

    # ----- write runs -----
    trec_write(SUB / "run_rrf.trec",        run_rrf,     f"rrf_k{args.rrf_k}")
    trec_write(SUB / "run_combsum_z.trec",  run_combsum, "combsum_z")
    trec_write(SUB / "run_combmnz_z.trec",  run_combmnz, "combmnz_z")
    trec_write(SUB / "run_linear_lsq.trec", run_lin,     "linear_lsq")
    trec_write(SUB / "run_bm25_vec.trec",   run_bm25,    "bm25_vec")
    trec_write(SUB / "run_dense_vec.trec",  run_dense,   "dense_vec")
    print("\nWrote TREC runs to subset dir.")
    print("Done.")

if __name__ == "__main__":
    main()

