#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ----------------- I/O helpers -----------------
def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

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


# ---------- NEW METRICS: P@k, R@k, MAP@k ----------

def compute_precision_at_k(run: Dict[str, List[Tuple[str, float]]],
                           qrels: Dict[str, Dict[str, int]], k: int = 10) -> float:
    """Macro-averaged Precision@k."""
    precs = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        top = ranked[:k]
        num_rel = sum(1 for docid, _ in top if rel.get(docid, 0) > 0)
        precs.append(num_rel / float(len(top)) if top else 0.0)
    return float(np.mean(precs)) if precs else 0.0


def compute_recall_at_k(run: Dict[str, List[Tuple[str, float]]],
                        qrels: Dict[str, Dict[str, int]], k: int = 10) -> float:
    """Macro-averaged Recall@k."""
    recs = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        total_rel = sum(1 for r in rel.values() if r > 0)
        if total_rel == 0:
            continue
        top = ranked[:k]
        num_rel = sum(1 for docid, _ in top if rel.get(docid, 0) > 0)
        recs.append(num_rel / float(total_rel))
    return float(np.mean(recs)) if recs else 0.0


def average_precision_at_k_single(qid: str,
                                  ranked_ids: List[str],
                                  qrels: Dict[str, Dict[str, int]],
                                  k: int = 100) -> float:
    """AP@k for a single query."""
    rel = qrels.get(qid, {})
    total_rel = sum(1 for r in rel.values() if r > 0)
    if total_rel == 0:
        return 0.0

    num_rel_seen = 0
    ap = 0.0
    for i, docid in enumerate(ranked_ids[:k], start=1):
        if rel.get(docid, 0) > 0:
            num_rel_seen += 1
            ap += num_rel_seen / float(i)
    return ap / float(total_rel) if num_rel_seen > 0 else 0.0


def compute_map_at_k(run: Dict[str, List[Tuple[str, float]]],
                     qrels: Dict[str, Dict[str, int]], k: int = 100) -> float:
    """Macro-averaged MAP@k (only over queries with at least one relevant)."""
    aps = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        if not any(r > 0 for r in rel.values()):
            continue
        ranked_ids = [d for d, _ in ranked]
        aps.append(average_precision_at_k_single(qid, ranked_ids, qrels, k))
    return float(np.mean(aps)) if aps else 0.0


# ----------------- Cross-attention fusion model -----------------
class CrossAttentionFusion(nn.Module):
    """
    Fusion model:
      input sequence per item (doc or query):
        [ dense_token ; term1_embed * w1 ; term2_embed * w2 ; ... ]
      -> TransformerEncoder
      -> use output at position 0 as fused vector
    """

    def __init__(
        self,
        vocab_size: int,
        dense_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dense_proj = nn.Linear(dense_dim, d_model)
        self.term_emb = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=False,  # we'll use (L, B, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        dense_vecs: torch.Tensor,      # (B, D_dense)
        term_ids: torch.Tensor,        # (B, T_max) long
        term_weights: torch.Tensor,    # (B, T_max) float
        term_mask: torch.Tensor,       # (B, T_max) bool, True = has term / valid
    ) -> torch.Tensor:
        """
        Returns fused vectors: (B, d_model)
        """
        B, T_max = term_ids.shape
        device = dense_vecs.device

        # dense token: (B, 1, d_model)
        dense_tok = self.dense_proj(dense_vecs).unsqueeze(1)

        # term embeddings: (B, T_max, d_model)
        term_embs = self.term_emb(term_ids)  # (B, T_max, d_model)
        # scale by BM25 weights
        term_embs = term_embs * term_weights.unsqueeze(-1)

        # build full sequence: (B, 1 + T_max, d_model)
        seq = torch.cat([dense_tok, term_embs], dim=1)  # (B, L, d_model), L = 1 + T_max

        # key padding mask: True = pad -> ignore
        # dense token is always valid
        pad_mask = torch.zeros((B, 1 + T_max), dtype=torch.bool, device=device)
        pad_mask[:, 1:] = ~term_mask  # invert: term_mask=True→valid, so pad_mask=True→invalid

        # Transformer expects (L, B, D)
        seq_t = seq.transpose(0, 1)  # (L, B, d_model)

        enc = self.encoder(seq_t, src_key_padding_mask=pad_mask)  # (L, B, d_model)
        enc = enc.transpose(0, 1)  # (B, L, d_model)

        # fused representation = output at position 0 (the dense token)
        fused = enc[:, 0, :]  # (B, d_model)
        fused = self.out_norm(fused)
        return fused


# ----------------- Helper: build term matrices -----------------
def build_term_matrices(
    entries: Dict[str, Dict[str, List[int]]],
    id_order: List[str],
    max_terms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    entries: {id: {"indices": [...], "values": [...]}}

    Returns:
      term_ids:  (N, max_terms) int64
      term_w:    (N, max_terms) float32
      term_mask: (N, max_terms) bool   (True if slot is used)
      vocab_size: int (max term_id + 1)
    """
    N = len(id_order)
    term_ids = np.zeros((N, max_terms), dtype=np.int64)
    term_w = np.zeros((N, max_terms), dtype=np.float32)
    term_mask = np.zeros((N, max_terms), dtype=bool)

    max_tid = 0

    for i, _id in enumerate(id_order):
        row = entries.get(_id)
        if row is None:
            continue
        idx = np.array(row["indices"], dtype=np.int64)
        val = np.array(row["values"], dtype=np.float32)
        if idx.size == 0:
            continue

        # keep terms with largest |weight| up to max_terms
        if idx.size > max_terms:
            order = np.argsort(-np.abs(val))[:max_terms]
            idx = idx[order]
            val = val[order]

        L = idx.size
        term_ids[i, :L] = idx
        term_w[i, :L] = val
        term_mask[i, :L] = True

        if idx.max() > max_tid:
            max_tid = int(idx.max())

    vocab_size = max_tid + 1
    return term_ids, term_w, term_mask, vocab_size


# ----------------- Training data for pairwise loss -----------------
def build_pairwise_training_triples(
    qids: List[str],
    doc_ids: List[str],
    qrels: Dict[str, Dict[str, int]],
) -> List[Tuple[int, int, int]]:
    """
    Returns list of (q_idx, pos_doc_idx, neg_doc_idx)
    """
    N_docs = len(doc_ids)
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    triples = []
    rng = np.random.default_rng(13)

    for qi, qid in enumerate(qids):
        rels = qrels.get(qid, {})
        pos_docs = [doc_id_to_idx[d] for d, r in rels.items() if r > 0 and d in doc_id_to_idx]
        if not pos_docs:
            continue

        all_pos = set(pos_docs)
        for _ in range(len(pos_docs)):
            pos = rng.choice(pos_docs)
            # sample negative not in positives
            while True:
                neg = int(rng.integers(0, N_docs))
                if neg not in all_pos:
                    break
            triples.append((qi, pos, neg))
    return triples


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Cross-attention fusion of dense + sparse BM25 vectors.")
    ap.add_argument("--subset_dir", type=str, required=True,
                    help="Path to subset folder (contains corpus.jsonl, queries.jsonl, qrels.jsonl)")
    ap.add_argument("--vec_dirname", type=str, default="vectors_local",
                    help="Directory inside subset_dir where make_vectors.py outputs live")
    ap.add_argument("--max_terms", type=int, default=32,
                    help="Max BM25 terms per doc/query for fusion")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--train_epochs", type=int, default=0,
                    help="Number of epochs of pairwise training (0 = no training, just random init)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=1000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SUB = Path(args.subset_dir)
    VEC = SUB / args.vec_dirname

    # ---------- Load dense vectors ----------
    dense_docs = np.load(VEC / "dense_docs.npy")      # (N_docs, dense_dim)
    dense_q = np.load(VEC / "dense_queries.npy")      # (N_queries, dense_dim)
    N_docs, dense_dim = dense_docs.shape
    N_q = dense_q.shape[0]

    # ---------- Load metadata & raw ids ----------
    docs = list(read_jsonl(SUB / "corpus.jsonl"))
    queries = list(read_jsonl(SUB / "queries.jsonl"))
    doc_ids = [d["doc_id"] for d in docs]
    qids = [q["qid"] for q in queries]

    # ---------- Load BM25 sparse JSONLs ----------
    bm25_docs_entries = {}
    for row in read_jsonl(VEC / "bm25_docs.jsonl"):
        did = row["doc_id"]
        s = row["sparse"]
        bm25_docs_entries[did] = {"indices": s["indices"], "values": s["values"]}

    bm25_q_entries = {}
    for row in read_jsonl(VEC / "bm25_queries.jsonl"):
        qid = row["qid"]
        s = row["sparse"]
        bm25_q_entries[qid] = {"indices": s["indices"], "values": s["values"]}

    # ---------- Build term matrices ----------
    doc_term_ids, doc_term_w, doc_term_mask, doc_vocab_size = build_term_matrices(
        bm25_docs_entries, doc_ids, args.max_terms
    )
    q_term_ids, q_term_w, q_term_mask, q_vocab_size = build_term_matrices(
        bm25_q_entries, qids, args.max_terms
    )
    vocab_size = max(doc_vocab_size, q_vocab_size)

    print(f"Docs: {N_docs}, Queries: {N_q}, dense_dim={dense_dim}, vocab_size={vocab_size}")

    # ---------- Load qrels ----------
    qrels = {}
    for r in read_jsonl(SUB / "qrels.jsonl"):
        qrels.setdefault(str(r["qid"]), {})[str(r["doc_id"])] = int(r["rel"])

    # ---------- Move to torch ----------
    dense_docs_t = torch.tensor(dense_docs, dtype=torch.float32, device=device)
    dense_q_t = torch.tensor(dense_q, dtype=torch.float32, device=device)

    doc_term_ids_t = torch.tensor(doc_term_ids, dtype=torch.long, device=device)
    doc_term_w_t = torch.tensor(doc_term_w, dtype=torch.float32, device=device)
    doc_term_mask_t = torch.tensor(doc_term_mask, dtype=torch.bool, device=device)

    q_term_ids_t = torch.tensor(q_term_ids, dtype=torch.long, device=device)
    q_term_w_t = torch.tensor(q_term_w, dtype=torch.float32, device=device)
    q_term_mask_t = torch.tensor(q_term_mask, dtype=torch.bool, device=device)

    # ---------- Build model ----------
    model = CrossAttentionFusion(
        vocab_size=vocab_size,
        dense_dim=dense_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    # ---------- Optional training (pairwise ranking) ----------
    if args.train_epochs > 0:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        margin = args.margin

        triples = build_pairwise_training_triples(qids, doc_ids, qrels)
        print(f"Training triples: {len(triples)}")

        if triples:
            for epoch in range(args.train_epochs):
                np.random.shuffle(triples)
                total_loss = 0.0
                n_batches = 0

                for start in range(0, len(triples), args.batch_size):
                    batch = triples[start:start + args.batch_size]
                    if not batch:
                        continue

                    q_idx = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
                    pos_idx = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
                    neg_idx = torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)

                    # fuse queries
                    q_fused = model(
                        dense_q_t[q_idx],
                        q_term_ids_t[q_idx],
                        q_term_w_t[q_idx],
                        q_term_mask_t[q_idx],
                    )  # (B, d_model)

                    # fuse pos/neg docs
                    pos_fused = model(
                        dense_docs_t[pos_idx],
                        doc_term_ids_t[pos_idx],
                        doc_term_w_t[pos_idx],
                        doc_term_mask_t[pos_idx],
                    )
                    neg_fused = model(
                        dense_docs_t[neg_idx],
                        doc_term_ids_t[neg_idx],
                        doc_term_w_t[neg_idx],
                        doc_term_mask_t[neg_idx],
                    )

                    # scores = dot product
                    s_pos = (q_fused * pos_fused).sum(dim=1)
                    s_neg = (q_fused * neg_fused).sum(dim=1)

                    # hinge ranking loss
                    loss = torch.clamp(margin - (s_pos - s_neg), min=0.0).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += float(loss.item())
                    n_batches += 1

                avg_loss = total_loss / max(1, n_batches)
                print(f"Epoch {epoch+1}/{args.train_epochs}  loss={avg_loss:.4f}")
        else:
            print("No training triples (no positives in qrels) – skipping training.")
    else:
        print("Skipping training (train_epochs=0): using random-initialized fusion.")

    # ---------- Inference: build fused vectors ----------
    model.eval()
    with torch.no_grad():
        # docs in batches
        B = 256
        fused_docs_list = []
        for start in range(0, N_docs, B):
            end = min(N_docs, start + B)
            fd = model(
                dense_docs_t[start:end],
                doc_term_ids_t[start:end],
                doc_term_w_t[start:end],
                doc_term_mask_t[start:end],
            )  # (b, d_model)
            fused_docs_list.append(fd.cpu().numpy())
        fused_docs = np.vstack(fused_docs_list)

        fused_q_list = []
        for start in range(0, N_q, B):
            end = min(N_q, start + B)
            fq = model(
                dense_q_t[start:end],
                q_term_ids_t[start:end],
                q_term_w_t[start:end],
                q_term_mask_t[start:end],
            )
            fused_q_list.append(fq.cpu().numpy())
        fused_q = np.vstack(fused_q_list)

    # ---------- Scoring & evaluation ----------
    print("Scoring with fused vectors...")
    run: Dict[str, List[Tuple[str, float]]] = {}
    docs_T = fused_docs.T  # (d_model, N_docs)

    for qi in tqdm(range(N_q), desc="score(fused)"):
        scores = fused_q[qi] @ docs_T  # (N_docs,)
        run[qids[qi]] = topk_pairs(scores, doc_ids, args.topk)

    # main metrics
    nd = compute_ndcg_at_k(run, qrels, k=10)
    mr = compute_mrr_at_k(run, qrels, k=10)
    p10 = compute_precision_at_k(run, qrels, k=10)      # NEW
    r10 = compute_recall_at_k(run, qrels, k=10)         # NEW
    map100 = compute_map_at_k(run, qrels, k=100)        # NEW

    print("\nCrossAttentionFusion metrics:")
    print(f"  nDCG@10  = {nd:.4f}")
    print(f"  MRR@10   = {mr:.4f}")
    print(f"  P@10     = {p10:.4f}")
    print(f"  R@10     = {r10:.4f}")
    print(f"  MAP@100  = {map100:.4f}")

    # ---------- Save run ----------
    out_run = SUB / "run_cross_attention_fusion.trec"
    with out_run.open("w", encoding="utf-8") as f:
        for qid, ranked in run.items():
            for rank, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} cross_attn\n")
    print("Wrote:", out_run)


if __name__ == "__main__":
    main()

