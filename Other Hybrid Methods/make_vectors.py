#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz

# --------- IO helpers ----------
def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

# --------- Tokenizer ----------
TOKEN_RE = re.compile(r"[a-z0-9]+")
def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())

# --------- BM25 core ----------
def bm25_idf(N, df_i):
    # Robertson/Sparck Jones IDF with +1 to avoid zero/negatives
    return math.log((N - df_i + 0.5) / (df_i + 0.5) + 1.0)

def build_bm25_docs(texts, k1=1.2, b=0.75):
    """
    Build BM25 CSR matrix for documents.
    Returns: (csr_matrix, vocab: dict[token->term_id], df: np.array, avgdl: float)
    """
    vocab = {}
    df = defaultdict(int)
    tok_docs = []
    print("Tokenizing corpus...")
    for txt in tqdm(texts, desc="tokenize"):
        toks = tokenize(txt)
        tok_docs.append(toks)
        for t in set(toks):
            if t not in vocab:
                vocab[t] = len(vocab)
            df[vocab[t]] += 1

    N = len(texts)
    avgdl = float(np.mean([len(t) for t in tok_docs])) if N > 0 else 0.0

    print("Computing IDF...")
    V = len(vocab)
    idf_vec = np.zeros(V, dtype=np.float32)
    for tid, dfi in df.items():
        idf_vec[tid] = bm25_idf(N, dfi)

    print("Building BM25 CSR...")
    rows, cols, data = [], [], []
    for di, toks in enumerate(tqdm(tok_docs, desc="bm25-docs")):
        tf = Counter(toks)
        dl = len(toks)
        norm = k1 * (1 - b + b * (dl / avgdl)) if avgdl > 0 else k1  # guard
        for t, c in tf.items():
            tid = vocab[t]
            denom = c + norm
            w = idf_vec[tid] * ((c * (k1 + 1.0)) / (denom if denom > 0 else 1.0))
            rows.append(di); cols.append(tid); data.append(w)

    bm25_csr = csr_matrix(
        (np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))),
        shape=(N, V),
        dtype=np.float32
    )
    # Pack df into dense array aligned with term ids
    df_arr = np.zeros(V, dtype=np.int32)
    for tid, dfi in df.items():
        df_arr[tid] = dfi

    return bm25_csr, vocab, df_arr, avgdl

def bm25_query_vector(qtext, vocab, df_arr, N, avgdl, k1=1.2, b=0.75):
    qtoks = tokenize(qtext)
    if not qtoks:
        return [], []
    tfq = Counter(qtoks)
    dlq = len(qtoks)
    out_idx, out_val = [], []
    norm = k1 * (1 - b + b * (dlq / avgdl)) if avgdl > 0 else k1

    for t, c in tfq.items():
        if t in vocab:
            tid = vocab[t]
            dfi = int(df_arr[tid])
            idfv = bm25_idf(N, dfi)
            denom = c + norm
            w = idfv * ((c * (k1 + 1.0)) / (denom if denom > 0 else 1.0))
            out_idx.append(tid); out_val.append(float(w))
        else:
            # OOV term: use DF=1 (your SOP note)
            dfi = 1
            idfv = bm25_idf(N, dfi)
            denom = c + norm
            w = idfv * ((c * (k1 + 1.0)) / (denom if denom > 0 else 1.0))
            # No tid to attach in fixed-vocab setting → skip adding to vector
            # (This matches "static DF + no new terms in vocab" behavior)
            # If you prefer to include it, you'd need a dynamic term id, which we avoid here.
            _ = w

    return out_idx, out_val

# --------- Dense vectors ----------
def build_dense(texts, model_name, batch, normalize):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False
    )
    return vecs

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Generate BM25 (CSR + JSONL) and Dense vectors (JSONL + NPY).")
    ap.add_argument("--subset_dir", type=str, required=True,
                    help="Path to subset folder containing corpus.jsonl and queries.jsonl")
    ap.add_argument("--out_dirname", type=str, default="vectors_local",
                    help="Output folder name inside subset_dir (default: vectors_local)")
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--dense_batch", type=int, default=256)
    ap.add_argument("--dense_normalize", action="store_true",
                    help="L2-normalize embeddings (recommended if using cosine/IP)")
    args = ap.parse_args()

    SUBSET_DIR = Path(args.subset_dir)
    OUT_DIR = SUBSET_DIR / args.out_dirname
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_path = SUBSET_DIR / "corpus.jsonl"
    queries_path = SUBSET_DIR / "queries.jsonl"
    assert corpus_path.exists(), f"Missing {corpus_path}"
    assert queries_path.exists(), f"Missing {queries_path}"

    print("Loading JSONLs...")
    docs = list(read_jsonl(corpus_path))
    queries = list(read_jsonl(queries_path))

    doc_ids = [d["doc_id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    qids = [q["qid"] for q in queries]
    qtexts = [q["text"] for q in queries]

    # ---------------- BM25 docs ----------------
    bm25_csr, vocab, df_arr, avgdl = build_bm25_docs(doc_texts, k1=args.k1, b=args.b)
    N, V = bm25_csr.shape
    print(f"BM25 matrix: docs={N}, vocab={V}, nnz={bm25_csr.nnz}, avgdl={avgdl:.2f}")

    # Save CSR and BM25 metadata (for queries)
    save_npz(OUT_DIR / "bm25_docs_csr.npz", bm25_csr)
    bm25_meta = {
        "k1": args.k1,
        "b": args.b,
        "N": int(N),
        "avgdl": float(avgdl),
        "vocab": vocab,             # token -> term_id
        "vocab_size": int(V),
    }
    with open(OUT_DIR / "bm25_meta.json", "w", encoding="utf-8") as f:
        json.dump(bm25_meta, f)
    np.save(OUT_DIR / "bm25_df.npy", df_arr)

    # Export per-doc {indices, values} JSONL (Pinecone-style)
    print("Writing per-doc BM25 sparse JSONL...")
    bm25_csr = bm25_csr.tocsr()
    with open(OUT_DIR / "bm25_docs.jsonl", "w", encoding="utf-8") as f:
        for di, did in enumerate(tqdm(doc_ids, desc="bm25-docs-jsonl")):
            start, end = bm25_csr.indptr[di], bm25_csr.indptr[di+1]
            idx = bm25_csr.indices[start:end].tolist()
            val = bm25_csr.data[start:end].tolist()
            f.write(json.dumps({"doc_id": did, "sparse": {"indices": idx, "values": val}}) + "\n")

    # ---------------- BM25 queries ----------------
    print("Encoding queries as BM25 sparse...")
    bm25_q_rows = []
    for qid, qt in tqdm(list(zip(qids, qtexts)), desc="bm25-queries", total=len(qids)):
        idx, val = bm25_query_vector(qt, vocab, df_arr, N, avgdl, k1=args.k1, b=args.b)
        bm25_q_rows.append({"qid": qid, "sparse": {"indices": idx, "values": val}})
    write_jsonl(OUT_DIR / "bm25_queries.jsonl", bm25_q_rows)

    # ---------------- Dense vectors ----------------
    print(f"Dense model: {args.dense_model}")
    doc_vecs = build_dense(doc_texts, args.dense_model, args.dense_batch, args.dense_normalize)
    q_vecs   = build_dense(qtexts,   args.dense_model, args.dense_batch, args.dense_normalize)
    print(f"Dense dims: {doc_vecs.shape[1]} | docs={doc_vecs.shape[0]} | queries={q_vecs.shape[0]}")

    # Save JSONL (Pinecone-ready) + fast arrays
    print("Writing dense JSONLs and NPZ/NPY...")
    with open(OUT_DIR / "dense_docs.jsonl", "w", encoding="utf-8") as f:
        for did, v in zip(doc_ids, doc_vecs):
            f.write(json.dumps({"doc_id": did, "values": v.tolist()}) + "\n")
    with open(OUT_DIR / "dense_queries.jsonl", "w", encoding="utf-8") as f:
        for qid, v in zip(qids, q_vecs):
            f.write(json.dumps({"qid": qid, "values": v.tolist()}) + "\n")

    np.save(OUT_DIR / "dense_docs.npy", doc_vecs)
    np.save(OUT_DIR / "dense_queries.npy", q_vecs)

    # ---------------- Stats summary ----------------
    avg_nnz = (bm25_csr.nnz / N) if N > 0 else 0.0
    print("\n=== SUMMARY ===")
    print(f"Subset dir        : {SUBSET_DIR}")
    print(f"Output dir        : {OUT_DIR}")
    print(f"Docs, Vocab       : {N}, {V}")
    print(f"BM25 nnz (total)  : {bm25_csr.nnz}  | avg nnz/doc: {avg_nnz:.1f}")
    print(f"Dense dim         : {doc_vecs.shape[1]}")
    print("Files written:")
    for fname in [
        "bm25_docs_csr.npz",
        "bm25_meta.json",
        "bm25_df.npy",
        "bm25_docs.jsonl",
        "bm25_queries.jsonl",
        "dense_docs.jsonl",
        "dense_queries.jsonl",
        "dense_docs.npy",
        "dense_queries.npy",
    ]:
        print(" -", OUT_DIR / fname)

if __name__ == "__main__":
    main()

