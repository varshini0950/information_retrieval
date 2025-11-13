<!-- PROJECT HEADER -->
<div align="center">

# 🔍 Hybrid Sparse + Dense Retrieval  
### **BM25 + Hybrid Fusion Pipeline **

A complete retrieval framework combining **sparse lexical relevance** and **dense semantic embeddings**, along with multiple hybrid fusion models:
**score-level**, **vector-level**, **hash-projected**, and **Transformer cross-attention fusion**.

</div>

---

**NOTE:** All the other files are made by me for creating a base of the project in which I runned only the part of the dataset bier trec covid itself 
#  Project Structure
```
├── make_vectors.py
├── hybrid_no_alpha.py
├── build_concat.py
├── hybrid_sum.py
├── hybrid_hadamard.py
├── cross_attention_fusion.py
├── eval_all_systems.py
├── compare_hybrids.py
└── work/
    └── subsets/
        └── beir_trec-covid/
            └── subset_1/
```




---
**NOTE:** All the other files are made by me for creating a base of the project in which I runned only the part of the dataset bier trec covid itself  
#  1. `make_vectors.py`
### **Purpose:**  
Generate all foundational vectors used by hybrid systems.

### **Outputs → `vectors_local/`**
- `dense_docs.npy`, `dense_queries.npy`  
- `bm25_docs_csr.npz`  
- `bm25_docs.jsonl`, `bm25_queries.jsonl`  
- `bm25_meta.json`, `bm25_df.npy`  

> **Must be executed first.**

---

# 2. `hybrid_no_alpha.py` — Score-Level Fusion

Implements classical **score-combination fusion** (no α required).

### Methods:
- 🔸 **RRF** (Reciprocal Rank Fusion)  
- 🔸 **CombSUM-Z**  
- 🔸 **CombMNZ-Z**  
- 🔸 **Linear Least-Squares Fusion**

### Outputs:
- `run_rrf.trec`
- `run_combsum_z.trec`
- `run_combmnz_z.trec`
- `run_linear_lsq.trec`

### Baseline Runs:
- `run_dense_vec.trec`
- `run_bm25_vec.trec`

---

#  3. `hybrid_sum.py` — Vector SUM Fusion

### Fusion Rule:


hybrid = λ_dense * dense + λ_sparse * projected_sparse


**Sparse → Dense dimension matching** via hashing projection.

### Outputs:
- `hybrid_sum_docs.npy`  
- `hybrid_sum_queries.npy`  
- `hybrid_sum_meta.json`  
- `run_hybrid_sum.trec`

---

#  4. `hybrid_hadamard.py` — Hadamard Multiplicative Fusion

### Fusion Rule:
ybrid = dense ⊙ (1 + λ_sparse * tanh(projected_sparse))

### Outputs:
- `hybrid_hadamard_docs.npy`
- `hybrid_hadamard_queries.npy`
- `hybrid_hadamard_meta.json`
- `run_hybrid_hadamard.trec`

---



#  5. `cross_attention_fusion.py` — Transformer-Based Fusion

Builds hybrid vectors using **cross-attention between dense embedding and BM25 terms**.

### Sequence Input:
[ dense_token ; top_bm25_term_1 ; top_bm25_term_2 ; ... ]

### Optional:
--train_epochs N

### Output:
- `run_cross_attention_fusion.trec`

---

#  6. `eval_all_systems.py` — Unified Evaluation

Metrics evaluated:
-  nDCG@10  
-  MRR@10  
-  Precision@10  
-  Recall@10  
-  MAP@100  

### Outputs:
- `metrics_summary.csv`
- `metrics_summary.png` (comparison chart)

### Models Compared:
- BM25  
- Dense  
- Hybrid_Concat  
- Hybrid_SUM  
- Hybrid_Hadamard  
- Cross_Attention  
- RRF  
- CombSUM_Z  
- CombMNZ_Z  
- Linear_LSQ  
- BM25_vec  
- Dense_vec  

---

# Recommended Execution Pipeline

### **Step 1 — Build Base Vectors**
```bash
python3 make_vectors.py \
  --subset_dir ./work/subsets/beir_trec-covid/subset_1 \
  --dense_normalize
Step 2 — Hybrid Vector Encoders

SUM Fusion
python3 build_hybrid_sum.py \
  --subset_dir ./work/subsets/beir_trec-covid/subset_1 \
  --vec_dirname vectors_local \
  --use_signed_hash \
  --normalize_parts \
  --lambda_dense 1.0 \
  --lambda_sparse 1.0 \
  --write_run \
  --topk 1000

Hadamard Fusion
python3 build_hybrid_hadamard.py \
  --subset_dir ./work/subsets/beir_trec-covid/subset_1 \
  --vec_dirname vectors_local \
  --lambda_sparse 0.5 \
  --use_signed_hash \
  --normalize_parts \
  --write_run \
  --topk 1000 \
  --metric_k 10 \
  --map_k 100


Step 3 — Score-Level Fusion
python3 hybrid_no_alpha.py \
  --subset_dir ./work/subsets/beir_trec-covid/subset_1 \
  --vec_dirname vectors_local \
  --topk 1000 \
  --metric_k 10 \
  --map_k 100


Step 4 — Cross-Attention Fusion
python3 cross_attention_fusion.py \
  --subset_dir ./work/subsets/beir_trec-covid/subset_1 \
  --vec_dirname vectors_local \
  --max_terms 32 \
  --train_epochs 1 \
  --topk 1000

Step 5 — Compare All Methods
python3 compare_hybrids.py



Final Run Files (Stored in subset directory)
work/subsets/beir_trec-covid/subset_1/
Generated:

run_bm25.trec

run_dense.trec

run_hybrid_concat.trec

run_hybrid_sum.trec

run_hybrid_hadamard.trec

run_cross_attention_fusion.trec

run_rrf.trec

run_combsum_z.trec

run_combmnz_z.trec

run_linear_lsq.trec

run_bm25_vec.trec

run_dense_vec.trec

These are all consumed by:
eval_all_systems.py
