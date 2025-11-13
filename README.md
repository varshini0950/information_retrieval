# Retrieval Experiments – Project Overview

This repository contains multiple Jupyter Notebooks which contains the code used in the implementation and analyzation of the Hybrid Retrieval methods mentioned in our report.

---

## 📘 1. `1_Run_Retrievers.ipynb`
This notebook runs Sparse and Dense Retrievers on a given dataset and generates a .trec file to store the results.

### Includes:
- Loading datasets (beir datasets using the ir_datasets python API)
- Running retrieval 
- Storing raw retrieval results for later evaluation


**Purpose:**  
This is the **main retrieval execution notebook** that generates raw results for the rest of the analysis.

### Note:
- The .trec files are stored in the "runs" directory.
---

## ⚡ 2. `2_Quick_Execution_run_retrievers.ipynb`
A simplified version of Notebook 1 to process multiple datasets in a loop with a single click.

---

## 🔍 3. `3_Heuristic_Alpha_Evaluation.ipynb`
This notebook computes and analyzes the **heuristic α** value for hybrid retrieval (linear combination method).

### Includes:
- Extracting dataset metadata
- Computing heuristic α using the defined formula
- Comparing fixed value α and heuristic α performance

**Purpose:**  
Determining the α value which should be used in the final hybrid retriever.

---

## 📊 4. `4_Comparing_All_Retrievers.ipynb`
This notebook performs the **full comparative study** of the retrievers.

### Includes:
- Score normalization (Min–Max, Z-score, Rank)
- Ranking retrievers across all metrics (nDCG@10, MRR@10, Recall@10, Precision@10, MAP@100.)
- Producing:
  - Line plots  
  - Ranking plots
  - 
- Determining which retrievers perform best overall

**Purpose:**  
Generate final evaluation and comparison plots for reporting.

---

## 📝 Notes
- For reproducibility, run notebooks **in numerical order (1 → 4)**.

