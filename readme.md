python3 make_vectors.py   --subset_dir ./work/subsets/beir_trec-covid/subset_1   --dense_normalize





python3 build_hybrid_sum.py   --subset_dir ./work/subsets/beir_trec-covid/subset_1   --vec_dirname vectors_local   --use_signed_hash   --normalize_parts   --lambda_dense 1.0   --lambda_sparse 1.0   --write_run   --topk 1000

python3 build_hybrid_hadamard.py   --subset_dir ./work/subsets/beir_trec-covid/subset_1   --vec_dirname vectors_local   --lambda_sparse 0.5   --use_signed_hash   --normalize_parts   --write_run   --topk 1000   --metric_k 10   --map_k 100

python3 hybrid_no_alpha.py \
  --subset_dir ./work/subsets/beir_trec-covid/subset_1 \
  --vec_dirname vectors_local \
  --topk 1000 \
  --metric_k 10 \
  --map_k 100


python3 cross_attention_fusion.py   --subset_dir ./work/subsets/beir_trec-covid/subset_1   --vec_dirname vectors_local   --max_terms 32   --train_epochs 1   --topk 1000



python3 compare_hybrids.py




