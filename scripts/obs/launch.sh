
# nohup bash scripts/obs/attn_sparsity.sh hf_2.7b-10000 opt-2.7b-1w-wiki103 0 > log_opt_2.7b_1w.out 2>&1 &
# nohup bash scripts/obs/attn_sparsity.sh hf_2.7b-20000 opt-2.7b-2w-wiki103 3 > log_opt_2.7b_2w.out 2>&1 &
# nohup bash scripts/obs/attn_sparsity.sh hf_2.7b-30000 opt-2.7b-3w-wiki103 2 > log_opt_2.7b_3w.out 2>&1 &
# nohup bash scripts/obs/attn_sparsity.sh hf_2.7b-40000 opt-2.7b-4w-wiki103 3 > log_opt_2.7b_4w.out 2>&1 &
# nohup bash scripts/obs/attn_sparsity.sh hf_2.7b-50000 opt-2.7b-5w-wiki103 1 > log_opt_2.7b_5w.out 2>&1 &
# nohup bash scripts/obs/attn_sparsity.sh hf_2.7b-150000 opt-2.7b-15w-wiki103 2 > log_opt_2.7b_15w.out 2>&1 &
# nohup bash scripts/obs/attn_sparsity.sh hf_2.7b-200000 opt-2.7b-20w-wiki103 1 > log_opt_2.7b_20w.out 2>&1 &
# nohup bash scripts/obs/attn_sparsity.sh hf_2.7b-250000 opt-2.7b-25w-wiki103 0 > log_opt_2.7b_25w.out 2>&1 &





# nohup bash scripts/obs/attn_sparsity.sh hf_2.7b-90000 opt-2.7b-9w-wiki103 1 > log_opt_2.7b_9w.out 2>&1 &




# GPU=3
# for step in 0 1 2 4 8 16 32 64 128 512 1000 2000 5000 10000 20000 50000 100000 110000 120000 130000 140000; do
# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_attention_sparsity_pythia.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path EleutherAI/pythia-70m \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --revision step${step} \
#     --output_name pythia_70m_wiki103_attn_entropy_step${step}
# done


# GPU=0
# for step in 0 1 2 4 8 16 32 64 128 512 1000 2000 5000 10000 20000 50000 100000 110000 120000 130000 140000; do
# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_attention_sparsity_pythia.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path EleutherAI/pythia-12b \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --revision step${step} \
#     --output_name pythia_12b_wiki103_attn_entropy_step${step} \
#     --cache_dir $1
# done

GPU=0
for step in 0 1 2 4 8 16 32 64 128 512 1000 2000 5000 10000 20000 50000 100000 110000 120000 130000 140000; do
CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_attention_sparsity_pythia.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --model_name_or_path EleutherAI/pythia-6.9b \
    --output_dir tmp \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --revision step${step} \
    --output_name pythia_6.9b_wiki103_attn_entropy_step${step} \
    --cache_dir $1
done