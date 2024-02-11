GPU=1
for step in 0 1 2 4 8 16 32 64 128 512 1000 2000 5000 10000 20000 50000 100000 110000 120000 130000 140000; do
CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_attention_sparsity_pythia.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --model_name_or_path EleutherAI/pythia-1.4b \
    --output_dir tmp \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --revision step${step} \
    --output_name pythia_1.4b_wiki103_attn_entropy_step${step} \
    --cache_dir $1
done


