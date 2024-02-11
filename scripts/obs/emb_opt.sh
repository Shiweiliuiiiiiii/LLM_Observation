GPU=3
CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_opt.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --model_name_or_path facebook/opt-6.7b \
    --output_dir tmp \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --output_name opt_6.7b_wiki103_emb
