GPU=$2
for step in 0; do
CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_pythia.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --model_name_or_path EleutherAI/pythia-$1 \
    --output_dir tmp \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --revision step${step} \
    --output_name pythia_${1}_wiki103_emb_step${step}
done

