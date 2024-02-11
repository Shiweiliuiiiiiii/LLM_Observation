GPU=1
# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_llama.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --output_name llama_2_7b_wiki103_emb


# for min in 1 0.5 0.8; do
# for max in 1 2 3; do
# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_llama_position.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --compress_ratio_max ${max} \
#     --compress_ratio_min ${min} \
#     --output_name llama_2_7b_wiki103_emb_position_${min}_${max}
# done
# done 




# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_llama_position.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --compress_ratio_max 1.2 \
#     --compress_ratio_min 0.8 \
#     --output_name llama_2_7b_wiki103_emb_position_${min}_${max}

# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_llama_position.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --compress_ratio_max 1.5 \
#     --compress_ratio_min 0.8 \
#     --output_name llama_2_7b_wiki103_emb_position_${min}_${max}

# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_llama_position.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --compress_ratio_max 2.5 \
#     --compress_ratio_min 1.5 \
#     --output_name llama_2_7b_wiki103_emb_position_${min}_${max}


# for min in 0.1 0.3 0.5 0.7 0.9; do
# for max in 0.1 0.3 0.5 0.7 0.9; do
# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_llama_position.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --compress_ratio_max ${max} \
#     --compress_ratio_min ${min} \
#     --output_name llama_2_7b_wiki103_emb_position_${min}_${max}
# done
# done 


GPU=$1
min=$2
for max in 1 1.1 1.2 1.3 1.5; do
CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_llama_t.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir tmp \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --t_min ${min} \
    --t_max ${max} \
    --output_name llama_2_7b_wiki103_emb_temperature_${min}_${max}
done
