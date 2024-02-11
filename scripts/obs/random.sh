GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_attention_sparsity.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --model_name_or_path facebook/opt-2.7b \
    --output_dir tmp \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --output_name opt-2.7b-wiki103-random \
    --random_init

# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_attention_sparsity_pythia.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path EleutherAI/pythia-70m \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --output_name pythia_70m_wiki103_random \
#     --random_init

# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_attention_sparsity_pythia.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path EleutherAI/pythia-1.4b \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --output_name pythia_1.4b_wiki103_random \
#     --random_init

# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_attention_sparsity_pythia.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path EleutherAI/pythia-12b \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --output_name pythia_12b_wiki103_random \
#     --random_init
