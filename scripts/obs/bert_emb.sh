# GPU=$1
# for key in step_0 step_1100000 step_1300000 step_1500000 step_1700000 step_1900000 step_2000000 step_400000 step_600000 step_800000 step_100000 step_120000 step_140000 step_160000 step_180000 step_20000 step_300000 step_500000 step_700000 step_900000 step_1000000 step_1200000 step_1400000 step_1600000 step_1800000 step_200000 step_40000 step_60000 step_80000; do
# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_bert.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path bert-base-uncased \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --random_init \
#     --load_weight bert_seed0_${key}.pt \
#     --output_name bert_base_wiki103_${key}
# done 


# GPU=0
# for key in step_0 step_1100000 step_1300000 step_1500000 step_1700000 step_1900000 step_2000000 step_400000 step_600000 step_800000 step_100000 step_120000 step_140000 step_160000 step_180000 step_20000 step_300000 step_500000 step_700000 step_900000 step_1000000 step_1200000 step_1400000 step_1600000 step_1800000 step_200000 step_40000 step_60000 step_80000; do
# CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_bert.py \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --model_name_or_path bert-base-uncased \
#     --output_dir tmp \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --random_init \
#     --load_weight bert_seed0_${key}.pt \
#     --output_name bert_base_wiki103_${key}
# done 


GPU=0
CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_bert.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --model_name_or_path bert-base-uncased \
    --output_dir tmp \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --random_init \
    --output_name bert_base_wiki103_random_init

