CUDA_VISIBLE_DEVICES=3 python -u main_observation_emb_vit.py \
    --dataset_name cats_vs_dogs \
    --output_dir ./cats_vs_dogs_outputs/ \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --model_name_or_path google/vit-huge-patch14-224-in21k \
    --output_name cats_vs_dogs_obs_emb_vit_huge
