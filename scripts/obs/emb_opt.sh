#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-task=18
#SBATCH -o opt_norm_v_a.out

source /home/sliu01/anaconda3/etc/profile.d/conda.sh
source activate UW

GPU=0
CUDA_VISIBLE_DEVICES=${GPU} python -u main_observation_emb_opt.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --model_name_or_path facebook/opt-6.7b \
    --output_dir tmp \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --sample_size 256 \
    --sketching_method l2_reverse \
    --output_name opt_6.7b_wiki103_att
