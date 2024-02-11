# name=2_10000
# iterations=10000
# for j in {0..2}; do
#     python -m metaseq.scripts.reshard_fsdp \
#     --input "../opt-2.7-${iterations}/checkpoint_${name}-model_part-$j-shard*.pt" \
#     --output "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-$j.pt" \
#     --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
# done

# name=2_10000
# iterations=10000
# for j in {0..3}; do
# python -m metaseq.scripts.reshard_fsdp \
#     --input "../opt-2.7-${iterations}/checkpoint_${name}-model_part-$j-shard*.pt" \
#     --output "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-$j.pt" \
#     --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
# done
# python -m metaseq.scripts.reshard_mp \
#     --input "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-*.pt" \
#     --output "opt-2.7-${iterations}-resharded/pytorch_model.bin" \
#     --num-output-parts 1

# name=4_20000
# iterations=20000
# for j in {0..3}; do
# python -m metaseq.scripts.reshard_fsdp \
#     --input "../opt-2.7-${iterations}/checkpoint_${name}-model_part-$j-shard*.pt" \
#     --output "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-$j.pt" \
#     --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
# done
# python -m metaseq.scripts.reshard_mp \
#     --input "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-*.pt" \
#     --output "opt-2.7-${iterations}-resharded/pytorch_model.bin" \
#     --num-output-parts 1

# name=6_30000
# iterations=30000
# for j in {0..3}; do
# python -m metaseq.scripts.reshard_fsdp \
#     --input "../opt-2.7-${iterations}/checkpoint_${name}-model_part-$j-shard*.pt" \
#     --output "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-$j.pt" \
#     --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
# done
# python -m metaseq.scripts.reshard_mp \
#     --input "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-*.pt" \
#     --output "opt-2.7-${iterations}-resharded/pytorch_model.bin" \
#     --num-output-parts 1

# name=7_40000
# iterations=40000
# for j in {0..3}; do
# python -m metaseq.scripts.reshard_fsdp \
#     --input "../opt-2.7-${iterations}/checkpoint_${name}-model_part-$j-shard*.pt" \
#     --output "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-$j.pt" \
#     --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
# done
# python -m metaseq.scripts.reshard_mp \
#     --input "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-*.pt" \
#     --output "opt-2.7-${iterations}-resharded/pytorch_model.bin" \
#     --num-output-parts 1

# name=9_50000
# iterations=50000
# for j in {0..3}; do
# python -m metaseq.scripts.reshard_fsdp \
#     --input "../opt-2.7-${iterations}/checkpoint_${name}-model_part-$j-shard*.pt" \
#     --output "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-$j.pt" \
#     --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
# done
# python -m metaseq.scripts.reshard_mp \
#     --input "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-*.pt" \
#     --output "opt-2.7-${iterations}-resharded/pytorch_model.bin" \
#     --num-output-parts 1

name=16_90000
iterations=90000
for j in {0..3}; do
python -m metaseq.scripts.reshard_fsdp \
    --input "../opt-2.7-${iterations}/checkpoint_${name}-model_part-$j-shard*.pt" \
    --output "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-$j.pt" \
    --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
done
python -m metaseq.scripts.reshard_mp \
    --input "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-*.pt" \
    --output "opt-2.7-${iterations}-resharded/pytorch_model.bin" \
    --num-output-parts 1

# name=26_150000
# iterations=150000
# for j in {0..3}; do
# python -m metaseq.scripts.reshard_fsdp \
#     --input "../opt-2.7-${iterations}/checkpoint_${name}-model_part-$j-shard*.pt" \
#     --output "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-$j.pt" \
#     --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
# done
# python -m metaseq.scripts.reshard_mp \
#     --input "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-*.pt" \
#     --output "opt-2.7-${iterations}-resharded/pytorch_model.bin" \
#     --num-output-parts 1

# name=34_200000
# iterations=200000
# for j in {0..3}; do
# python -m metaseq.scripts.reshard_fsdp \
#     --input "../opt-2.7-${iterations}/checkpoint_${name}-model_part-$j-shard*.pt" \
#     --output "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-$j.pt" \
#     --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
# done
# python -m metaseq.scripts.reshard_mp \
#     --input "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-*.pt" \
#     --output "opt-2.7-${iterations}-resharded/pytorch_model.bin" \
#     --num-output-parts 1

# name=43_250000
# iterations=250000
# for j in {0..3}; do
# python -m metaseq.scripts.reshard_fsdp \
#     --input "../opt-2.7-${iterations}/checkpoint_${name}-model_part-$j-shard*.pt" \
#     --output "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-$j.pt" \
#     --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
# done
# python -m metaseq.scripts.reshard_mp \
#     --input "opt-2.7-${iterations}-resharded/checkpoint_${name}-model_part-*.pt" \
#     --output "opt-2.7-${iterations}-resharded/pytorch_model.bin" \
#     --num-output-parts 1