
# for iteration in 2_10000 4_20000 6_30000 7_40000 9_50000 17_100000 26_150000 34_200000 43_250000
# do
# for part in 0 1 2 3
# do
# for((i=0;i<=63;i++))
# do
# wget https://dl.fbaipublicfiles.com/OPT/2.7B/checkpoint_${iteration}-model_part-${part}-shard${i}.pt
# done
# done
# done 

# for iteration in 2_10000 4_20000 6_30000 7_40000 9_50000 17_100000 26_150000 34_200000 43_250000
# do
# for part in 3
# do
# for((i=0;i<=63;i++))
# do
# wget https://dl.fbaipublicfiles.com/OPT/2.7B/checkpoint_${iteration}-model_part-${part}-shard${i}.pt
# done
# done
# done 

for iteration in 16_90000
do
for part in 0 1 2 3
do
for((i=0;i<=63;i++))
do
wget https://dl.fbaipublicfiles.com/OPT/2.7B/checkpoint_${iteration}-model_part-${part}-shard${i}.pt
done
done
done 