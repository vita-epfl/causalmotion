# TRANSFER PROTOCOL

# MLP + V4

GPU=2 # 1. Set GPU


# EVALUATION='--metrics accuracy'

# f_envs='0.6'
# dataset='v2'
# bs=64
# DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs"

# exp='finetune' 
# finetune='decoder' # 2. Set module to finetune
# irm=0.0
# step='P3'
# for reduce in 0 2 4 8 16 32 64
# do
#     for seed in 1 2 3 4
#     do
#         CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume models/$dataset/$exp/$step/$irm/$finetune/$seed/model_t$reduce.pth.tar --styleinteg none --seed $seed &
#     done
#     seed=5
#     CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume models/$dataset/$exp/$step/$irm/$finetune/$seed/model_t$reduce.pth.tar --styleinteg none --seed $seed
# done

####

# Modular Architecture (Our)

EVALUATION='--metrics accuracy'

f_envs='0.6'
dataset='v4'
bs=64
DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs"

exp='finetune' 
finetune='integ' # 2. Set module updated during finetuning
irm=0.1 # 3. Set IRM (used in training)
step='P6'
for reduce in 0 64 128 192 256 320 384
do
    for seed in 1 2 3 4
    do
        CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume models/$dataset/$exp/$step/$irm/$finetune/$seed/model_t$reduce.pth.tar --seed $seed &
    done
    seed=5
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume models/$dataset/$exp/$step/$irm/$finetune/$seed/model_t$reduce.pth.tar --seed $seed
done


#######################

# STGAT + V2

# GPU=3 # 1. Set GPU

# STGAT ERM 

# EVALUATION='--metrics accuracy'

# f_envs='0.6'
# dataset='v2'
# bs=64
# DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs"

# exp='finetune' 
# finetune='decoder' # 2. Set module to finetune
# irm=0.0
# step='P3'
# for reduce in 0 2 4 8 16 32 64
# do
#     for seed in 1 2 3 4
#     do
#         CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume models/$dataset/$exp/$step/$irm/$finetune/$seed/model_t$reduce.pth.tar --styleinteg none --seed $seed &
#     done
#     seed=5
#     CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume models/$dataset/$exp/$step/$irm/$finetune/$seed/model_t$reduce.pth.tar --styleinteg none --seed $seed
# done

####

# Our ERM 

# EVALUATION='--metrics accuracy'

# f_envs='0.6'
# dataset='v2'
# bs=64
# DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs"

# exp='finetune' 
# finetune='style' # 2. Set module to finetune
# irm=0.0
# step='P6'
# for reduce in 0 2 4 8 16 32 64
# do
#     for seed in 1 2 3 4
#     do
#         CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume models/$dataset/$exp/$step/$irm/$finetune/$seed/model_t$reduce.pth.tar --styleinteg concat --seed $seed &
#     done
#     seed=5
#     CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume models/$dataset/$exp/$step/$irm/$finetune/$seed/model_t$reduce.pth.tar --styleinteg concat --seed $seed
# done