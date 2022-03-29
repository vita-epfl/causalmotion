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
finetune='all' # 2. Set module updated during finetuning
irm=1.0 # 3. Set IRM (used in training)
step='P6'
for reduce in  384 #0 64 128 192 256 320
do
    # for seed in 1 2 3 4
    # do
    #     CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume models/$dataset/$exp/$step/$irm/$finetune/$seed/model_t$reduce.pth.tar --seed $seed &
    # done
    seed=1
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --finetune true --resume models/$dataset/$exp/$step/$irm/$finetune/$seed/model_t$reduce.pth.tar --seed $seed
done


# step="P6" 
# reduceall=128
# epochs_string='0-0-5-2-2-5'
# epoch=14 #470 
# irm=1.0 # 2. Set IRM
# for f_envs in "0.1" # "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8"
# do
#     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
#     for seed in 1 #2 3 4
#     do  
#         CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_[$epochs_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$reduceall]_relsocial[True]stylefs[all].pth.tar" --seed $seed #&
#     done
#     # seed=5
#     # CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_((0, 0, 100, 50, 20, 300))_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[9000]_relsocial[True]stylefs[all].pth.tar" --seed $seed
# done








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