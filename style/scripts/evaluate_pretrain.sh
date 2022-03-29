# PRETRAIN MODELS (v4 MLP)

GPU=2 # 0. Set GPU

EVALUATION="--metrics accuracy"
exp="pretrain"
dataset="v4" # 1. Set Dataset
dset_type="test"
bs=64

# Baseline

# step="P3" 
# epoch=100 
# irm=0.0 # 2. Set IRM
# for f_envs in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8"
# do
#     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
#     for seed in 1 2 3 4
#     do  
#         CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_((0, 0, 100, 50, 20, 300))_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[9000]_relsocial[True]stylefs[all].pth.tar" --seed $seed &
#     done
#     seed=5
#     CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_((0, 0, 100, 50, 20, 300))_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[9000]_relsocial[True]stylefs[all].pth.tar" --seed $seed 
# done

# Modular Architecture (Our)

step="P6" 
reduceall=128
epochs_string='0-0-5-2-2-5'
epoch=14 #470 
irm=1.0 # 2. Set IRM
for f_envs in "0.1" # "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8"
do
    DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
    for seed in 1 #2 3 4
    do  
        CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_[$epochs_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$reduceall]_relsocial[True]stylefs[all].pth.tar" --seed $seed #&
    done
    # seed=5
    # CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_((0, 0, 100, 50, 20, 300))_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[9000]_relsocial[True]stylefs[all].pth.tar" --seed $seed
done
