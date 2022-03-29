# FINE TUNE

# MLP + V4

exp='finetune'
GPU=0 # 0. Set GPU

# Our ERM

# data
filter_envs='0.6' # 1. Set env(s) to filter for training
filter_envs_pretrain='0.1-0.3-0.5'
dataset='v4'
DATA="--dataset_name $dataset --filter_envs $filter_envs --filter_envs_pretrain $filter_envs_pretrain"

# training
step='P6'
bs=64
p6=30
finetune='all' # 2. Set which module to finetune (i.e. 'all','integ+','integ')
lrinteg=0.001
contrastive=0.05
epoch=14
epoch_string='0-0-5-2-2-5'
oldreduceall=128

irm=1.0 # 3. Set IRM (used in pretraining)
TRAINING="--num_epochs $epoch-0-0-0-0-$p6 --batch_size $bs --finetune $finetune --lrinteg $lrinteg --contrastive $contrastive --irm $irm"
    
for seed in 1 #2 3 4 5
do
    # pretrained model
    model_dir="./models/$dataset/$exp/$step/$irm/$finetune/$seed"
    DIR="--tfdir runs/$dataset/$exp/$step/$irm/$finetune/$seed"

    # for reduce in 64 #128 192 256 320 
    # do
    #     CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $MODEL $DIR --reduce $reduce --resume "models/$dataset/pretrain/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_[$epoch_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[9000]_relsocial[True]stylefs[all].pth.tar" &
    # done
    CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $MODEL $DIR --reduce 384 --original_seed $seed --resume "models/$dataset/pretrain/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_[$epoch_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$oldreduceall]_relsocial[True]stylefs[all].pth.tar"
done



#################

# # STGAT + V2

# exp='finetune'
# GPU=0 # 1. Set GPU

# # Our ERM

# # data
# filter_envs='0.6'
# filter_envs_pretrain='0.1-0.3-0.5'
# dataset='v2'
# DATA="--dataset_name $dataset --filter_envs $filter_envs --filter_envs_pretrain $filter_envs_pretrain"

# # training
# bs=64
# p6=30
# finetune='all' # 2. Set which module to finetune (i.e. 'all' or 'decoder')
# TRAINING="--num_epochs 210-0-0-0-0-$p6 --batch_size $bs --finetune $finetune"


# for seed in 1 2 3 4 5
# do
#     # pretrained model
#     irm=0.0
#     STYLE="--styleinteg concat" 
#     model_dir="./models/$dataset/$exp/P6/$irm/$finetune/$seed/"
#     DIR="--tfdir runs/$dataset/$exp/P6/$irm/$finetune/$seed --model_dir $model_dir"
#     modelname="STGAT_data_v2_irm[${irm}]_ep_(25-25-40-40-40-40)_seed_${seed}_tstep_P6_epoch_210_aggrstyle[minpol-mean]_styleinteg[concat]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar"
#     MODEL="--resume models/$dataset/pretrain/P6/$irm/$modelname"

#     for reduce in 2 4 8 16 32
#     do
#         CUDA_VISIBLE_DEVICES=$GPU python train_all.py $DATA $TRAINING $STYLE $MODEL $DIR --reduce $reduce &
#     done
#     CUDA_VISIBLE_DEVICES=$GPU python train_all.py $DATA $TRAINING $STYLE $MODEL $DIR --reduce 64
# done