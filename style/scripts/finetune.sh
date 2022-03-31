# FINE TUNE
exp='finetune'
GPU=0 # 0. Set GPU

# data
filter_envs='0.6' # 1. Set env(s) to filter for training
filter_envs_pretrain='0.1-0.3-0.5'
dataset='v4'
DATA="--dataset_name $dataset --filter_envs $filter_envs --filter_envs_pretrain $filter_envs_pretrain"
bs=64

# training
step='P6'
lrinteg=0.001
contrastive=0.05

## TO CHANGE DEPENDING ON PREVIOUS STEPS
p6=300 # number of finetuning steps
oldreduceall=9000
epoch_string='0-0-100-50-20-300'
epoch=470 # sum of above
irm=1.0 # 3. Set IRM (used in pretraining)

for finetune in 'all' 'integ+' 'integ'
do
    for seed in 1 2 3 4 5
    do
        # pretrained model
        model_dir="./models/$dataset/$exp/$step/$irm/$finetune/$seed"
        DIR="--tfdir runs/$dataset/$exp/$step/$irm/$finetune/$seed"

        TRAINING="--num_epochs $epoch-0-0-0-0-$p6 --batch_size $bs --finetune $finetune --lrinteg $lrinteg --contrastive $contrastive --irm $irm"

        for reduce in 64 128 192 256 320
        do
            CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $MODEL $DIR --reduce $reduce --original_seed $seed --resume "models/$dataset/pretrain/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_[$epoch_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$oldreduceall]_relsocial[True]stylefs[all].pth.tar" &
        done
        CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $MODEL $DIR --reduce 384 --original_seed $seed --resume "models/$dataset/pretrain/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_[$epoch_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$oldreduceall]_relsocial[True]stylefs[all].pth.tar"
    done
done