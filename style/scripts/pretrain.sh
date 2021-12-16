# PRETRAIN
GPU=0 # 1. Set GPU
exp='pretrain'

# SSE (ERM, IRM)

dataset='v4' # 2. Set dataset
f_envs='0.1-0.3-0.5'
#DATA="--dataset_name $dataset --filter_envs $f_envs"
DATA="--dataset_name $dataset --filter_envs $f_envs --reduceall 9000"


#USUAL="--contrastive 0.1 --styleinteg none" 
USUAL="--contrastive 1" 

bs=64
e='0-0-100-50-20-300'
irm=1.0 # 3. Set IRM weight
TRAINING="--num_epochs $e --batch_size $bs --irm $irm"
#e='0-0-100-0-0-0'
#TRAINING="--num_epochs $e --batch_size $bs --counter true" # 4. Set Counter

DIR="--tfdir runs/$dataset/$exp/$irm"
#DIR="--tfdir runs/$dataset/$exp/counter" # 5. Set counter

for seed in 1 2 3 4 5
do  
    CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed $seed &
done


# ## Our

# dataset='v2'
# f_envs='0.1-0.3-0.5'
# DATA="--dataset_name $dataset --filter_envs $f_envs"

# USUAL="--contrastive 0.1 --styleinteg none --newstyleinteg concat --reduce 9000" 

# bs=64
# e='25-25-40-40-40-40'
# irm=1.0 # 1. Set IRM weight
# TRAINING="--num_epochs $e --batch_size $bs --irm $irm"

# DIR="--tfdir runs/$dataset/$exp/$irm"

# for seed in 1 2 3 4 5
# do  
#     MODEL="--resume models/v2/$exp/P3/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar"
#     # 2. Set GPU
#     CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed $seed &
# done









################

# f_envs='0.1-0.3-0.5'
# DATA="--dataset_name v2 --filter_envs $f_envs --aggrstyle minpol-mean"

# bs=64
# e='25-25-40-20-20-30'
# irm=4
# TRAINING="--num_epochs $e --batch_size $bs --irm $irm"

# styleinteg="adain"
# STYLE="--styleinteg $styleinteg --newstyleinteg adainnew"

# MODEL="--resume 'models_trained/$styleinteg/$irm.0/model.pth.tar'"

# DIR="--tfdir runs/reprodcomp"

# python train.py $DATA $TRAINING $STYLE $MODEL $DIR