
### THE TRAINING ENVS
f_envs='0.5'

### IF TRAINING OR FINETUNING
DATA="--dataset_name v4 --filter_envs $f_envs --filter_envs_pretrain 0.1-0.3-0.5"

### NAME OF EXPERIMENT
expname="refinement"
DIR="--tfdir runs/$expname --model_dir models/$expname"

### EXPERIMENT HYPERPARAM
irm=0.0
bs=64
TRAINING="--irm $irm --batch_size $bs --shuffle true --seed 72 --contrastive 1"


### NUM EPOCHS AND SET THE REDUCE (--reduce to only reduce the training envs, --reducall to reduce both train and val datasets)
EPOCH='--num_epochs 0-0-100-50-20-300 --reduceall 64'

### MODEL TO LOAD (if want to start again from checkpoint)
MOD='--resume MODEL_NAME'

### START SCRIPT

CUDA_VISIBLE_DEVICES=0 python train.py $DATA $TRAINING $DIR $EPOCH $MOD --testonly 3 --ttr 3 --ttrlr 0.001 --wrongstyle true & # refine with wrong style, ADE will increase
CUDA_VISIBLE_DEVICES=1 python train.py $DATA $TRAINING $DIR $EPOCH $MOD --testonly 3 --ttr 3 --ttrlr 0.001 # refine with right style, ADE will decrease
