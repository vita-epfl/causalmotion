# Set hyper-parameters
gpu=0
dataset_name="hotel"
num_epochs="300-200-300"
seed=1

# Create folders
mkdir -p log
mkdir -p models/$dataset_name

# Empirical Risk Minimization (ERM)
CUDA_VISIBLE_DEVICES=$gpu python train.py --risk "erm" --dataset_name $dataset_name --seed $seed --batch_hetero true --num_epochs $num_epochs &

# Counterfactual Analysis
CUDA_VISIBLE_DEVICES=$gpu python train.py --counter true --risk "erm" --dataset_name $dataset_name --seed $seed --batch_hetero true --num_epochs $num_epochs &

# Invariant Risk Minimization (IRM)
for ic_weight in 0.1 0.5 1
do 
    CUDA_VISIBLE_DEVICES=$gpu python train.py --risk "irm" --dataset_name $dataset_name --ic_weight $ic_weight --seed $seed --batch_hetero true --num_epochs $num_epochs &
done

# Risk Extrapolation (v-REx)
# for ic_weight in 1 5 10 20
# do 
#     CUDA_VISIBLE_DEVICES=$gpu python train.py --risk "vrex" --dataset_name $dataset_name --ic_weight $ic_weight --seed $seed --batch_hetero true --num_epochs $num_epochs & 
# done