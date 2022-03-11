gpu=0
dataset_name="eth"
num_epochs="150-100-150"
domain_shift="1-2-4-8-1"
seed=1

# Create folders
mkdir -p log
mkdir -p models/$dataset_name

# Empirical Risk Minimization (ERM)
CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset_name $dataset_name --num_epochs $num_epochs --add_confidence true --domain_shift $domain_shift --seed $seed &

# Counterfactual Analysis
CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset_name $dataset_name --num_epochs $num_epochs --counter true --add_confidence true --domain_shift $domain_shift --seed $seed &

# Invariant Risk Minimization (IRM)
for ic_weight in 1 3 5
do 
    CUDA_VISIBLE_DEVICES=$gpu python train.py --risk "irm" --dataset_name $dataset_name --num_epochs $num_epochs --ic_weight $ic_weight --add_confidence true --domain_shift $domain_shift --seed $seed &
done