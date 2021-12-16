dataset_name="eth"
num_epochs="150-100-150"
domain_shift="1-2-4-8-1"

# Baseline (ERM)
python train.py --dataset_name $dataset_name --num_epochs $num_epochs --add_confidence true --domain_shift $domain_shift

# Counterfactual
python train.py --dataset_name $dataset_name --num_epochs $num_epochs --counter true --add_confidence true --domain_shift $domain_shift

# Invariant (IRM)
for ic_weight in 1 3 5
do 
    python train.py --dataset_name $dataset_name --num_epochs $num_epochs --ic_weight $ic_weight --add_confidence true --domain_shift $domain_shift
done
