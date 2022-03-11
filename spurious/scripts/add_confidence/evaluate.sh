# Set hyper-parameters
gpu=0
dataset_name="eth"
domain_shifts="1-2-4-8-1"
num_epochs="150-100-150"
metrics="quantitative" # quantititative, qualitative, collisions
seed=1

# Create Folders
mkdir -p log/$dataset_name

for domain_shift in 1 2 4 8 16 32 64
do
    # Empirical Risk Minimization (ERM)
    model_name="STGAT_factual_erm_0.0_data_eth_hom_ds_${domain_shifts}_bk_1_ep_${num_epochs}_seed_${seed}.pth.tar"
    CUDA_VISIBLE_DEVICES=$gpu python evaluate_model.py --dataset_name $dataset_name --metrics $metrics --add_confidence true --domain_shift $domain_shift --resume $model_name

    # Counterfactual Analysis
    model_name="STGAT_counter_erm_0.0_data_eth_hom_ds_${domain_shifts}_bk_1_ep_${num_epochs}_seed_${seed}.pth.tar"
    CUDA_VISIBLE_DEVICES=$gpu python evaluate_model.py --dataset_name $dataset_name --metrics $metrics --add_confidence true --domain_shift $domain_shift --counter true --resume $model_name &

    # Invariant Risk Minimization (IRM)
    for ic_weight in 1.0 3.0 5.0
    do
        model_name="STGAT_factual_irm_${ic_weight}_data_eth_hom_ds_${domain_shifts}_bk_1_ep_${num_epochs}_seed_${seed}.pth.tar"
        CUDA_VISIBLE_DEVICES=$gpu python evaluate_model.py --dataset_name $dataset_name --metrics $metrics --add_confidence true --domain_shift $domain_shift --resume $model_name &
    done
done