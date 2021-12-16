dataset_name="eth"
metrics="quantitative" # quantititative, qualitative, collisions
model_name="STGAT_factual_irm_0.0_data_eth_ds_1-2-4-8-1_bk_1_ep_150-100-150_seed_1.pth.tar"
best_k=20


for domain_shift in 1 2 4 8 16 32 64
do
    # Baseline (ERM)
    model_name="STGAT_factual_irm_0.0_data_eth_ds_1-2-4-8-1_bk_1_ep_150-100-150_seed_1.pth.tar"
    python evaluate_model.py --dataset_name $dataset_name --metrics $metrics --add_confidence true --domain_shift $domain_shift --resume $model_name --best_k $best_k

    # Counterfactual
    model_name="STGAT_counter_irm_0.0_data_eth_ds_1-2-4-8-1_bk_1_ep_150-100-150_seed_1.pth.tar"
    python evaluate_model.py --dataset_name $dataset_name --metrics $metrics --add_confidence true --domain_shift $domain_shift --counter true --resume $model_name --best_k $best_k

    # Invariant (IRM)
    for ic_weight in 1.0 3.0 5.0
    do
        model_name="STGAT_factual_irm_${ic_weight}_data_eth_ds_1-2-4-8-1_bk_1_ep_150-100-150_seed_1.pth.tar"
        python evaluate_model.py --dataset_name $dataset_name --metrics $metrics --add_confidence true --domain_shift $domain_shift --resume $model_name --best_k $best_k
    done
done

