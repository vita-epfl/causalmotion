# Set hyper-parameters
gpu=0
dataset_name="hotel" 
metrics="quantitative" # {quantititative, qualitative, collisions}
batch_type="het"
num_epochs="300-200-300"
seed="1"

# Create Folders
mkdir -p log/$dataset_name

for dset_type in "train" "val" "test"
do
    # Empirical Risk Minimization (ERM)
    model_name="STGAT_factual_erm_0.0_data_${dataset_name}_${batch_type}_ds_0_bk_1_ep_${num_epochs}_seed_${seed}.pth.tar"
    CUDA_VISIBLE_DEVICES=$gpu python evaluate_model.py --dataset_name $dataset_name --metrics $metrics --resume $model_name --dset_type $dset_type

    # Counterfactual Analysis
    model_name="STGAT_counter_erm_0.0_data_${dataset_name}_${batch_type}_ds_0_bk_1_ep_${num_epochs}_seed_${seed}.pth.tar"
    CUDA_VISIBLE_DEVICES=$gpu python evaluate_model.py --dataset_name $dataset_name --metrics $metrics --counter true --resume $model_name --dset_type $dset_type

    # Invariant Risk Minimization (IRM)
    for ic_weight in 0.1 0.5 1.0 
    do
        model_name="STGAT_factual_irm_${ic_weight}_data_${dataset_name}_${batch_type}_ds_0_bk_1_ep_${num_epochs}_seed_${seed}.pth.tar"
        CUDA_VISIBLE_DEVICES=$gpu python evaluate_model.py --dataset_name $dataset_name --metrics $metrics --resume $model_name --dset_type $dset_type 
    done

    # Risk Extrapolation (v-REx)
    # for ic_weight in 1.0 5.0 10.0 20.0
    # do
    #     model_name="STGAT_factual_vrex_${ic_weight}_data_${dataset_name}_het_ds_0_bk_1_ep_${num_epochs}_seed_${seed}.pth.tar"
    #     CUDA_VISIBLE_DEVICES=$gpu python evaluate_model.py --dataset_name $dataset_name --metrics $metrics --resume $model_name --dset_type $dset_type 
    # done
done