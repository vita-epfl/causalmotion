# PRETRAIN MODELS (v4 MLP)

GPU=2 # 0. Set GPU

EVALUATION="--metrics accuracy"
exp="pretrain"
dataset="v4" # 1. Set Dataset
dset_type="test"
bs=64

# Baseline

step="P3" 
epoch=100 
irm=0.0 # 2. Set IRM
for f_envs in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8"
do
    DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
    for seed in 1 2 3 4
    do  
        CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_((0, 0, 100, 50, 20, 300))_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[9000]_relsocial[True]stylefs[all].pth.tar" --seed $seed &
    done
    seed=5
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_((0, 0, 100, 50, 20, 300))_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[9000]_relsocial[True]stylefs[all].pth.tar" --seed $seed 
done

# Modular Architecture (Our)

step="P6" 
epoch=470 
irm=0.0 # 2. Set IRM
for f_envs in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8"
do
    DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
    for seed in 1 2 3 4
    do  
        CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_((0, 0, 100, 50, 20, 300))_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[9000]_relsocial[True]stylefs[all].pth.tar" --seed $seed &
    done
    seed=5
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_((0, 0, 100, 50, 20, 300))_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[9000]_relsocial[True]stylefs[all].pth.tar" --seed $seed
done


# ####################################

# # # PRETRAIN MODELS (v2+stgat)

# # EVALUATION="--metrics accuracy"
# # exp="pretrain"
# # dataset="v4" # 1. Set Dataset
# # dset_type="test"
# # bs=64

# # # STGAT + ERM

# # step="P3"
# # irm=0.0 
# # for seed in 1 2 3 4 5
# # do  
# #     f_envs="0.1-0.3-0.5"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg none --seed $seed &

# #     f_envs="0.4"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg none --seed $seed &
    
# #     f_envs="0.6"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg none --seed $seed 
# # done

# # ## STGAT + IRM 0.1

# # step="P3"
# # irm=0.1
# # for seed in 1 2 3 4 5
# # do  
# #     f_envs="0.1-0.3-0.5"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg none --seed $seed &

# #     f_envs="0.4"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg none --seed $seed &
    
# #     f_envs="0.6"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg none --seed $seed 
# # done

# # ## STGAT + IRM 1.0

# # step="P3"
# # irm=1.0
# # for seed in 1 2 3 4 5
# # do  
# #     f_envs="0.1-0.3-0.5"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg none --seed $seed &

# #     f_envs="0.4"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg none --seed $seed &
    
# #     f_envs="0.6"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg none --seed $seed 
# # done

# # ## STGAT + Counter

# # step="P3"
# # irm=0.0
# # for seed in 1 2 3 4 5
# # do  
# #     f_envs="0.1-0.3-0.5"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/counter/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[True].pth.tar" --styleinteg none --counter true --seed $seed &

# #     f_envs="0.4"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/counter/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[True].pth.tar" --styleinteg none --counter true --seed $seed &
    
# #     f_envs="0.6"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/counter/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-0-0-0)_seed_${seed}_tstep_P3_epoch_90_aggrstyle[minpol-mean]_styleinteg[none]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[True].pth.tar" --styleinteg none --counter true --seed $seed 
# # done

# # ## Our + ERM

# # step="P6"
# # irm=0.0
# # for seed in 1 2 3 4 5
# # do  
# #     f_envs="0.1-0.3-0.5"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-40-40-40)_seed_${seed}_tstep_P6_epoch_210_aggrstyle[minpol-mean]_styleinteg[concat]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg concat --seed $seed &

# #     f_envs="0.4"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-40-40-40)_seed_${seed}_tstep_P6_epoch_210_aggrstyle[minpol-mean]_styleinteg[concat]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg concat --seed $seed &
    
# #     f_envs="0.6"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-40-40-40)_seed_${seed}_tstep_P6_epoch_210_aggrstyle[minpol-mean]_styleinteg[concat]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg concat --seed $seed 
# # done

# # ## Our + ERM

# # step="P6"
# # irm=0.1
# # for seed in 1 2 3 4 5
# # do  
# #     f_envs="0.1-0.3-0.5"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-40-40-40)_seed_${seed}_tstep_P6_epoch_210_aggrstyle[minpol-mean]_styleinteg[concat]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg concat --seed $seed &

# #     f_envs="0.4"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-40-40-40)_seed_${seed}_tstep_P6_epoch_210_aggrstyle[minpol-mean]_styleinteg[concat]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg concat --seed $seed &
    
# #     f_envs="0.6"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-40-40-40)_seed_${seed}_tstep_P6_epoch_210_aggrstyle[minpol-mean]_styleinteg[concat]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg concat --seed $seed 
# # done

# # ## Our + ERM

# # step="P6"
# # irm=1.0
# # for seed in 1 2 3 4 5
# # do  
# #     f_envs="0.1-0.3-0.5"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-40-40-40)_seed_${seed}_tstep_P6_epoch_210_aggrstyle[minpol-mean]_styleinteg[concat]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg concat --seed $seed &

# #     f_envs="0.4"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-40-40-40)_seed_${seed}_tstep_P6_epoch_210_aggrstyle[minpol-mean]_styleinteg[concat]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg concat --seed $seed &
    
# #     f_envs="0.6"
# #     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
# #     CUDA_VISIBLE_DEVICES=0 python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/STGAT_data_v2_irm[${irm}]_ep_(25-25-40-40-40-40)_seed_${seed}_tstep_P6_epoch_210_aggrstyle[minpol-mean]_styleinteg[concat]_addloss[0]_lrinteg[0.001]_lrstgat[0.001]_contrastive[0.1]_counter[False].pth.tar" --styleinteg concat --seed $seed 
# # done