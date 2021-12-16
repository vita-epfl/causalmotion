# Towards Robust Motion Forecasting: A Causal Representation Perspective 



## Introduction

Promote the robustness of motion representations under **spurious shifts** from a causal perspective.

The code is built upon a recent paper `Human Trajectory Prediction via Counterfactual Analysis, ICCV'21`.

## Requirements
```
pip install --upgrade pip

pip install -r requirements.txt

pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```


## Dataset

Synthetic data is currently saved in `/datasets/`.


## Training
Train STGAT with Empirical Risk Minimization (ERM), Counterfactual Analysis and Invariant Risk Minimization (IRM).
```
sh scripts/train.sh
```

Pretrained models on dataset `eth` for all the methods with seed=1,10,20,50,100 are already saved in `/models/`.


## Evaluation
Evaluate a model.
```
sh scripts/evaluate.sh
```
- Change the variable `model_name` expressing the model you want to evaluate
- Change the variable `metrics` expressing the type of evaluation that you want to run:
    - `quantitative`: compute the Average Displacement Error (ADE) and Final Displacement Error (FDE)
    - `qualitative`: visualize a scene
    - `collisions`: compute the number of collisions


Save all the quantitative results in a CSV file.
```
sh scripts/extract.sh
```
