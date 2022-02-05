## Spurious Shifts

Robustness of motion forecasting models under spurious shifts

### Requirements

```
pip install --upgrade pip
pip install -r requirements.txt
pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html 		# optional
```


### Training

Train STGAT with three different methods 
* Empirical Risk Minimization (ERM)
* Counterfactual Analysis
* Invariant Risk Minimization (IRM)

```
bash scripts/train.sh
```

Pretrained models from all the methods on the `eth` dataset are already saved in `/models/`, with seed=1,10,20,50,100.


### Evaluation
Evaluate the trained models under various spurious shifts
```
bash scripts/evaluate.sh
```

- Change the variable `model_name` expressing the model you want to evaluate
- Change the variable `metrics` expressing the type of evaluation that you want to run:
    - `quantitative`: compute the Average Displacement Error (ADE) and Final Displacement Error (FDE)
    - `qualitative`: visualize a scene
    - `collisions`: compute the number of collisions

Save all the quantitative results in a CSV file.
```
bash scripts/extract.sh
```

### Basic Results

Results of different methods under spurious shifts.

<img src="images/ade.png" height="240"/> <img src="images/fde.png" height="240"/> 

