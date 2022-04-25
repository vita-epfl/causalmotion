import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from parser import get_training_parser
import os
import warnings

from utils import set_name_method


def main(args):
    # create folder images
    if not os.path.exists(f'images/{args.dataset_name}/'):
        os.makedirs(f'images/{args.dataset_name}/')

    result = pd.read_csv(f'results/{args.dataset_name}/summary.csv', sep=', ', engine='python')
    result = result[result.split=='test']
    result['Method']=result['method']+result['risk']+result['lambda'].apply(str)
    result['Method']=result['Method'].apply(set_name_method)
    result.rename(columns={'ev_shift':'Domain Parameter (α)','ade':'ADE','fde':'FDE'}, inplace=True)
    result = result.drop(['data_tr','batch_type','tr_shift','epoch','tr_k','seed','split','data_te','risk','lambda','method'], axis=1)

    print(f'\nRESULTS\nDataset: {args.dataset_name}\n\nBaseline: ')
    # Baseline
    baseline = result[result['Domain Parameter (α)']==0]
    if baseline.shape[0]==0:
        warnings.warn("No 'baseline' experiments available.")
    else: 
        baseline_summary = pd.pivot_table(baseline, 
                    values=['ADE', 'FDE'], 
                    index=['Method'],
                    aggfunc={'ADE': [np.mean,np.std],
                            'FDE': [np.mean,np.std]},
                    sort=True
                    ).round(decimals =4)
        print(baseline_summary)

    # Modified
    print(f'\n\nModified (add_confidence): ')
    addconfidence = result[result['Domain Parameter (α)']!=0]
    if addconfidence.shape[0]==0:
        warnings.warn("No 'add_confidence' experiments available, Figure 5 cannot be reproduced.")

    else: 
        # Plot ADE
        f, ax = plt.subplots(figsize=(5.5, 5))
        sns.despine(f)
        sns.lineplot(data=addconfidence, x="Domain Parameter (α)", y="ADE", hue='Method')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticks([2**i for i in range(7)])
        plt.savefig(f'images/{args.dataset_name}/ade.png', bbox_inches='tight', pad_inches=0)

        # Plot FDE
        f, ax = plt.subplots(figsize=(5.5, 5))
        sns.despine(f)
        sns.lineplot(data=addconfidence, x="Domain Parameter (α)", y="FDE", hue='Method')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticks([2**i for i in range(7)])
        plt.savefig(f'images/{args.dataset_name}/fde.png', bbox_inches='tight', pad_inches=0)
        print('see plots `images/{args.dataset_name}/ade.png` and `images/{args.dataset_name}/fde.png`')

if __name__ == "__main__":
    args = get_training_parser().parse_args()
    main(args)