import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from parser import get_training_parser
import warnings


def main(args):
    result = pd.read_csv(f'results/{args.dataset_name}/summary.csv', sep=', ', engine='python')
    result = result[(result.split=='test') & (result.ev_shift!=0)]
    result['Method']=result['method']+result['risk']+result['lambda'].apply(str)
    result['Method']=result['Method'].apply(set_name_method)
    result.rename(columns={'ev_shift':'Domain Parameter (α)','ade':'ADE','fde':'FDE'}, inplace=True)
    result = result.drop(['data_tr','batch_type','tr_shift','epoch','tr_k','seed','split','data_te','risk','lambda','method'], axis=1)

    if result.shape[0]==0:
        warnings.warn("No experiments with controlled spurious feature, Figure 5 cannot be reproduced.")

    else: 
        # Plot ADE
        f, ax = plt.subplots(figsize=(5.5, 5))
        sns.despine(f)
        sns.lineplot(data=result, x="Domain Parameter (α)", y="ADE", hue='Method')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticks([2**i for i in range(7)])
        plt.savefig(f'images/{args.dataset_name}_ade.png', bbox_inches='tight', pad_inches=0)

        # Plot FDE
        f, ax = plt.subplots(figsize=(5.5, 5))
        sns.despine(f)
        sns.lineplot(data=result, x="Domain Parameter (α)", y="FDE", hue='Method')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticks([2**i for i in range(7)])
        plt.savefig(f'images/{args.dataset_name}_fde.png', bbox_inches='tight', pad_inches=0)

def set_name_method(method):
   if 'counter' in method:
       return 'Counterfactual'
   else:
       if 'erm' in method:
           return 'ERM'
       if 'irm' in method:
           lambda_ = method.replace('factualirm','')
           return f'IRM (λ={lambda_})'

if __name__ == "__main__":
    args = get_training_parser().parse_args()
    main(args)