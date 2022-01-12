import argparse
from utils import int_tuple


def get_training_parser():
    parser = get_parser()

    # training
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="Path to latest checkpoint (default: none)")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="Manual epoch number (useful on restarts)")
    parser.add_argument("--num_epochs", default='150-100-150', type=str)
    parser.add_argument("--best_k", default=1, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, metavar="LR", help="Initial learning rate", dest="lr")
    parser.add_argument("--counter", default=False, type=bool, help='Counterfactual/Factual Loss')
    parser.add_argument("--risk", default='erm', type=str, choices=['erm', 'irm', 'vrex'], help='Risk minimization method')
    parser.add_argument("--unbiased", default=True, type=bool, help='Use an Unbiased Estimator for SGD')
    parser.add_argument("--ic_weight", default=0.0, type=float, help='Invariance constraint strenght')

    return parser


def get_evaluation_parser():
    parser = get_parser()

    # dataset
    parser.add_argument("--dset_type", default="test", type=str)

    # validation
    parser.add_argument("--resume", default="STGAT_IRM_ic3.0_synthetic_bk1_ep20,20,20.pth.tar", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--best_k", default=20, type=int)
    parser.add_argument("--counter", default=False, type=bool, help='counterfactual analysis')
    parser.add_argument('--metrics', type=str, default='quantitative', choices=['quantitative', 'collision', 'qualitative'], help='evaluate metrics')

    return parser


def get_parser():
    parser = argparse.ArgumentParser()

    # directories
    parser.add_argument("--log_dir", default="./log/", help="Directory containing logging file")
    
    # dataset
    parser.add_argument("--dataset_name", default="eth", type=str)
    parser.add_argument("--delim", default="\t")
    parser.add_argument("--obs_len", default=8, type=int)
    parser.add_argument("--fut_len", default=12, type=int)
    parser.add_argument("--skip", default=1, type=int)
    parser.add_argument("--n_coordinates", type=int, default=2, help="Number of coordinates")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--batch_hetero", default=False, type=bool, help='Use Homogeneous/Heterogeneous batches during training')


    # spurious feature
    parser.add_argument("--add_confidence", default=False, type=bool)
    parser.add_argument("--domain_shifts", default='0', type=str, help='Domain_shifts per environment: hotel,univ,zara1,zara2,eth')

    # randomness
    parser.add_argument("--seed", type=int, default=72, help="Random seed")
    parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
    parser.add_argument("--noise_type", default="gaussian")
    #parser.add_argument("--noise_mix_type", default="global")

    # architecture (STGAT)
    parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)
    parser.add_argument("--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma")
    parser.add_argument("--hidden-units", type=str, default="16", help="Hidden units in each hidden layer, splitted with comma")
    parser.add_argument("--graph_network_out_dims", default=32, type=int, help="dims of every node after through GAT module")
    parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu")

    # computation
    parser.add_argument("--loader_num_workers", default=4, type=int)
    parser.add_argument("--gpu_num", default="0", type=str)
    #parser.add_argument("--use_gpu", default=1, type=int)

    return parser


