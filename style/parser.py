import argparse
from utils import int_tuple


def get_evaluation_parser():
    parser = get_training_parser()
    parser.add_argument("--resume", default="STGAT_IRM_ic3.0_synthetic_bk1_ep20,20,20.pth.tar", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--dset_type", default="test", type=str)
    parser.add_argument("--noise_mix_type", default="global")
    parser.add_argument('--metrics', type=str, default='accuracy', choices=['accuracy', 'collision', 'qualitative'], help='evaluate metrics')
    return parser


def get_training_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./log/", help="Directory containing logging file")
    parser.add_argument("--model_dir", default="", help="Directory containing logging file")

    # dataset
    parser.add_argument("--dataset_name", default="synthetic", type=str)
    parser.add_argument("--delim", default="\t")
    parser.add_argument("--obs_len", default=8, type=int)
    parser.add_argument("--fut_len", default=12, type=int)
    parser.add_argument("--skip", default=1, type=int)
    parser.add_argument("--n_coordinates", type=int, default=2, help="Number of coordinates")
    parser.add_argument("--filter_envs", type=str, default="", help="Filter only certain environments (i.e 0.1-0.3-0.5)")
    parser.add_argument("--filter_envs_pretrain", type=str, default="", help="Say which env were used during pretraining (for contrastive loss) (i.e 0.1-0.3-0.5)")
    parser.add_argument('--reduce', default=0, type=int)
    parser.add_argument('--reduceall', default=0, type=int)
    parser.add_argument('--testonly', default=0, type=int, help='Only test model. 0 -> training, 1 -> testing, 3 -> testing with refinement') # 0 is normal train, 1 is test, 2 is test with k, 3 is ttr
    # randomness
    parser.add_argument("--seed", type=int, default=72, help="Random seed")
    parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
    parser.add_argument("--noise_type", default="gaussian")
    # architecture (STGAT)
    parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)
    parser.add_argument("--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma")
    parser.add_argument("--hidden-units", type=str, default="16", help="Hidden units in each hidden layer, splitted with comma")
    parser.add_argument("--graph_network_out_dims", type=int, default=32, help="dims of every node after through GAT module")
    parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu")
    parser.add_argument('--teachingratio', default=0, type=float)
    # architecture (Style)
    parser.add_argument('--stylefs', type=str, default='all', choices=['all', 'traj', 'graph'])
    parser.add_argument("--relsocial", action='store_false') # default value true
    parser.add_argument('--contrastive', default=0, type=float)
    parser.add_argument("--aggrstyle", default='minpol-mean', type=str)
    parser.add_argument("--classification", default=3, type=int)
    # full pipeline arguments
    parser.add_argument('--styleinteg', default='adain', type=str, help='Integrator type ("concat", "adain", "adainnew"')
    parser.add_argument('--newstyleinteg', default='', type=str, help='Used when loading a pretrained model but you \
            want to change the styleinteg. Set the --styleinteg param to the value of the checkpoint \
            (to avoid state_dict problems) one you want to load, and then set the new styleinteg value in this parameter ')
    # computation
    parser.add_argument("--loader_num_workers", default=2, type=int)
    parser.add_argument("--gpu_num", default="1", type=str)
    # training
    parser.add_argument("--best_k", default=1, type=int)
    parser.add_argument("--batch_size", default='', type=str)
    parser.add_argument("--batch_method", default='het', type=str,
                        help='Use Homogeneous (hom), Heterogeneous (het) or alternated homogeneous (alt) batches during training')
    parser.add_argument("--shuffle", default=True, type=bool)
    # spurious feature
    parser.add_argument("--add_confidence", default=False, type=bool)
    parser.add_argument("--domain_shifts", default='0', type=str, help='domain_shifts per environment: hotel,univ,zara1,zara2,eth')
    # method
    parser.add_argument("--counter", default=False, type=bool, help='counterfactual analysis')


    parser.add_argument("--start-epoch", default=1, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    parser.add_argument("--use_gpu", default=1, type=int)

    # general training
    parser.add_argument("--finetune", default="", type=str)
    parser.add_argument("--num_epochs", default='5-5-10', type=lambda x: int_tuple(x, '-'))  # '150-100-150',
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--batch_hetero", default=True, type=bool, help='Use Homogeneous/Heterogeneous batches during training')
    parser.add_argument("--tfdir", default='runs', type=str)
    
    # arguments for training style encoder
    parser.add_argument("--fut", default=True, type=bool, help='Use future or not to train style encoder')
    parser.add_argument("--absolut", default=True, type=bool)
    parser.add_argument('--backclassencoder', default=False, type=bool)

    # learning rates
    parser.add_argument("--lrclass", default=1e-2, type=float, help="initial learning rate for style classifier optimizer")
    parser.add_argument("--lrstgat", default=1e-3, type=float, help="initial learning rate for stgat optimizer")
    parser.add_argument("--lrstyle", default=5e-4, type=float, help="initial learning rate for style encoder optimizer")
    parser.add_argument('--lrinteg', default=0.01, type=float,  help="initial learning rate for the integrator optimizer")
   
    # other parameters to test after
    parser.add_argument('--addloss', default=0, type=float)


    parser.add_argument('--ttr', default=0, type=int,  help="Number of steps of refinement during test time")
    parser.add_argument('--ttrlr', default=0, type=float, help="initial learning rate for the refinement optimizer")
    parser.add_argument('--wrongstyle', default=False, type=bool, help="True if we refine with the accurate style, False if we want to perturb the style with a false one")

    parser.add_argument('--styleconsistency', default=0, type=float,  help="Adding a loss of style prediction to the training")
    
    # method
    parser.add_argument("--irm", default=0.0, type=float, help='IRM parameter (lambda)')
    parser.add_argument("--vrex", default=0.0, type=float, help='v-REx parameter (beta)')
    parser.add_argument("--unbiased", default=True, type=bool,
                    help='Use (mini-batch) unbiased estimatore for IRM invariant constraint')
    return parser
