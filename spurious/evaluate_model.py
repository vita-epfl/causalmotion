import argparse
import logging
import os

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from loader import data_loader
from models import STGAT
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./log/", help="Directory containing logging file")
parser.add_argument("--resume", default="STGAT_IRM_ic3.0_synthetic_bk1_ep20,20,20.pth.tar", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

# dataset
parser.add_argument("--dataset_name", default="eth", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--fut_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)
parser.add_argument("--n_coordinates", type=int, default=2, help="Number of coordinates")
parser.add_argument("--dset_type", default="test", type=str)

# randomness
parser.add_argument("--seed", type=int, default=72, help="Random seed")
parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")

# architecture (STGAT)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)
parser.add_argument("--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma")
parser.add_argument("--hidden-units",type=str,default="16",help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument("--graph_network_out_dims",type=int,default=32,help="dims of every node after through GAT module")
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)
parser.add_argument("--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)")
parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu")

# computation
parser.add_argument("--gpu_num", default="1", type=str)

# validation
parser.add_argument("--best_k", default=20, type=int)

# spurious feature
parser.add_argument("--add_confidence", default=False, type=bool)
parser.add_argument("--domain_shifts", default='0', type=str, help='domain_shifts per environment: hotel,univ,zara1,zara2,eth')

# method
parser.add_argument("--counter", default=False, type=bool, help='counterfactual analysis')

# evaluate
parser.add_argument('--metrics', type=str, default='accuracy', choices=['accuracy', 'collision', 'qualitative'], help='evaluate metrics')


def get_generator(checkpoint):
    '''
    Upload model
    '''
    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )
    n_heads = [int(x) for x in args.heads.strip().split(",")]
    model = STGAT(
        obs_len=args.obs_len,
        fut_len=args.fut_len,
        n_coordinates=args.n_coordinates,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        add_confidence=args.add_confidence,
        counter=args.counter,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model


def cal_ade_fde(fut_traj, pred_fut_traj):
    '''
    Compute the ADE and FDE
    '''
    ade = displacement_error(pred_fut_traj, fut_traj, mode="raw")
    fde = final_displacement_error(pred_fut_traj[-1], fut_traj[-1], mode="raw")
    return ade, fde


def evaluate(args, loader, generator):
    '''
    Evaluate the performances
    '''
    ade_outer, fde_outer = [], []
    total_traj = 0
    step = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                fut_traj,
                obs_traj_rel,
                _,
                seq_start_end,
            ) = batch
            
            step += seq_start_end.shape[0]
            ade, fde = [], []
            total_traj += fut_traj.size(1)

            for _ in range(args.best_k):
                pred_fut_traj_rel = generator(
                                        obs_traj_rel, 
                                        seq_start_end, 
                                        0, # No Teacher
                                        3
                                    )
                pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1,:,:2])
                ade_, fde_ = cal_ade_fde(fut_traj, pred_fut_traj)
                ade.append(ade_)
                fde.append(fde_)
            ade_sum_batch = evaluate_helper(ade, seq_start_end)
            fde_sum_batch = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum_batch)
            fde_outer.append(fde_sum_batch)
        ade_sum = sum(ade_outer) 
        fde_sum = sum(fde_outer) 
        return ade_sum, fde_sum, total_traj


def sceneplot(obsv_scene, pred_scene, gt_scene, figname='scene.png', lim=9.0):
    '''
    Plot a scene
    '''
    num_traj = pred_scene.shape[0]
    obsv_frame = obsv_scene.shape[1]
    pred_frame = pred_scene.shape[1]
    cm_subsection = np.linspace(0.0, 1.0, num_traj)
    colors = [matplotlib.cm.jet(x) for x in cm_subsection]

    for i in range(num_traj):
        for k in range(1, obsv_frame):
            plt.plot(obsv_scene[i, k-1:k+1, 0], obsv_scene[i, k-1:k+1, 1],
                '-o', color=colors[i], alpha=1.0)

        plt.plot([obsv_scene[i, -1, 0], pred_scene[i, 0, 0]], [obsv_scene[i, -1, 1], pred_scene[i, 0, 1]],
            '--', color=colors[i], alpha=1.0, linewidth=1.0)        
        for k in range(1, pred_frame):
            alpha = 1.0 - k / pred_frame
            width = (1.0 - alpha) * 24.0
            plt.plot(pred_scene[i, k-1:k+1, 0], pred_scene[i, k-1:k+1, 1],
                '--', color=colors[i], alpha=alpha, linewidth=width)

    xc = obsv_scene[:, -1, 0].mean()
    yc = obsv_scene[:, -1, 1].mean()
    plt.xlim(xc-lim, xc+lim)
    plt.ylim(yc-lim/2.0, yc+lim/2.0)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.savefig(figname, bbox_inches='tight', pad_inches=.1)
    plt.close()


def visualize(args, loader, generator):
    '''
    Viasualize some scenes
    '''
    keywords = args.resume.split('_')
    suffix = 'ds_' + args.domain_shifts + '_' + keywords[1] + '_irm_' + keywords[3] + '.png'

    # range of idx for visualization
    lb_idx = 44
    ub_idx = 44

    with torch.no_grad():
        for b, data in enumerate(loader):
            batch = [tensor.cuda() for tensor in data]
            (
                obs_traj,
                fut_traj,
                obs_traj_rel,
                _,
                seq_start_end,
            ) = batch

            for k in range(args.best_k):
                pred_fut_traj_rel = generator(
                                        obs_traj_rel,
                                        seq_start_end,
                                        0, # No Teacher
                                        3
                                    )
                pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1,:,:2])
                idx_sample = seq_start_end.shape[0]
                for i in range(idx_sample):
                    if i < lb_idx or i > ub_idx:
                        continue   # key scenes
                    idx_start, idx_end = seq_start_end[i][0], seq_start_end[i][1]
                    obsv_scene = obs_traj[:, idx_start:idx_end, :]
                    pred_scene = pred_fut_traj[:, idx_start:idx_end, :]
                    gt_scene = fut_traj[:, idx_start:idx_end, :]

                    figname = 'images/visualization/scene_{:02d}_{:02d}_sample_{:02d}_{}'.format(i, b, k, suffix)
                    sceneplot(obsv_scene.permute(1,0,2).cpu().detach().numpy(), pred_scene.permute(1,0,2).cpu().detach().numpy(), gt_scene.permute(1,0,2).cpu().detach().numpy(), figname)

def compute_col(predicted_traj, predicted_trajs_all, thres=0.2, num_interp=4):
    '''
    Compute the collisions
    '''
    dense_all = interpolate_traj(predicted_trajs_all, num_interp)
    dense_ego = interpolate_traj(predicted_traj[None, :], num_interp)
    distances = np.linalg.norm(dense_all - dense_ego, axis=-1)
    mask = distances[:, 0] > 0
    return distances[mask].min(axis=0) < thres


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    checkpoint = torch.load(os.path.join('./models/', args.dataset_name, args.resume))
    generator = get_generator(checkpoint)
    path = get_dset_path(args.dataset_name, args.dset_type)
    files = os.listdir(path)
    envs_path = [os.path.join(path, file) for file in files] 
    loaders = [data_loader(args, env_path) for env_path in envs_path]
    logging.info('Model: {}'.format(args.resume))
    logging.info('Dataset: {}'.format(args.dataset_name))
    logging.info('Eval shift: {}'.format(args.domain_shifts))
    logging.info('Dataset type: {}'.format(args.dset_type))

    # quantitative
    if args.metrics == 'quantitative':
        ade = 0
        fde = 0
        total_traj = 0
        for loader in loaders:
            ade_sum_i, fde_sum_i, total_traj_i = evaluate(args, loader, generator)
            ade += ade_sum_i
            fde += fde_sum_i
            total_traj += total_traj_i
        ade = ade / (total_traj * args.fut_len)
        fde = fde / total_traj
        logging.info('ADE: {:.4f}\tFDE: {:.4f}'.format(ade, fde))

    # qualitative
    if args.metrics == 'qualitative':
        for loader in loaders:
            visualize(args, loader, generator)

    # collisions [to be implemented]
    if args.metrics == 'collisions':
        for loader in loaders:
            visualize(args, loader, generator)

if __name__ == "__main__":
    args = parser.parse_args()
    set_logger(os.path.join(args.log_dir, args.dataset_name, args.resume[:-8]+"_"+args.dset_type+"_ds_"+str(args.domain_shifts)+".log"))
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
