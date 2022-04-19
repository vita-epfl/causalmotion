import logging
import os

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from models import CausalMotionModel
from parser import get_evaluation_parser
from loader import data_loader
from utils import *

def evaluate(args, loaders, model):
    with torch.no_grad():
        model.eval()
        ade_tot_meter, fde_tot_meter = AverageMeter("ADE", ":.4f"), AverageMeter("FDE", ":.4f")
        for loader in loaders:
            #ade_meter, fde_meter = AverageMeter("ADE", ":.4f"), AverageMeter("FDE", ":.4f")
            for _, batch in enumerate(loader):
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, fut_traj, _, _, _, _, _) = batch

                step = args.resume.split('/')[3]

                pred_fut_traj_rel = model(batch, step)

                pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1, :, :2])
                
                ade_, fde_ = cal_ade_fde(fut_traj, pred_fut_traj)
                ade_, fde_ = ade_ / (obs_traj.shape[1] * fut_traj.shape[0]), fde_ / (obs_traj.shape[1])
                #ade_meter.update(ade_, obs_traj.shape[1]), fde_meter.update(fde_, obs_traj.shape[1])
                ade_tot_meter.update(ade_, obs_traj.shape[1]), fde_tot_meter.update(fde_, obs_traj.shape[1])
        logging.info('ADE: {:.4f}\tFDE: {:.4f}'.format(ade_tot_meter.avg, fde_tot_meter.avg))


def sceneplot(obsv_scene, pred_scene, gt_scene, figname='scene.png', lim=9.0):
    """
    Plot a scene
    """
    num_traj = pred_scene.shape[0]
    obsv_frame = obsv_scene.shape[1]
    pred_frame = pred_scene.shape[1]
    cm_subsection = np.linspace(0.0, 1.0, num_traj)
    colors = [matplotlib.cm.jet(x) for x in cm_subsection]

    for i in range(num_traj):
        for k in range(1, obsv_frame):
            plt.plot(obsv_scene[i, k - 1:k + 1, 0], obsv_scene[i, k - 1:k + 1, 1],
                     '-o', color=colors[i], alpha=1.0)

        plt.plot([obsv_scene[i, -1, 0], pred_scene[i, 0, 0]], [obsv_scene[i, -1, 1], pred_scene[i, 0, 1]],
                 '--', color=colors[i], alpha=1.0, linewidth=1.0)
        for k in range(1, pred_frame):
            alpha = 1.0 - k / pred_frame
            width = (1.0 - alpha) * 24.0
            plt.plot(pred_scene[i, k - 1:k + 1, 0], pred_scene[i, k - 1:k + 1, 1],
                     '--', color=colors[i], alpha=alpha, linewidth=width)

    xc = obsv_scene[:, -1, 0].mean()
    yc = obsv_scene[:, -1, 1].mean()
    plt.xlim(xc - lim, xc + lim)
    plt.ylim(yc - lim / 2.0, yc + lim / 2.0)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.savefig(figname, bbox_inches='tight', pad_inches=.1)
    plt.close()


def visualize(args, loader, generator):
    """
    Viasualize some scenes
    """
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
                    0,  # No Teacher
                    3
                )
                pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1, :, :2])
                idx_sample = seq_start_end.shape[0]
                for i in range(idx_sample):
                    if i < lb_idx or i > ub_idx:
                        continue  # key scenes
                    idx_start, idx_end = seq_start_end[i][0], seq_start_end[i][1]
                    obsv_scene = obs_traj[:, idx_start:idx_end, :]
                    pred_scene = pred_fut_traj[:, idx_start:idx_end, :]
                    gt_scene = fut_traj[:, idx_start:idx_end, :]

                    figname = 'images/visualization/scene_{:02d}_{:02d}_sample_{:02d}_{}'.format(i, b, k, suffix)
                    sceneplot(obsv_scene.permute(1, 0, 2).cpu().detach().numpy(),
                              pred_scene.permute(1, 0, 2).cpu().detach().numpy(),
                              gt_scene.permute(1, 0, 2).cpu().detach().numpy(), figname)


def compute_col(predicted_traj, predicted_trajs_all, thres=0.2, num_interp=4):
    """
    Compute the collisions
    """
    dense_all = interpolate_traj(predicted_trajs_all, num_interp)
    dense_ego = interpolate_traj(predicted_traj[None, :], num_interp)
    distances = np.linalg.norm(dense_all - dense_ego, axis=-1)
    mask = distances[:, 0] > 0
    return distances[mask].min(axis=0) < thres


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    generator = CausalMotionModel(args).cuda()
    load_all_model(args, generator, None)
    envs_path, envs_name = get_envs_path(args.dataset_name, args.dset_type, args.filter_envs)
    loaders = [data_loader(args, env_path, env_name) for env_path, env_name in zip(envs_path, envs_name)]
    logging.info('Model: {}'.format(args.resume))
    logging.info('Split: {}'.format(args.dset_type))
    logging.info('Envs: {}'.format(args.filter_envs))
    logging.info('Seed: {}'.format(args.seed))



    # quantitative
    if args.metrics == 'accuracy':
        evaluate(args, loaders, generator)
        

    # qualitative
    if args.metrics == 'qualitative':
        for loader in loaders:
            visualize(args, loader, generator)

    # collisions [to be implemented]
    if args.metrics == 'collisions':
        for loader in loaders:
            visualize(args, loader, generator)


if __name__ == "__main__":
    args = get_evaluation_parser().parse_args()
    model_param = args.resume.split('/')
    set_logger(os.path.join(args.log_dir, args.dataset_name,'finetune' if args.finetune else 'pretrain',
                            f'exp_{model_param[2]}_irm_{model_param[3]}_data_{args.dset_type}_{args.filter_envs}_ft_{model_param[4]}_red_{model_param[5][7:-8]}_seed_{args.seed}_reduce_{args.reduce}.log'))
    set_seed_globally(args.seed)
    main(args)
