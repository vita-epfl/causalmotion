import argparse
import logging
from torch.utils.tensorboard import SummaryWriter

import os
import random

import numpy as np
import torch

from loader import data_loader
from parser import get_training_parser
from models import STGAT
from utils import *

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    logging.info("Initializing Training Set")
    train_path = get_dset_path(args.dataset_name, "train")
    train_files = os.listdir(train_path)
    train_envs_path = [os.path.join(train_path, file) for file in train_files] 
    train_loaders = [data_loader(args, train_env_path) for train_env_path in train_envs_path]
    
    logging.info("Initializing Validation Set")
    val_path = get_dset_path(args.dataset_name, "val")
    val_files = os.listdir(val_path)
    val_envs_path = [os.path.join(val_path, file) for file in val_files] 
    val_loaders = [data_loader(args, val_env_path) for val_env_path in val_envs_path]
    
    writer = SummaryWriter()

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
    model.cuda()
    optimizer = torch.optim.Adam(
        [
            {"params": model.traj_lstm_model.parameters(), "lr": 1e-2},
            {"params": model.traj_hidden2pos.parameters()},
            {"params": model.gatencoder.parameters(), "lr": 3e-2},
            {"params": model.graph_lstm_model.parameters(), "lr": 1e-2},
            {"params": model.traj_gat_hidden2pos.parameters()},
            {"params": model.pred_lstm_model.parameters()},
            {"params": model.pred_hidden2pos.parameters()},
        ],
        lr=args.lr,
    )
    best_ade = 1e10 # +inf
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
    
    num_epochs = [int(x) for x in args.num_epochs.strip().split("-")]
    num_batches = min([len(train_loader) for train_loader in train_loaders])
    training_step = 1
    for epoch in range(args.start_epoch, sum(num_epochs)):
        if epoch < num_epochs[0]:
            training_step = 1
        elif epoch < num_epochs[0]+num_epochs[1]:
            training_step = 2
        else:
            if epoch == num_epochs[0]+num_epochs[1]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 5e-3
            training_step = 3
        train(args, model, train_loaders, optimizer, epoch, num_batches, training_step, train_files, writer)
        if training_step == 3:
            ade = validate(args, model, val_loaders, epoch, val_files, writer)
            is_best = ade < best_ade
            best_ade = min(ade, best_ade)

            counter = 'counter' if args.counter else 'factual'
            batch_type = 'het' if args.batch_hetero else 'hom'
            name_model = f"./models/{args.dataset_name}/STGAT_{counter}_{args.risk}_{args.ic_weight}_data_{args.dataset_name}_{batch_type}_ds_{args.domain_shifts}_bk_{args.best_k}_ep_{args.num_epochs}_seed_{args.seed}.pth.tar"

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_ade": best_ade,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                name_model,
            )
    writer.close()


def train(args, model, train_loaders, optimizer, epoch, num_batches, training_step, train_files, writer):
    '''
    Train the model for a epoch
    '''
    model.train()
    logging.info('Epoch: {}'.format(epoch+1))
    logging.info("Training")

    # use heterogeneous batches
    if args.batch_hetero:
        train_loaders_iter = [iter(train_loader) for train_loader in train_loaders]
        losses = AverageMeter("Loss", ":.4f")
        progress = ProgressMeter(num_batches, [losses], prefix="")
        for _ in range(num_batches):
            loss = []
            ped_tot = torch.zeros(1).cuda() 
            for train_loader_iter in train_loaders_iter:
                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    raise RuntimeError()
                loss_i = torch.zeros(1).cuda()
                batch = [tensor.cuda() for tensor in batch]
                (
                    obs_traj,
                    fut_traj,
                    obs_traj_rel,
                    fut_traj_rel,
                    seq_start_end,
                ) = batch
                ped_tot += obs_traj.shape[1]
                optimizer.zero_grad()
                l2_loss_rel = []
                scale = torch.tensor(1.).to(obs_traj.device).requires_grad_()

                if training_step == 1 or training_step == 2:
                    pred_traj_rel = model( 
                                            obs_traj_rel, # past rel
                                            seq_start_end, 
                                            0, # no Teacher 
                                            training_step
                                        ) # pred_obs_rel
                    l2_loss_rel.append(
                        l2_loss(pred_traj_rel*scale, 
                                obs_traj_rel,  
                                mode="raw"
                            )
                    )
                else:
                    model_input = torch.cat((obs_traj_rel, fut_traj_rel), dim=0)
                    for _ in range(args.best_k):
                        pred_traj_rel = model(
                                                model_input, # past+fut rel
                                                seq_start_end, 
                                                0 # no Teacher
                                            ) # pred_fut_rel
                        l2_loss_rel.append(
                            l2_loss(
                                pred_traj_rel*scale,
                                fut_traj_rel,
                                mode="raw",
                            )
                        )

                l2_loss_sum_rel1 = torch.zeros(1).to(fut_traj)
                l2_loss_sum_rel2 = torch.zeros(1).to(fut_traj)
                l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
                even = True
                for start, end in seq_start_end.data:
                    _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                    _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [best_k elements]
                    _l2_loss_rel = torch.min(_l2_loss_rel) / (
                        (pred_traj_rel.shape[0]) * (end - start)
                    )  
                    if even==True:
                        l2_loss_sum_rel1 += _l2_loss_rel
                        even=False
                    else:
                        l2_loss_sum_rel2 += _l2_loss_rel
                        even=True

                # empirical risk (ERM)
                loss_i = l2_loss_sum_rel1 + l2_loss_sum_rel2
                
                # invariance constraint (IRM)
                if args.risk=="irm":
                    if args.unbiased:
                        g1 = torch.autograd.grad(l2_loss_sum_rel1, [scale], create_graph=True)[0]
                        g2 = torch.autograd.grad(l2_loss_sum_rel2, [scale], create_graph=True)[0]
                        inv_constr = g1*g2
                        loss_i += inv_constr * args.ic_weight
                    else:
                        grad = torch.autograd.grad(loss_i, [scale], create_graph=True)[0]
                        inv_constr = torch.sum(grad ** 2)
                        loss_i += inv_constr * args.ic_weight

                loss.append(loss_i)
            
            loss_tot = (torch.stack(loss)).sum()

            # variance risk extrapolation
            if args.risk=="vrex":
                loss_tot += (torch.stack(loss)).var() * args.ic_weight
                
            losses.update(loss_tot.item(), ped_tot.item())
            loss_tot.backward()
            optimizer.step()
        progress.display(num_batches)
        loss=losses.avg


    # use homogeneous batches
    else:
        for train_idx, train_loader in enumerate(train_loaders):
            losses = AverageMeter("Loss", ":.4f")
            progress = ProgressMeter(len(train_loader), [losses], prefix="Dataset: {:<25}".format(train_files[train_idx]))
            for batch_idx, batch in enumerate(train_loader):
                batch = [tensor.cuda() for tensor in batch]
                (
                    obs_traj,
                    fut_traj,
                    obs_traj_rel,
                    fut_traj_rel,
                    seq_start_end,
                ) = batch
                optimizer.zero_grad()
                loss = torch.zeros(1).to(fut_traj)
                l2_loss_rel = []
                scale = torch.tensor(1.).to(obs_traj.device).requires_grad_()

                if training_step == 1 or training_step == 2:
                    pred_traj_rel = model( 
                                            obs_traj_rel, # past rel
                                            seq_start_end, 
                                            0, # no teacher 
                                            training_step
                                        ) # pred_obs_rel
                    l2_loss_rel.append(
                        l2_loss(pred_traj_rel*scale, 
                                obs_traj_rel,  
                                mode="raw"
                            )
                    )
                else:
                    model_input = torch.cat((obs_traj_rel, fut_traj_rel), dim=0)
                    for _ in range(args.best_k):
                        pred_traj_rel = model(
                                                model_input, # past+fut rel
                                                seq_start_end, 
                                                0 # no teacher
                                            ) # pred_fut_rel
                        l2_loss_rel.append(
                            l2_loss(
                                pred_traj_rel*scale,
                                fut_traj_rel,
                                mode="raw",
                            )
                        )

                l2_loss_sum_rel1 = torch.zeros(1).to(fut_traj)
                l2_loss_sum_rel2 = torch.zeros(1).to(fut_traj)
                l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
                even = True
                for start, end in seq_start_end.data:
                    _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                    _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [best_k elements]
                    _l2_loss_rel = torch.min(_l2_loss_rel) / (
                        (pred_traj_rel.shape[0]) * (end - start)
                    )  
                    if even==True:
                        l2_loss_sum_rel1 += _l2_loss_rel
                        even=False
                    else:
                        l2_loss_sum_rel2 += _l2_loss_rel
                        even=True

                # emprical risk (ERM)
                loss += l2_loss_sum_rel1 + l2_loss_sum_rel2 
                
                # invariance constraint (IRM)
                if args.risk=="irm":
                    if args.unbiased:
                        g1 = torch.autograd.grad(l2_loss_sum_rel1, [scale], create_graph=True)[0]
                        g2 = torch.autograd.grad(l2_loss_sum_rel2, [scale], create_graph=True)[0]
                        inv_constr = g1*g2
                        loss += inv_constr * args.ic_weight
                    else:
                        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
                        inv_constr = torch.sum(grad ** 2)
                        loss += inv_constr * args.ic_weight           
                
                losses.update(loss.item(), obs_traj.shape[1])
                loss.backward()
                optimizer.step()
            progress.display(batch_idx+1)
    writer.add_scalar("train_loss", losses.avg, epoch)



def validate(args, model, val_loaders, epoch, val_files, writer):
    '''
    Evaluate the performances on the validation set
    '''
    ade_tot = AverageMeter("ADE", ":.4f")
    fde_tot = AverageMeter("FDE", ":.4f")
    model.eval()
    logging.info("Validation")
    with torch.no_grad():
        for val_idx, val_loader in enumerate(val_loaders):
            ade = AverageMeter("ADE", ":.4f")
            fde = AverageMeter("FDE", ":.4f")
            progress = ProgressMeter(len(val_loader), [ade, fde], prefix="Dataset: {:<25}".format(val_files[val_idx]))
            for batch_idx, batch in enumerate(val_loader):
                batch = [tensor.cuda() for tensor in batch]
                (
                    obs_traj,
                    fut_traj,
                    obs_traj_rel,
                    _,
                    seq_start_end,
                ) = batch
                pred_fut_traj_rel = model(
                                        obs_traj_rel, 
                                        seq_start_end,
                                        training_step=3
                                    )
                pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1,:,:2])
                ade_, fde_ = cal_ade_fde(fut_traj, pred_fut_traj)
                ade_ = ade_ / (obs_traj.shape[1] * args.fut_len)
                fde_ = fde_ / (obs_traj.shape[1])
                ade.update(ade_, obs_traj.shape[1])
                fde.update(fde_, obs_traj.shape[1])
                ade_tot.update(ade_, obs_traj.shape[1])
                fde_tot.update(fde_, obs_traj.shape[1])
            progress.display(batch_idx+1)

        logging.info(
            "Average:\tADE  {ade.avg:.4f}\tFDE  {fde.avg:.4f}".format(ade=ade_tot, fde=fde_tot)
        )
        writer.add_scalar("val_ade", ade_tot.avg, epoch)
    return ade_tot.avg


def cal_ade_fde(fut_traj, pred_fut_traj):
    '''
    Compute the ADE and FDE
    '''
    ade = displacement_error(pred_fut_traj, fut_traj)
    fde = final_displacement_error(pred_fut_traj[-1], fut_traj[-1])
    return ade, fde


def save_checkpoint(state, is_best, filename):
    '''
    Save the model
    '''
    if is_best:
        torch.save(state, filename)
        logging.info("Performances improved --> Model Saved")
    logging.info("------------------------------")


if __name__ == "__main__":
    print('Using GPU: ' + str(torch.cuda.is_available()))
    args = get_training_parser().parse_args()
    if args.risk=="vrex": args.batch_hetero=True
    set_logger(os.path.join(args.log_dir,"train.log"))
    main(args)