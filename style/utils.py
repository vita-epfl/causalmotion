import os
import logging
import random
from re import L
import torch
import numpy as np

NUMBER_PERSONS = 2
NUMBER_COUPLES = 2

from datetime import datetime

class AverageMeter(object):
    """
    Computes and stores the average and current value of a specific metric
    """

    def __init__(self, name, fmt=":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "Batch: " + fmt + "/" + fmt.format(num_batches)


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        fold=log_path.rsplit('/', 1)[0]
        if not os.path.exists(fold):
            os.makedirs(fold)
        open(log_path, "w+")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def relative_to_abs(rel_traj, start_pos):
    """
    Convert relative coordinates in 'natural' coordinates

    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    rel_traj = rel_traj.permute(1, 0, 2)  # --> (batch, seq_len, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    return os.path.join(_dir, "datasets", dset_name, dset_type)


def get_envs_path(dataset_name, dset_type, filter_envs):
    dset_path = get_dset_path(dataset_name, dset_type)

    # ETH-UCY Dataset
    ETH_UCY = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    if dataset_name in ETH_UCY:
        files_name = os.listdir(dset_path)
        if dset_type == 'test':
            ETH_UCY = [dataset_name]
        else:
            ETH_UCY.remove(dataset_name)
        envs_names = []
        for i, env in enumerate(ETH_UCY):
            envs_names.append([])
            if env == 'eth':
                for file_name in files_name:
                    if 'biwi_eth' in file_name:
                        envs_names[i].append(file_name)
            elif env == 'hotel':
                for file_name in files_name:
                    if 'biwi_hotel' in file_name:
                        envs_names[i].append(file_name)
            elif env == 'univ':
                for file_name in files_name:
                    if ('students' in file_name) or ('uni_examples' in file_name):
                        envs_names[i].append(file_name)
            elif env == 'zara1':
                for file_name in files_name:
                    if 'crowds_zara01' in file_name:
                        envs_names[i].append(file_name)
            elif env == 'zara2':
                for file_name in files_name:
                    if ('crowds_zara02' in file_name) or ('crowds_zara03' in file_name):
                        envs_names[i].append(file_name)
        envs_paths = [[os.path.join(dset_path, env_name) for env_name in env_names] for env_names in envs_names]
        return envs_paths, ETH_UCY

    # Synthetic Dataset
    elif dataset_name in ['synthetic', 'v2', 'v2full', 'v4'] or 'synthetic' in dataset_name:
        envs_name = os.listdir(dset_path)
        if filter_envs!='':
            filter_envs = [i for i in filter_envs.split('-')]
            envs_name_ = []
            for env_name in envs_name:
                for filter_env in filter_envs:
                    if filter_env+'_radius' in env_name:
                        envs_name_.append(env_name)
            envs_name = envs_name_
        envs_path = [os.path.join(dset_path, env_name) for env_name in envs_name]
        return envs_path, envs_name

    else:
        logging.raiseExceptions(dataset_name + ' dataset doesn\'t exists')


def int_tuple(s, delim=','):
    return tuple(int(i) for i in s.strip().split(delim))


def l2_loss(pred_fut_traj, fut_traj, mode="average"):
    """
    Compute L2 loss

    Input:
    - pred_fut_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - fut_traj: Tensor of shape (seq_len, batch, 2). Groud truth future trajectory.
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """

    loss = (fut_traj[:, :, :2].permute(1, 0, 2) - pred_fut_traj.permute(1, 0, 2)) ** 2
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.mean(loss)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)


def displacement_error(pred_fut_traj, fut_traj, consider_ped=None, mode="sum"):
    """
    Compute ADE

    Input:
    - pred_fut_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory. [12, person_num, 2]
    - fut_traj: Tensor of shape (seq_len, batch, 2). Groud truth future trajectory.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the Euclidean displacement error
    """

    loss = (fut_traj.permute(1, 0, 2) - pred_fut_traj.permute(1, 0, 2)) ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss


def final_displacement_error(pred_fut_pos, fut_pos, consider_ped=None, mode="sum"):
    """
    Compute FDE

    Input:
    - pred_fut_pos: Tensor of shape (batch, 2). Predicted last pos.
    - fut_pos: Tensor of shape (seq_len, batch, 2). Groud truth last pos.
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """

    loss = (fut_pos - pred_fut_pos) ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == "raw":
        return loss
    else:
        return torch.sum(loss)


def set_domain_shift(domain_shifts, env_name):
    """
    Set the domain shift
    """
    domain_shifts = [int(i) for i in domain_shifts.split('-')]
    if len(domain_shifts) == 5:
        if env_name == 'hotel' or 'env1' in env_name:
            alpha_e = domain_shifts[0]
        elif env_name == 'univ' or 'env2' in env_name:
            alpha_e = domain_shifts[1]
        elif env_name == 'zara1' or 'env3' in env_name:
            alpha_e = domain_shifts[2]
        elif env_name == 'zara2' or 'env4' in env_name:
            alpha_e = domain_shifts[3]
        elif env_name == 'eth' or 'env5' in env_name:
            alpha_e = domain_shifts[4]
        else:
            raise ValueError('Unkown Environment!')
    elif len(domain_shifts) == 1:
        alpha_e = domain_shifts[0]
    else:
        raise ValueError('Express a domain_shift for each of the 5 enviroment or 1 for all.')
    return alpha_e


def set_name_experiment(args):
    if args.counter:
        name_method = 'counter'
    else:
        name_method = 'factual'

    if args.irm > 0:
        name_risk = 'irm_' + str(args.irm)
    elif args.vrex > 0:
        name_risk = 'vrex_' + str(args.vrex)
    else:
        name_risk = 'erm_0.0'

    return f'STGAT_{name_method}_risk_{name_risk}_batch_{args.batch_method}_data_{args.dataset_name}_ds_{args.domain_shifts}_bk_{args.best_k}_ep_{args.num_epochs}_shuffle_{str(args.shuffle).lower()}_seed_{args.seed}'


def set_batch_size(batch_method, batch_sizes, env_name):
    '''
    Set the batch size
    '''
    # heterogenous batches
    if batch_method=='het' or batch_method=='alt':
        if batch_sizes == '':
            # ETH-UCY Dataset
            if env_name == 'hotel':
                return 7
            elif env_name == 'univ':
                return 30
            elif env_name == 'zara1':
                return 16
            elif env_name == 'zara2':
                return 38
            elif env_name == 'eth':
                return 1
            # Synthetic Dataset
            else:
                return 64
        else:
            batch_sizes = [int(i) for i in batch_sizes.split('-')]
            if len(batch_sizes) == 5:
                if env_name == 'hotel' or 'env1' in env_name:
                    return batch_sizes[0]
                elif env_name == 'univ' or 'env2' in env_name:
                    return batch_sizes[1]
                elif env_name == 'zara1' or 'env3' in env_name:
                    return batch_sizes[2]
                elif env_name == 'zara2' or 'env4' in env_name:
                    return batch_sizes[3]
                elif env_name == 'eth' or 'env5' in env_name:
                    return batch_sizes[4]
                else:
                    raise ValueError('Unkown Environment!')
            elif len(batch_sizes) == 1:
                return batch_sizes[0]
            else:
                raise ValueError('Express a domain_shift for each of the 5 enviroment or 1 for all.')

    # homogeneous batches
    elif batch_method == 'hom':
        if batch_sizes == '':
            return 64
        else:
            return int(batch_sizes)
    else:
        raise ValueError('Unkown batch method')


def interpolate_traj(traj, num_interp=4):
    """
    Add linearly interpolated points of a trajectory
    """
    sz = traj.shape
    dense = np.zeros((sz[0], (sz[1] - 1) * (num_interp + 1) + 1, 2))
    dense[:, :1, :] = traj[:, :1]

    for i in range(num_interp + 1):
        ratio = (i + 1) / (num_interp + 1)
        dense[:, i + 1::num_interp + 1, :] = traj[:, 0:-1] * (1 - ratio) + traj[:, 1:] * ratio

    return dense


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def cal_ade_fde(fut_traj, pred_fut_traj, mode='sum'):
    """
    Compute the ADE and FDE
    """
    ade = displacement_error(pred_fut_traj, fut_traj, mode=mode)
    fde = final_displacement_error(pred_fut_traj[-1], fut_traj[-1], mode=mode)
    return ade, fde

best_ade = 100

def save_checkpoint(state, ade, filename, is_best):
    """
    Save the model
    """
    global best_ade
    best_ade = min(ade, best_ade)

    if is_best:
        torch.save(state, filename)
        logging.info("Performances improved --> Model Saved")


def set_seed_globally(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_method_name(args):
    name_risk = 'erm_0.0'
    if args.irm > 0:
        name_risk += 'irm_' + str(args.irm)
    elif args.vrex > 0:
        name_risk += 'vrex_' + str(args.vrex)
    else:
        name_risk += 'erm_0.0'
    return name_risk


def get_model_name(args, name='SSE', epoch=None, t_step=None, time=False, olde=None):
    if time: name = datetime.now().strftime("%m-%d_%H:%M_") + name
    # name += f'_risk_{get_method_name(args)}
    name += f'_data_{args.dataset_name}'
    # name += f'_ds_{args.domain_shifts}_bk_{args.best_k}'
    name += f'_irm[{args.irm}]'
    # name += f'batch_size[{args.batch_size}]'
    #h ='-'.join([str(i) for i in args.filter_envs])
    #name += f'_filter_envs[{h}]'
    name += f'_filter_envs[{args.filter_envs}]'

    # to keep training if we want
    # epochs_ = '-'.join([str(i) for i in list(args.num_epochs)])
    # if olde: epochs_[-1] = olde
    l='-'.join(map(str, args.num_epochs))
    name += f'_ep_[{l}]'
    name += f'_seed_{args.seed}'
    # name += f'_class_[{args.classification}]'
    if t_step: name += f'_tstep_{t_step}'
    if epoch: name += f'_epoch_{epoch}'
    # name += f'_fut[{args.fut}]'
    # name += f'_absolut[{args.absolut}]'
    # name += f'_aggrstyle[{args.aggrstyle}]'
    name += f'_reduceall[{args.reduceall}]'

    # name += f'_wrongstyle[{args.wrongstyle}]'
    # name += f'_backclassencoder[{args.backclassencoder}]'
    # name += f'_batchsize[{args.batch_size}]'
    # name += f'_styleinteg[{args.styleinteg}]'
    # name += f'_teachingratio[{args.teachingratio}]'
    # name += f'_addloss[{args.addloss}]'
    # name += f'_lrinteg[{args.lrinteg}]'
    # name += f'_lrstgat[{args.lrstgat}]'
    
    name += f'_relsocial[{args.relsocial}]'
    # name += f'_couples[{NUMBER_COUPLES}]'
    # name += f'_contrast[{args.contrastive}]'

    # TO ADD LATER
    # name += f'_counter[{args.counter}]'
    # name += f'_consist[{args.styleconsistency}]'

    # name += f'_ttr[{args.ttr}]'
    # name += f'_shuffle[{args.shuffle}]'
    # name += f'_ttrlr[{args.ttrlr}]'

    name += f'stylefs[{args.stylefs}]'
    

    if args.finetune:
        
        #name += f'finetune[{args.filter_envs}]'
        #name += f'reduce[{args.reduce}]'
        name = f'model_t{args.reduce}'
        # name += f'_finetune{args.finetune}'

    return name


def set_name_method(name_raw):
    if 'counter' in name_raw:
        return 'Counterfactual'
    elif 'factual' in name_raw:
        name_raw = name_raw.replace('factual', '')
        if 'irm' in name_raw:
            lambda_ = float(name_raw.replace('irm', ''))
            return f'IRM (λ={lambda_})'
        elif 'vrex' in name_raw:
            lambda_ = float(name_raw.replace('irm', ''))
            return f'v-REx (λ={lambda_})'
        elif 'erm' in name_raw:
            return 'ERM'


def set_name_experiment(name):
    if 'baseline' in name:
        return 'baseline'
    if 'ablation' in name:
        lambda_ = name.replace('ablation','')
        if float(lambda_)>0:
            return f'Ablation (IRM={lambda_})'
        else:
            return 'Ablation (ERM)'


def save_all_model(args, model, optimizers, metric, epoch, training_step):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dicts': {
            'inv': model.inv_encoder.state_dict(),
            'style': model.style_encoder.state_dict(),
            'decoder': model.decoder.state_dict(),
            # 'decoder_solo': [
            #     model.decoder.pred_lstm_model.state_dict(), model.decoder.pred_hidden2pos.state_dict()
            # ],
        },
        'optimizers': {
            key: val.state_dict() for key, val in optimizers.items()
        },
        'metric': metric,
        'styleinteg': args.styleinteg,
        'complex': args.complexdecoder
    }

    if args.model_dir: filefolder = f'{args.model_dir}/{training_step}'
    else: 
        if args.finetune: phase='finetune'
        else: phase= 'pretrain'
        #filefolder = f'./models/{args.dataset_name}/{phase}/{training_step}/{args.irm}/{real_style_integ}'
        filefolder = f'./models/{args.dataset_name}/{phase}/{training_step}/{args.irm}'

        if args.finetune: filefolder += f'/{args.finetune}/{args.original_seed}'

    # Check whether the specified path exists or not
    if not os.path.exists(filefolder): os.makedirs(filefolder)  
   
    filename = f'{filefolder}/{get_model_name(args, epoch=epoch, t_step=training_step)}.pth.tar'
    torch.save(checkpoint, filename)
    logging.info(f" --> Model Saved in {filename}")


def load_all_model(args, model, optimizers):
    model_path = args.resume

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location='cuda')
        args.start_epoch = checkpoint['epoch']
        
        models_checkpoint = checkpoint['state_dicts']

        # invariant encoder
        model.inv_encoder.load_state_dict(models_checkpoint['inv'])
        if optimizers != None: 
            optimizers['inv'].load_state_dict(checkpoint['optimizers']['inv'])
            if args.start_epoch >= args.num_epochs[0] + args.num_epochs[1]: update_lr(optimizers['inv'], 5e-3)

        # decoder
        assert(not 'complexdecoder' in checkpoint or args.complexdecoder == checkpoint['complexdecoder'])
        model.decoder.load_state_dict(models_checkpoint['decoder'])
        if optimizers != None: 
            optimizers['decoder'].load_state_dict(checkpoint['optimizers']['decoder'])
            update_lr(optimizers['decoder'], args.lrstgat)
            

        # style encoder
        assert(not 'styleinteg' in checkpoint or args.styleinteg == checkpoint['styleinteg'])
        try:
            model.style_encoder.load_state_dict(models_checkpoint['style'])
            if optimizers != None: 
                optimizers['style'].load_state_dict(checkpoint['optimizers']['style'])
                update_lr(optimizers['style'], args.lrstyle)
        except Exception:
            print('Styleinteg was wrongly chosen')


        # integrator
        if args.newstyleinteg == '': # keep the curent style integrator
            if optimizers != None: 
                optimizers['integ'].load_state_dict(checkpoint['optimizers']['integ'])
                update_lr(optimizers['integ'], args.lrinteg)

        else: # change the style integrator
            model.decoder.set_integrator(args.newstyleinteg)
            if optimizers != None: 
                optimizers['integ'] = torch.optim.Adam([ {"params": model.decoder.style_blocks.parameters(), 'lr': args.lrinteg}]    
                        ) if args.newstyleinteg != 'none' else get_fake_optim()

            logging.info(f'=> loading a model with "{args.styleinteg}" but changed the integrator to "{args.newstyleinteg}"')
            args.styleinteg = args.newstyleinteg

        logging.info("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint["epoch"]))
    else:
        logging.info('model {} not found'.format(model_path))


def load_model(args, model):
    model_path = args.resume

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location='cuda')
        models_checkpoint = checkpoint['state_dicts']

        model.inv_encoder.load_state_dict(models_checkpoint['inv'])
        model.decoder.load_state_dict(models_checkpoint['decoder'])
        model.style_encoder.load_state_dict(models_checkpoint['style'])
    else:
        logging.info('model {} not found'.format(model_path))


def get_fake_optim():
    import torch.nn as nn
    l = nn.Linear(1, 1)
    return torch.optim.Adam(l.parameters())


def freeze(freez, models):
    for model in models:
        if model != None:
            for p in model.parameters():
                p.requires_grad = not freez

def update_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def from_abs_to_social(abs_coord):
    res = []
    for f in range(abs_coord.shape[0]):
        sub = []
        for k in range(abs_coord.shape[1]//NUMBER_PERSONS):
            # for i in range(NUMBER_PERSONS):
            #     for j in range(i+1, NUMBER_PERSONS):

            #         sub.append(abs_coord[f, k*NUMBER_PERSONS+i]-abs_coord[f, k*NUMBER_PERSONS+j])


            for i in range(NUMBER_PERSONS):
                for j in range(NUMBER_PERSONS):  # each possible couple i,j
                    if i==j: continue
                    
                    sub.append(abs_coord[f, k*NUMBER_PERSONS+i]-abs_coord[f, k*NUMBER_PERSONS+j])


                    #     augm_data[idx, count, x, k] = data[idx, i, x, k] - data[idx, j, x, k]
                    # count += 1

        res.append(torch.stack(sub))
    res = torch.stack(res)
    return res