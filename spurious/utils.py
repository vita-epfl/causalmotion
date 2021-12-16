import os
import logging
import torch


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
    rel_traj = rel_traj.permute(1, 0, 2) # --> (batch, seq_len, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    return os.path.join(_dir, "datasets", dset_name, dset_type)


def int_tuple(s):
    return tuple(int(i) for i in s.split(","))


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

    loss = (fut_traj[:,:,:2].permute(1, 0, 2) - pred_fut_traj.permute(1, 0, 2)) ** 2
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

    loss = (fut_traj.permute(1, 0, 2) - pred_fut_traj.permute(1, 0, 2))**2
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

    loss = (fut_pos - pred_fut_pos)** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == "raw":
        return loss
    else:
        return torch.sum(loss)

def domain_shift(domain_shifts, data_dir):
    '''
    Set the domain shift
    '''
    domain_shifts=[int(i) for i in domain_shifts.split('-')]
    if len(domain_shifts)==5:
        if 'biwi_hotel' in data_dir:
            alpha_e = domain_shifts[0]
        elif ('students' in data_dir) or ('uni_examples' in data_dir):
            alpha_e = domain_shifts[1]
        elif 'crowds_zara01' in data_dir:
            alpha_e = domain_shifts[2]
        elif ('crowds_zara02' in data_dir) or ('crowds_zara03' in data_dir):
            alpha_e = domain_shifts[3]
        elif 'biwi_eth' in data_dir:
            alpha_e = domain_shifts[4]
        else:
            raise ValueError('Unkown Environment!')
    elif len(domain_shifts)==1:
        alpha_e = domain_shifts[0]
    else:
        raise ValueError('Express a domain_shift for each of the 5 enviroment or 1 for all.')
    return alpha_e

def interpolate_traj(traj, num_interp=4):
    '''
    Add linearly interpolated points of a trajectory
    '''
    sz = traj.shape
    dense = np.zeros((sz[0], (sz[1] - 1) * (num_interp + 1) + 1, 2))
    dense[:, :1, :] = traj[:, :1]

    for i in range(num_interp+1):
        ratio = (i + 1) / (num_interp + 1)
        dense[:, i+1::num_interp+1, :] = traj[:, 0:-1] * (1 - ratio) + traj[:, 1:] * ratio

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