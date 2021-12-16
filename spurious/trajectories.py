import logging
import math
import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class TrajectoryDataset(Dataset):
    """
    Dataloader for the ETH-UCY Trajectory Dataset
    """

    def __init__(
        self,
        data_dir,
        alpha_e,
        obs_len=8,
        fut_len=12,
        skip=1,
        min_ped=1,
        delim="\t",
        n_coordinates=2,
        add_confidence=True,
    ):
        """
        Args:
        - data_dir: Directory containing dataset file in the format
        <frame_id> <ped_id> <x> <y>
        - alpha_e: domain shift
        - obs_len: Number of time-steps in input trajectories
        - fut_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        - n_coordinates: number of (input) coordinates (i.e. x, y)
        - add_confidence: add an artificial spurious feature representing 
        the confidence of feature tracking
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.fut_len = fut_len
        self.skip = skip
        self.seq_len = self.obs_len + self.fut_len
        self.delim = delim
        self.alpha_e = alpha_e
        self.n_coordinates = n_coordinates
        self.add_confidence = add_confidence

        # initialization
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
            
        data = read_file(self.data_dir, delim) # 2d array: _ X (frame, ped, coord)
        # allocate dimension for syntethic variable
        if self.add_confidence:
            data = np.pad(data, ((0,0),(0,1)), mode='constant', constant_values=0)

        frames = np.unique(data[:, 0]).tolist() # 1d array: unique frames
        num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip)) # number of sequences
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :]) # list of 2d array: [(_ X (frame, ped, coord)) per frame]
    
        for idx in range(0, num_sequences * self.skip, skip): # for all the sequences
            curr_seq_data = np.concatenate(
                frame_data[idx : idx + self.seq_len], axis=0
            ) # 2d array: _ X (frame, ped, coord)
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) # 1d array: pedestrian (unique in seq)
            curr_seq_rel = np.zeros((len(peds_in_curr_seq), self.n_coordinates+self.add_confidence, self.seq_len)) # null 3d array: ped X coord X frame
            curr_seq = np.zeros((len(peds_in_curr_seq), self.n_coordinates+self.add_confidence, self.seq_len)) # null 3d array: ped X coord X frame
            num_peds_considered = 0

            for ped_id in peds_in_curr_seq: # for all pedestrians in seq
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :] # 2d array: _ X (frame, ped, coord)
                pad_front = frames.index(curr_ped_seq[0, 0]) - idx # idx of the first Ped'soccurence in the sequence
                pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1 # idx of the last Ped'soccurence in the sequence
                # ignore pedestrians partially framed
                if pad_end - pad_front != self.seq_len:
                    continue
                # relative coordinates
                curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) # 2d array: coord X frame
                rel_curr_ped_seq = np.zeros(curr_ped_seq.shape) # null 2d array: coord X frame
                rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1] # 2d array: coord X frame
                # add synthetic spurious variable
                if self.add_confidence:
                    rel_curr_ped_seq[self.n_coordinates,:self.obs_len] = np.linalg.norm(
                                                                                rel_curr_ped_seq[:,self.obs_len:2*self.obs_len]-rel_curr_ped_seq[:,:self.obs_len],
                                                                                ord=2,
                                                                                axis=0
                                                                                ) # 2d array: (x,y, curv) X frame
                    rel_curr_ped_seq[self.n_coordinates,:self.obs_len] = self.alpha_e * (rel_curr_ped_seq[self.n_coordinates,:self.obs_len]+1) # linear
                                                                        # 2d array: (x,y, conf) X frame               
                rel_curr_ped_seq = np.around(rel_curr_ped_seq, decimals=4)
                curr_ped_seq = np.around(curr_ped_seq, decimals=4) 

                _idx = num_peds_considered
                curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq # 3d array: ped X coord X frame
                curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq # 3d array: ped X rel_coord X frame
                num_peds_considered += 1

            if num_peds_considered > min_ped: # filter only sequences with multi-agents
                num_peds_in_seq.append(num_peds_considered) # list number of pedestrians per seq
                seq_list.append(curr_seq[:num_peds_considered]) # list of 3d array: ped X coord X frame
                seq_list_rel.append(curr_seq_rel[:num_peds_considered]) # list of 3d array: ped X rel_coord X frame

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0) # 3d array: (ped X coord X frame) seq by seq
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)  # 3d array: (ped X rel_coord X frame) seq by seq

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :self.n_coordinates, : self.obs_len]).type(
            torch.float
        ) # 3d tensor: (ped X coord X frame) seq by seq
        self.fut_traj = torch.from_numpy(seq_list[:, :self.n_coordinates, self.obs_len :]).type(
            torch.float
        ) # 3d tensor: (ped X coord X frame) seq by seq
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).type(
            torch.float
        ) # 3d tensor: (ped X rel_coord+conf X frame) seq by seq
        self.fut_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len :]).type(
            torch.float
        ) # 3d tensor: (ped X rel_coord+conf X frame) seq by seq
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist() # list of cumulative number of peds per seq
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ] # list of couples for each sequence: n ped seen, n ped seen + n ped in current seq

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index): # index of the sequence 
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], # 3d tensor: ped X coord X frame
            self.fut_traj[start:end, :], # 3d tensor: ped X coord X frame
            self.obs_traj_rel[start:end, :], # 3d tensor: ped X rel_coord X frame
            self.fut_traj_rel[start:end, :], # 3d tensor: ped X rel_coord X frame
        ]
        return out


class SynTrajectoryDataset(Dataset):
    """
    Dataloader for the Synthetic Trajectory datasets
    """

    def __init__(self, 
                data_dir='datasets/synthetic/train/orca_circle_crossing_4_ped_10000_scenes_0.2_radius_2.0_horizon.npz',
                obs_len=8,
                ):
        """
        Args:
        - data_dir: Directory containing dataset file in the format
        <seq_id> <ped_id> <x> <y>
        - obs_len: : Number of time-steps in input trajectories
        """
        super(SynTrajectoryDataset, self).__init__()
        self.obs_len = obs_len

        # load synthetic data (seq, ped, coord, frame)
        data = torch.from_numpy(np.load(data_dir)['raw']).type(torch.float).permute(0,1,3,2)

        # Split observed trajectory (features) and future trajectory (target)
        self.obs_traj = data[:,:,:,:self.obs_len]
        self.fut_traj = data[:,:,:,self.obs_len:]

        self.obs_traj_rel = torch.zeros_like(self.obs_traj)
        self.obs_traj_rel[:,:,:,1:] = self.obs_traj[:,:,:,1:]-self.obs_traj[:,:,:,:-1]
        self.fut_traj_rel = torch.zeros_like(self.fut_traj)
        self.fut_traj_rel[:,:,:,1:] = self.fut_traj[:,:,:,1:]-self.fut_traj[:,:,:,:-1]
        
    def __len__(self):
        return self.obs_traj.shape[0]

    def __getitem__(self, index): # index of the sequence 
        out = [
            self.obs_traj[index], # 3d tensor: ped X coord X frame
            self.fut_traj[index], # 3d tensor: ped X coord X frame
            self.obs_traj_rel[index], # 3d tensor: ped X rel_coord X frame
            self.fut_traj_rel[index], # 3d tensor: ped X rel_coord X frame
        ]
        return out


def seq_collate(data):
    '''
    Input: 
        Data format: batch of groups of pedestrians X coord X frame
    Output: 
        LSTM input format: frame X batch of groups of pedestrians X coord
    '''
    (
        obs_seq_list,
        fut_seq_list,
        obs_seq_rel_list,
        fut_seq_rel_list,
    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list] # lists of n_ped
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    fut_traj = torch.cat(fut_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    fut_traj_rel = torch.cat(fut_seq_rel_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj,
        fut_traj,
        obs_traj_rel,
        fut_traj_rel,
        seq_start_end,
    ]

    return tuple(out)


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)
