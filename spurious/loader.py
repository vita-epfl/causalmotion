from torch.utils.data import DataLoader

from trajectories import TrajectoryDataset, SynTrajectoryDataset, seq_collate
from utils import domain_shift

def data_loader(args, path):
    # ETH-UCY Dataset
    if args.dataset_name in ('eth','hotel','univ','zara1','zara2'):
        alpha_e=domain_shift(args.domain_shifts,path)
        dset = TrajectoryDataset(
            path,
            alpha_e=alpha_e,
            obs_len=args.obs_len,
            fut_len=args.fut_len,
            skip=args.skip,
            delim=args.delim,
            n_coordinates=args.n_coordinates,
            add_confidence=args.add_confidence,
            )
    # Synthetic Dataset
    elif args.dataset_name=='synthetic':
        dset = SynTrajectoryDataset(
            path,
            obs_len=args.obs_len,
            )

    loader = DataLoader(
        dset,
        batch_size=args.batch_size//4 if args.batch_hetero else args.batch_size,
        shuffle=True if args.batch_hetero else False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=False
        )
    return loader