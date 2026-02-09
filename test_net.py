from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from Datasets.dataset import HiMAPDataset
from HiMAP.himap import Net
from transforms import TargetBuilder
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='/mnt/d/av2_data/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/HiMAP_state_dict.ckpt')
    parser.add_argument('--map_w_emb', type=bool, default=False)
    parser.add_argument('--submission_file_name', type=str, default='submission')
    parser.add_argument('--submission_dir', type=str, default='./result/')
    args = parser.parse_args()
    model = Net(**vars(args))
    model.load_state_dict(torch.load(args.ckpt_path))
    val_dataset = HiMAPDataset(root=args.root, split='test',
                     transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')

    trainer.test(model, dataloader)
