from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch
from datamodules import HiMAPDataModule
from HiMAP.himap import Net
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
if __name__ == '__main__':
    pl.seed_everything(2025, workers=True)
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--processed', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    Net.add_model_specific_args(parser)
    args = parser.parse_args()

    model = Net(**vars(args))
    datamodule = HiMAPDataModule(**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=10, mode='min', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs)
    trainer.fit(model, datamodule)