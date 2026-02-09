from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minFDE
from HiMAP.encoder import Encoder
from HiMAP.decoder import Decoder
from utils import weight_init

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object

class Net(pl.LightningModule):
    def __init__(self,
                 input_dim: int=2,
                 hidden_dim: int=128,
                 num_historical_steps: int=50,
                 num_future_steps: int=60,
                 pl2pl_radius: float=150,
                 a2a_radius: float=50,
                 pl2a_radius: float = 50,
                 num_freq_bands: int=64,
                 num_map_layers: int=1,
                 num_heads: int=8,
                 num_agent_layers: int=2,
                 head_dim: int=16,
                 dropout: float=0.1,
                 num_modes: int=6,
                 output_dim: int=2,
                 num_recurrent_steps: int=3,
                 num_t2m_steps: int=30,
                 pl2m_radius: int=150,
                 a2m_radius: int=150,
                 num_dec_layers: int=2,
                 map_w_emb: bool=False,
                 lr: float=5e-4,
                 weight_decay: float=1e-4,
                 T_max: int=64,
                 submission_dir: str='./',
                 submission_file_name: str='submission',
                 pred_his_timestep: int = 30,
                 **kwargs):
        super(Net, self).__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.pl2pl_radius = pl2pl_radius
        self.a2a_radius = a2a_radius
        self.pl2a_radius = pl2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_heads = num_heads
        self.num_agent_layers = num_agent_layers
        self.head_dim = head_dim
        self.dropout = dropout
        self.num_modes = num_modes
        self.output_dim = output_dim
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_dec_layers = num_dec_layers
        self.map_w_emb = map_w_emb
        self.pred_his_timestep = pred_his_timestep

        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            map_w_emb=map_w_emb)
        self.decoder = Decoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pred_his_timestep=pred_his_timestep
        )
        self.apply(weight_init)
        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'],
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'],
                                       reduction='none')

        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.MR = MR(max_guesses=6)
        self.test_predictions = dict()

    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        return pred

    def training_step(self,
                      data,
                      batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        his_mask = data['agent']['valid_mask'][:, :self.num_historical_steps]
        his_mask[~his_mask[:, self.num_historical_steps - 1]] = False
        his_mask = his_mask[:, self.num_historical_steps - self.pred_his_timestep:]
        pred = self(data)
        hist_loc = pred['hist_loc']
        hist_gt = data['agent']['target'][:, :self.num_historical_steps, :self.output_dim]
        hist_loss = torch.abs(
            hist_gt[:, self.num_historical_steps - self.pred_his_timestep:, :self.output_dim] - hist_loc).sum(
            dim=-1) * his_mask
        hist_loss = hist_loss.sum(dim=0) / his_mask.sum(dim=0).clamp_(min=1)
        hist_loss = hist_loss.mean()
        self.log('hist_loss', hist_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        traj = torch.cat([pred['loc'][..., :self.output_dim],
                                  pred['scale'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][:, self.num_historical_steps:, :self.output_dim],
                        data['agent']['target'][:, self.num_historical_steps:, -1:]], dim=-1)
        l2_norm = (torch.norm(traj[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_best = traj[torch.arange(traj.size(0)), best_mode]
        reg_loss = self.reg_loss(traj_best,
                                         gt[..., :self.output_dim]).sum(dim=-1) * reg_mask
        reg_loss = reg_loss.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss = reg_loss.mean()
        cls_loss = self.cls_loss(pred=traj[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('train_reg_loss', reg_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        loss = reg_loss + cls_loss + hist_loss
        return loss

    def validation_step(self,
                        data,
                        batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        his_mask = data['agent']['valid_mask'][:, :self.num_historical_steps]
        his_mask[~his_mask[:, self.num_historical_steps - 1]] = False
        his_mask = his_mask[:, self.num_historical_steps - self.pred_his_timestep:]
        pred = self(data)
        hist_loc = pred['hist_loc']
        hist_gt = data['agent']['target'][:, :self.num_historical_steps, :self.output_dim]
        hist_loss = torch.abs(
            hist_gt[:, self.num_historical_steps - self.pred_his_timestep:, :self.output_dim] - hist_loc).sum(
            dim=-1) * his_mask
        hist_loss = hist_loss.sum(dim=0) / his_mask.sum(dim=0).clamp_(min=1)
        hist_loss = hist_loss.mean()
        self.log('val_hist_loss', hist_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        traj = torch.cat([pred['loc'][..., :self.output_dim],
                                  pred['scale'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][:, self.num_historical_steps:, :self.output_dim],
                        data['agent']['target'][:, self.num_historical_steps:, -1:]], dim=-1)
        l2_norm = (torch.norm(traj[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_best = traj[torch.arange(traj.size(0)), best_mode]
        reg_loss = self.reg_loss(traj_best,
                                         gt[..., :self.output_dim]).sum(dim=-1) * reg_mask
        reg_loss = reg_loss.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss = reg_loss.mean()
        cls_loss = self.cls_loss(pred=traj[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('val_reg_loss_propose', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        eval_mask = data['agent']['category'] == 3
        valid_mask_eval = reg_mask[eval_mask]
        traj_eval = traj[eval_mask, :, :, :self.output_dim]
        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]

        self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                          valid_mask=valid_mask_eval)
        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                       valid_mask=valid_mask_eval)
        self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def test_step(self,
                  data,
                  batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        traj_refine = torch.cat([pred['loc'][..., :self.output_dim],
                                 pred['scale'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        eval_mask = data['agent']['category'] == 3
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                 rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        traj_eval = traj_eval.cpu().numpy()
        pi_eval = pi_eval.cpu().numpy()
        eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
        if isinstance(data, Batch):
            for i in range(data.num_graphs):
                self.test_predictions[data['scenario_id'][i]] = (pi_eval[i], {eval_id[i]: traj_eval[i]})
        else:
            self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})

    def on_test_end(self):
        ChallengeSubmission(self.test_predictions).to_parquet(
            Path(self.submission_dir) / f'{self.submission_file_name}.parquet')

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiMAP')
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--num_historical_steps', type=int, default=50)
        parser.add_argument('--num_future_steps', type=int, default=60)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_recurrent_steps', type=int, default=3)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, default=150)
        parser.add_argument('--pl2a_radius', type=float, default=50)
        parser.add_argument('--a2a_radius', type=float, default=50)
        parser.add_argument('--num_t2m_steps', type=int, default=30)
        parser.add_argument('--pl2m_radius', type=float, default=150)
        parser.add_argument('--a2m_radius', type=float, default=150)
        parser.add_argument('--map_w_emb', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--pred_his_timestep', type=int, default=30)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')

        return parent_parser
