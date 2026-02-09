from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch_cluster import radius
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import weight_init
from utils import wrap_angle

class SenarioEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(SenarioEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2pl_radius = pl2pl_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        self.r_a2pl_emb = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.a2pl_attn_layer = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.scenario_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.scenario_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))

        self.apply(weight_init)

    def forward(self, data: HeteroData, map_enc: torch.Tensor, agent_enc: torch.Tensor) -> Dict[str, torch.Tensor]:
        num_nodes = data['agent']['num_nodes']
        mask = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        pos_a = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].contiguous()
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)
        head_a = data['agent']['heading'][:, :self.num_historical_steps].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous()
        orient_pl = data['map_polygon']['orientation'].contiguous()
        device = pos_a.device

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        mask_s = mask.transpose(0, 1).reshape(-1)
        pos_pl_a = pos_pl.repeat(self.num_historical_steps, 1)
        orient_pl_a = orient_pl.repeat(self.num_historical_steps)
        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                 for t in range(self.num_historical_steps)], dim=0)
            batch_pl = torch.cat([data['map_polygon']['batch'] + data.num_graphs * t
                                  for t in range(self.num_historical_steps)], dim=0)
        else:
            batch_s = torch.arange(self.num_historical_steps,
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(self.num_historical_steps,
                                    device=pos_pl_a.device).repeat_interleave(data['map_polygon']['num_nodes'])
        edge_index_a2pl = radius(x= pos_pl_a[:, :2],
                                 y= pos_s[:, :2],
                                 r=50.0,
                                 batch_x=batch_pl,
                                 batch_y=batch_s,
                                 max_num_neighbors=300)
        edge_index_a2pl = edge_index_a2pl[:, mask_s[edge_index_a2pl[0]]]
        rel_pos_a2pl = pos_s[edge_index_a2pl[0]] - pos_pl_a[edge_index_a2pl[1]]
        rel_orient_a2pl = wrap_angle(head_s[edge_index_a2pl[0]] - orient_pl_a[edge_index_a2pl[1]])
        r_a2pl = torch.stack(
            [torch.norm(rel_pos_a2pl[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=orient_pl_a[edge_index_a2pl[1]], nbr_vector=rel_pos_a2pl[:, :2]),
             rel_orient_a2pl], dim=-1)
        r_a2pl = self.r_a2pl_emb(continuous_inputs=r_a2pl, categorical_embs=None)
        scenario_enc = map_enc.repeat_interleave(repeats=self.num_historical_steps, dim=0).reshape(-1, self.num_historical_steps, 128)
        agent_enc = agent_enc.reshape(-1, self.num_historical_steps,
                          self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        for i in range(self.num_layers):
            scenario_enc = self.a2pl_attn_layer[i]((agent_enc, scenario_enc.transpose(0, 1).reshape(-1, self.hidden_dim)), r_a2pl,
                                   edge_index_a2pl)
        scenario_enc = scenario_enc.view(self.num_historical_steps, -1, self.hidden_dim)
        scenario_map_enc = self.scenario_emb(scenario_enc, self.scenario_emb_h0.unsqueeze(1).repeat(1, scenario_enc.size(1), 1))[1].squeeze(0)

        return {"scenario_enc":scenario_enc,
                "scenario_map_enc":scenario_map_enc}