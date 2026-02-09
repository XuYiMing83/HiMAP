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
class AgentEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(AgentEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_x_a = 4
        input_dim_r_pl2a = 3
        input_dim_r_a2a = 3

        self.type_a_emb = nn.Embedding(10, hidden_dim)
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.pl2a_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout, bipartite=True, has_pos_emb=True)
        self.apply(weight_init)

    def forward(self,
                data: HeteroData,
                map_enc) -> Dict[str, torch.Tensor]:
        mask = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        pos_a = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].contiguous()
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)
        head_a = data['agent']['heading'][:, :self.num_historical_steps].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous()
        orient_pl = data['map_polygon']['orientation'].contiguous()
        vel = data['agent']['velocity'][:, :self.num_historical_steps, :self.input_dim].contiguous()
        map_enc_a = map_enc.repeat_interleave(repeats=self.num_historical_steps, dim=0).reshape(-1,
                                                                                               self.num_historical_steps,
                                                                                               self.hidden_dim)
        categorical_embs = [
            self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=self.num_historical_steps, dim=0)]
        x_a = torch.stack(
            [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
             torch.norm(vel[:, :, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1)
        x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs)
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        mask_s = mask.transpose(0, 1).reshape(-1)
        pos_pl = pos_pl.repeat(self.num_historical_steps, 1)
        orient_pl = orient_pl.repeat(self.num_historical_steps)
        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                 for t in range(self.num_historical_steps)], dim=0)
            batch_pl = torch.cat([data['map_polygon']['batch'] + data.num_graphs * t
                                  for t in range(self.num_historical_steps)], dim=0)
        else:
            batch_s = torch.arange(self.num_historical_steps,
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(self.num_historical_steps,
                                    device=pos_pl.device).repeat_interleave(data['map_polygon']['num_nodes'])
        edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius, batch_x=batch_s, batch_y=batch_pl,
                                 max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
             rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        x_a = x_a.reshape(-1, self.num_historical_steps,
                          self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        x_a = self.pl2a_attn_layer((map_enc_a.transpose(0, 1).reshape(-1, self.hidden_dim), x_a), r_pl2a,
                                       edge_index_pl2a)
        x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)
        return x_a