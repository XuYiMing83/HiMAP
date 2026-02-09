from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle

class Decoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pred_his_timestep:int,) -> None:
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pred_his_timestep = pred_his_timestep
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3
        self.r_pl2a_t_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a_t2a_t_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.pl2a_t_attn_layers = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=True, has_pos_emb=True)
        self.scenario2a_t_attn_layers = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                             dropout=dropout, bipartite=True, has_pos_emb=True)
        self.scenariom2a_t_attn_layers = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                       dropout=dropout, bipartite=True, has_pos_emb=True)
        self.x_a_t2x_a_t_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                             dropout=dropout, bipartite=False, has_pos_emb=True)
        self.to_his_loc = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=output_dim)
        self.his_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.his_loc_emb = FourierEmbedding(input_dim=output_dim, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.his_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)

        self.pl2m_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.scenario2m_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)])
        self.a2m_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)])
        self.his2m_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)])
        self.m2m_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                             dropout=dropout, bipartite=False, has_pos_emb=False)
        self.to_loc = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                               output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_scale = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                 output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.apply(weight_init)

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos_a_t = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        head_a_t = data['agent']['heading'][:, self.num_historical_steps - 1]
        head_vector_a_t = torch.stack([head_a_t.cos(), head_a_t.sin()], dim=-1)
        x_scenario_t = scene_enc['scenario_enc']
        x_scenario = scene_enc['scenario_map_enc']
        x_pl = scene_enc['map_enc']
        x_a_t = scene_enc['x_a'][:, -1]

        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']
        edge_index_pl2a_t = radius(
            x=pos_a_t[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300)
        edge_index_pl2a_t = edge_index_pl2a_t[:, mask_dst[edge_index_pl2a_t[1], 0]]
        rel_pos_pl2a_t = pos_pl[edge_index_pl2a_t[0]] - pos_a_t[edge_index_pl2a_t[1]]
        rel_orient_pl2a_t = wrap_angle(orient_pl[edge_index_pl2a_t[0]] - head_a_t[edge_index_pl2a_t[1]])
        r_pl2a_tx = torch.stack(
            [torch.norm(rel_pos_pl2a_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_a_t[edge_index_pl2a_t[1]], nbr_vector=rel_pos_pl2a_t[:, :2]),
             rel_orient_pl2a_t], dim=-1)
        r_pl2a_t = self.r_pl2a_t_emb(continuous_inputs=r_pl2a_tx, categorical_embs=None)

        edge_index_x_a_t2x_a_t = radius_graph(
            x=pos_a_t[:, :2],
            r=self.a2m_radius,
            batch=data['agent']['batch'] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_x_a_t2x_a_t = edge_index_x_a_t2x_a_t[:, mask_src[:, -1][edge_index_x_a_t2x_a_t[0]] & mask_dst[edge_index_x_a_t2x_a_t[1], 0]]
        rel_pos_x_a_t2x_a_t = pos_a_t[edge_index_x_a_t2x_a_t[0]] - pos_a_t[edge_index_x_a_t2x_a_t[1]]
        rel_head_x_a_t2x_a_t = wrap_angle(head_a_t[edge_index_x_a_t2x_a_t[0]] - head_a_t[edge_index_x_a_t2x_a_t[1]])
        r_x_a_t2x_a_tx = torch.stack(
            [torch.norm(rel_pos_x_a_t2x_a_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_a_t[edge_index_x_a_t2x_a_t[1]], nbr_vector=rel_pos_x_a_t2x_a_t[:, :2]),
             rel_head_x_a_t2x_a_t], dim=-1)
        r_x_a_t2x_a_t = self.r_a_t2a_t_emb(continuous_inputs=r_x_a_t2x_a_tx, categorical_embs=None)

        hist_locs: List[Optional[torch.Tensor]] = [None] * self.pred_his_timestep
        x_a_t = self.pl2a_t_attn_layers((x_pl, x_a_t), r_pl2a_t, edge_index_pl2a_t)
        x_a_t = self.scenariom2a_t_attn_layers((x_scenario, x_a_t), r_pl2a_t, edge_index_pl2a_t)
        for timestep in range(self.pred_his_timestep):
            x_a_t = self.scenario2a_t_attn_layers((x_scenario_t[self.num_historical_steps-1-timestep], x_a_t), r_pl2a_t, edge_index_pl2a_t)
            x_a_t = self.x_a_t2x_a_t_attn_layer(x_a_t, r_x_a_t2x_a_t, edge_index_x_a_t2x_a_t)
            hist_locs[timestep] = self.to_his_loc(x_a_t)
        hist_loc = torch.cumsum(
            torch.stack(hist_locs, dim=-1).view(-1, self.pred_his_timestep, self.output_dim),
            dim=-2)

        hist = self.his_loc_emb(hist_loc.detach().view(-1, self.output_dim))
        hist = hist.reshape(-1, self.pred_his_timestep, self.hidden_dim).transpose(0, 1)
        hist_emb = self.his_emb(hist, self.his_emb_h0.unsqueeze(1).repeat(1, hist.size(1), 1))[1].squeeze(0)

        x_scenario_pl = x_scenario.repeat(self.num_modes, 1)
        x_pl = x_pl.repeat(self.num_modes, 1)
        m = self.mode_emb.weight.repeat(scene_enc['x_a'].size(0), 1)
        x_a = x_a_t.repeat(self.num_modes, 1)
        hist_emb_p = hist_emb.repeat(self.num_modes, 1)
        x_a_m = x_a_t.repeat_interleave(repeats=self.num_modes, dim=0)
        hist_emb_m = hist_emb.repeat_interleave(repeats=self.num_modes, dim=0)
        x_m = scene_enc['x_a'][:, -1].repeat_interleave(repeats=self.num_modes, dim=0)
        m = m + x_a_m + hist_emb_m + x_m

        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes)

        edge_index_pl2m = torch.cat([edge_index_pl2a_t + i * edge_index_pl2a_t.new_tensor(
            [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2a_tx, categorical_embs=None)
        r_pl2m = r_pl2m.repeat(self.num_modes, 1)

        edge_index_a2m = torch.cat(
            [edge_index_x_a_t2x_a_t + i * edge_index_x_a_t2x_a_t.new_tensor([data['agent']['num_nodes']]) for i in
             range(self.num_modes)], dim=1)
        r_a2m = self.r_a2m_emb(continuous_inputs=r_x_a_t2x_a_tx, categorical_embs=None)
        r_a2m = r_a2m.repeat(self.num_modes, 1)

        edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]
        locs: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        for t in range(self.num_recurrent_steps):
            for i in range(self.num_layers):
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.pl2m_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                m = self.scenario2m_attn_layers[i]((x_scenario_pl, m), r_pl2m, edge_index_pl2m)
                m = self.a2m_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                m = self.his2m_attn_layers[i]((hist_emb_p, m), r_a2m, edge_index_a2m)
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.m2m_attn_layer(m, None, edge_index_m2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim)
            locs[t] = self.to_loc(m[:, :6, :])
            scales[t] = self.to_scale(m[:, :6, :])
        loc = torch.cumsum(
            torch.cat(locs, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            dim=-2)
        scale = torch.cumsum(
            F.elu_(
                torch.cat(scales, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                alpha=1.0) +
            1.0,
            dim=-2) + 0.1
        pi = self.to_pi(m[:, :6, :]).squeeze(-1)

        return {
            'hist_loc': hist_loc,
            'loc': loc,
            'scale': scale,
            'pi': pi,
        }