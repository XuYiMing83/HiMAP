from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from HiMAP.agent_encoder import AgentEncoder
from HiMAP.map_encoder import MapEncoder
from HiMAP.senario_encoder import SenarioEncoder

class Encoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_w_emb: bool) -> None:
        super(Encoder, self).__init__()
        self.map_w_emb = map_w_emb
        self.map_encoder = MapEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.agent_encoder = AgentEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.senario_encoder = SenarioEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder(data, map_enc)
        senario_enc = self.senario_encoder(data, map_enc, agent_enc)
        return {'x_a': agent_enc, 'map_enc': map_enc, **senario_enc}

