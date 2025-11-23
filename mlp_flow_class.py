import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from flow_matching.utils import ModelWrapper



    
# time embedding layer, a simple linear projection would map this scalar time input to a high-dimension.
class LinearEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Linear(1, dim)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.unsqueeze(-1))    

# this is the main nn, taking in current state and time and giving out the new velocity
class DiscreteVelocityMLP(nn.Module):
    def __init__(
        self, 
        input_dim: int = 128, 
        time_dim: int = 64, 
        hidden_dim: int = 128, 
        length: int = 2,
        activation: str = "silu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.length = length
        
        # Activation Logic
        activate_map = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}
        ActLayer = activate_map.get(activation.lower(), nn.SiLU)



        self.time_embed = LinearEmbedding(time_dim)


        # Time MLP Head
        self.time_mlp = nn.Sequential(
            self.time_embed,
            nn.Linear(time_dim, time_dim),
            ActLayer(),
            nn.Linear(time_dim, time_dim),
        )
        
        #Maps discrete vocabulary indices (0-127) to vvectors
        self.token_embedding = nn.Embedding(input_dim, hidden_dim)
        flat_dim = (length * hidden_dim) + time_dim
        
        #layernorm for stabilising the gradients
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            ActLayer(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            ActLayer(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            ActLayer(),
            nn.Linear(hidden_dim, length * input_dim)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        #processes the inputs and predicts the logits
        B, L = x.shape
        t_emb = self.time_mlp(t)
        x_emb = self.token_embedding(x)
        x_flat = x_emb.view(B, -1)
        h = torch.cat([x_flat, t_emb], dim=1)
        return self.net(h).view(B, L, self.input_dim)
    
#adapting nn output for ode solver
class ProbWrapper(ModelWrapper):
        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
           
            return torch.softmax(self.model(x, t, **extras), dim=-1)