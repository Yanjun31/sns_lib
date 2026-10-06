import math

import torch
from torch import nn


class SNS(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias_on: bool = True,
        skip_connection: bool = False,
        sparse: bool = False,
        epsilon=1e-1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias_on = bias_on
        self.skip_connection = skip_connection
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        self.v = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.empty(self.out_dim))
        self.sparse = sparse

        self.w_gate = nn.Parameter(3 * torch.ones_like(self.w))
        self.v_gate = nn.Parameter(3 * torch.ones_like(self.v))
        self.b_gate = nn.Parameter(3 * torch.ones_like(self.b))

        self.w_gate.requires_grad = False
        self.v_gate.requires_grad = False
        self.b_gate.requires_grad = False

        if self.sparse:
            # self.w_gate.data, self.b_gate.data = self.gate

            self.w_gate.requires_grad = True
            self.v_gate.requires_grad = True
            self.b_gate.requires_grad = True



        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w.T, a=math.sqrt(5))
        fan_in = self.in_dim
        nn.init.uniform_(self.v, 0, torch.max(torch.abs(self.w)).item())
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w
        v = self.v
        b = self.b

        if self.sparse:
            w_gate, v_gate, b_gate = self.gate
            w = w * w_gate
            v = v * v_gate
            if self.bias_on:
                b = b * b_gate
        
        num = torch.matmul(x, w)
        if self.bias_on:
            num += self.b
        if self.skip_connection:
            num += x
        den = torch.matmul(x, v) + 1
        is_small_den = torch.abs(den) < self.epsilon
        den = ~is_small_den * den + is_small_den * self.epsilon * self.sign(den)
        return torch.div(num, den)

    def sign(self, x: torch.Tensor) -> torch.Tensor:
        return (x >= 0).float() - (x < 0).float()

    @property
    def w_weight_strength_matrix(
        self,
    ) -> torch.Tensor:

        weight_strength = torch.abs(self.w)
        weight_strength /= (1 + weight_strength)

        return weight_strength

    @property
    def v_weight_strength_matrix(
        self,
    ) -> torch.Tensor:

        weight_strength = torch.abs(self.v)
        weight_strength /= (1 + weight_strength)

        return weight_strength

    @property
    def bias_strength_matrix(
        self,
    ) -> torch.Tensor:

        bias_strength = torch.abs(self.b)
        bias_strength /= (1 + bias_strength)

        return bias_strength
    
    @property
    def sparsity(
        self,
    ) -> torch.Tensor:

        if self.sparse:
            sparsity = torch.sum(torch.sigmoid(self.w_gate)) + torch.sum(torch.sigmoid(self.v_gate))
            
            if self.bias_on:
                sparsity += torch.sum(torch.sigmoid(self.b_gate))
        else:
            sparsity = torch.sum(self.w_weight_strength_matrix) + torch.sum(self.v_weight_strength_matrix)

            if self.bias_on:
                sparsity += torch.sum(self.bias_strength_matrix)
        
        return sparsity


    def set_sparse_mode(
            self,
            sparse:bool,
        ) -> None:
        
            self.sparse = sparse
    
            # self.w_gate.data, self.b_gate.data = self.gate
    
            if self.sparse:
                self.w_gate.requires_grad = True
                self.v_gate.requires_grad = True
                self.b_gate.requires_grad = True
            else:
                self.w_gate.requires_grad = False
                self.v_gate.requires_grad = False
                self.b_gate.requires_grad = False

    @property
    def gate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w_weight_logits = torch.stack([torch.zeros_like(self.w_gate), self.w_gate], dim=-1)
        v_weight_logits = torch.stack([torch.zeros_like(self.v_gate), self.v_gate], dim=-1)
        bias_logits = torch.stack([torch.zeros_like(self.b_gate), self.b_gate], dim=-1)
        w_weight_gate = torch.nn.functional.gumbel_softmax(w_weight_logits, tau=1, hard=True)[..., -1]
        v_weight_gate = torch.nn.functional.gumbel_softmax(v_weight_logits, tau=1, hard=True)[..., -1]
        bias_gate = torch.nn.functional.gumbel_softmax(bias_logits, tau=1, hard=True)[..., -1]

        return w_weight_gate, v_weight_gate, bias_gate
