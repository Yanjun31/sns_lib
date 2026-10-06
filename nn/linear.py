import math

import torch
from torch import nn

from utils.k_means import k_means


class Linear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias_on: bool = True,
        skip_connection: bool = False,
        sparse: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.skip_connection = skip_connection
        self.bias_on = bias_on
        self.sparse = sparse
        self.w = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.empty(self.out_dim))

        self.w_gate = nn.Parameter(3 * torch.ones_like(self.w))
        self.b_gate = nn.Parameter(3 * torch.ones_like(self.b))

        self.w_gate.requires_grad = False
        self.b_gate.requires_grad = False

        if self.sparse:
            # self.w_gate.data, self.b_gate.data = self.gate

            self.w_gate.requires_grad = True
            self.b_gate.requires_grad = True

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w.T, a=math.sqrt(5))
        fan_in = self.in_dim
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        w = self.w
        b = self.b

        if self.sparse:
            w_gate, b_gate = self.gate
            w = self.w * w_gate
            if self.bias_on:
                b = self.b * b_gate

        if self.bias_on:
            yhat = torch.matmul(x, w) + self.b
        else:
            yhat = torch.matmul(x, w)
        return yhat

    @property
    def sparsity(
        self,
    ) -> torch.Tensor:

        if self.sparse:
            sparsity = torch.sum(torch.sigmoid(self.w_gate))
            
            if self.bias_on:
                sparsity += torch.sum(torch.sigmoid(self.b_gate))
        else:
            sparsity = torch.sum(self.weight_strength_matrix)

            if self.bias_on:
                sparsity += torch.sum(self.bias_strength_matrix)
        
        return sparsity

    @property
    def weight_strength_matrix(
        self,
    ) -> torch.Tensor:

        weight_strength = torch.abs(self.w)
        weight_strength /= (1 + weight_strength)

        return weight_strength

    @property
    def bias_strength_matrix(
        self,
    ) -> torch.Tensor:

        bias_strength = torch.abs(self.b)
        bias_strength /= (1 + bias_strength)

        return bias_strength

    def set_sparse_mode(
        self,
        sparse: bool,
    ) -> None:
    
        self.sparse = sparse

        # self.w_gate.data, self.b_gate.data = self.gate

        if self.sparse:
            self.w_gate.requires_grad = True
            self.b_gate.requires_grad = True
        else:
            self.w_gate.requires_grad = False
            self.b_gate.requires_grad = False

    # @property
    # def gate(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     strength_matrix = torch.cat([self.weight_strength_matrix.flatten(), self.bias_strength_matrix]).unsqueeze(1)
    #     gate, _ = k_means(strength_matrix, num_clusters=2)
    #     w_gate = gate[:-self.out_dim].reshape(self.w.shape)
    #     b_gate = gate[-self.out_dim:].reshape(self.b.shape)
    #     if torch.max(self.weight_strength_matrix * w_gate) < torch.max(self.weight_strength_matrix * (1 - w_gate)):
    #         w_gate = 1 - w_gate
    #         b_gate = 1 - b_gate

    #     return w_gate, b_gate

    @property
    def gate(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight_logits = torch.stack([torch.zeros_like(self.w_gate), self.w_gate], dim=-1)
        bias_logits = torch.stack([torch.zeros_like(self.b_gate), self.b_gate], dim=-1)
        weight_gate = torch.nn.functional.gumbel_softmax(weight_logits, tau=1, hard=True)[..., -1]
        bias_gate = torch.nn.functional.gumbel_softmax(bias_logits, tau=1, hard=True)[..., -1]

        return weight_gate, bias_gate
