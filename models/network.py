import torch
from torch import nn


class Network(nn.Module):
    def __init__(
        self,
        layer_model: type[nn.Module],
        layer_size_list: list[int],
        skip_connection: bool,
        ac: nn.Module | None = None,
        bias_on=True,
        sparse: bool = False,
    ):
        super().__init__()
        self.layer_size_list = layer_size_list
        self.in_dim = self.layer_size_list[0]
        self.out_dim = self.layer_size_list[-1]
        self.depth = len(self.layer_size_list) - 1
        self.bias_on = bias_on
        self.skip_connection = skip_connection
        self.sparse = sparse
        self.ac = nn.SiLU() if ac is None else ac

        self.layer_list = [
            layer_model(
                self.layer_size_list[i],
                self.layer_size_list[i + 1],
                bias_on=self.bias_on,
                skip_connection=self.skip_connection if i < self.depth - 1 else False,
                sparse=self.sparse,
            )
            for i in range(self.depth)
        ]
        self.layers = nn.ModuleList(self.layer_list)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < self.depth - 1:
                x = self.ac(x)
        return x

    @property
    def sparsity(self) -> torch.Tensor:

        sparsities = [layer.sparsity for layer in self.layer_list] 
        return torch.stack(sparsities, dim=0).sum(dim=0)

    def set_sparse_mode(self, sparse:bool):
        self.sparse = sparse
        for layer in self.layer_list:
            layer.set_sparse_mode(sparse)

