from collections.abc import Iterator
from typing import Protocol

import torch


class SyntheticFunction(Protocol):
    def __call__(
        self,
    ) -> float: ...


class ZerothOrderOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = "radazo",
    ):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not epsilon >= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if not num_queries > 0:
            raise ValueError(f"Invalid number of queries: {num_queries}")
        if not mu >= 0.0:
            raise ValueError(f"Invalid mu value: {mu}")

        defaults = {"lr": lr, "betas": betas, "epsilon": epsilon}
        super().__init__(params, defaults)
        self.num_queries = num_queries
        self.mu = mu
        self.update_rule = update_rule
        self.total_dim = sum(
            p.numel() for group in self.param_groups for p in group["params"]
        )
        self.param_shapes = [
            [p.shape for p in group["params"]] for group in self.param_groups
        ]

    def _generate_flat_noises(
        self,
        num_queries: int = 1,
        orthogonal: bool = False,
        previous_noise=None,
    ) -> torch.Tensor:
        if orthogonal:
            raise NotImplementedError("Orthogonal noise generation is not implemented.")
        else:
            noise = torch.randn(
                num_queries,
                self.total_dim,
                device=self.param_groups[0]["params"][0].device,
            )

        return noise

    def _perturb_params(
        self,
        flat_param: torch.Tensor,
        flat_noise: torch.Tensor,
        mu: float,
    ) -> None:
        perturb_flat_param = flat_param + mu * flat_noise
        self._set_flat_params(perturb_flat_param)

    def estimate_gradient(
        self,
        flat_param: torch.Tensor,
        closure: SyntheticFunction,
    ) -> tuple[float, torch.Tensor]:

        raise NotImplementedError("This method should be implemented in subclasses.")

    def _get_flat_param(
        self,
    ) -> torch.Tensor:
        return torch.cat(
            [p.data.view(-1) for group in self.param_groups for p in group["params"]]
        )

    def _set_flat_params(
        self,
        flat_tensor: torch.Tensor,
    ) -> None:
        pointer = 0
        for group_idx, group in enumerate(self.param_groups):
            for p_idx, p in enumerate(group["params"]):
                shape = self.param_shapes[group_idx][p_idx]
                numel = p.numel()
                p.data.copy_(flat_tensor[pointer : pointer + numel].view(shape))
                pointer += numel

    @torch.no_grad()
    def step(
        self,
        closure: SyntheticFunction,
    ) -> float:  # ty: ignore[invalid-method-override]
        assert closure is not None, (
            "Closure function is required for zeroth order optimization"
        )
        flat_param = self._get_flat_param()
        loss, flat_grad = self.estimate_gradient(flat_param, closure)

        pointer = 0
        for group_idx, group in enumerate(self.param_groups):
            for p_idx, p in enumerate(group["params"]):
                shape = self.param_shapes[group_idx][p_idx]
                numel = p.numel()
                param = flat_param[pointer : pointer + numel].view(shape)
                grad = flat_grad[pointer : pointer + numel].view(shape)
                pointer += numel

                if self.update_rule == "sgd":
                    lr = group["lr"]
                    p.data.copy_(param - lr * grad)
                else:  # adam radazo
                    lr = group["lr"]
                    beta1, beta2 = group["betas"]
                    epsilon = group["epsilon"]

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                    m, v = state["m"], state["v"]

                    state["step"] += 1
                    m.mul_(beta1).add_((1 - beta1) * grad)
                    if self.update_rule == "radazo":
                        v.mul_(beta2).add_((1 - beta2) * (m**2))
                    else:
                        v.mul_(beta2).add_((1 - beta2) * (grad**2))
                    m_hat = m
                    v_hat = v

                    p.data.copy_(param - lr * m_hat / (v_hat.sqrt() + epsilon))

        return loss
