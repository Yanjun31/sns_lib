from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from optim.zo_optimizer import ZerothOrderOptimizer

if TYPE_CHECKING:
    from optim import SyntheticFunction  # noqa: TC004


class ZoAR(ZerothOrderOptimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = "radazo",
        num_histories: int = 100,
    ):
        super().__init__(params, lr, betas, epsilon, num_queries, mu, update_rule)

        self.num_histories = num_histories
        self.past = []

    def estimate_gradient(
        self,
        flat_param: torch.Tensor,
        closure: SyntheticFunction,
    ) -> tuple[float, torch.Tensor]:

        loss = closure()

        flat_noises = self._generate_flat_noises(num_queries=self.num_queries)
        for q in range(self.num_queries):
            self._perturb_params(flat_param, flat_noises[q], self.mu)
            reward = closure()
            self.past.append([flat_noises[q], reward])

        if len(self.past) > (self.num_histories + 1) * self.num_queries:
            self.past = self.past[-(self.num_histories + 1) * self.num_queries :]

        historic_flat_noises = [p[0] for p in self.past]
        historic_flat_noises = torch.stack(
            historic_flat_noises
        )  # shape: (num_queries * (num_histories + 1), param_shape)
        historic_rewards = [
            torch.tensor(p[1], device=self.param_groups[0]["params"][0].device)
            for p in self.past
        ]
        historic_rewards = torch.stack(historic_rewards)
        historic_rewards = historic_rewards - historic_rewards.mean()

        ws = historic_rewards / self.mu  # shape: (num_queries * (num_histories + 1),)

        grad = torch.sum(ws.unsqueeze(-1) * historic_flat_noises, dim=0) / (
            len(historic_rewards) - 1
        )

        return loss, grad
