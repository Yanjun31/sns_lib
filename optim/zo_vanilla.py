from typing import TYPE_CHECKING

import torch

from optim.zo_optimizer import ZerothOrderOptimizer

if TYPE_CHECKING:
    from optim import SyntheticFunction  # noqa: TC004


class ZoVanilla(ZerothOrderOptimizer):
    def estimate_gradient(
        self,
        flat_param: torch.Tensor,
        closure: SyntheticFunction,
    ) -> tuple[float, torch.Tensor]:
        loss = closure()

        flat_noises = self._generate_flat_noises(num_queries=self.num_queries)
        fs = []
        for q in range(self.num_queries):
            self._perturb_params(flat_param, flat_noises[q], self.mu)
            f_x_plus_h = closure()
            fs.append(f_x_plus_h)

        fs = torch.tensor(fs, device=self.param_groups[0]["params"][0].device)
        # fs_baseline = torch.mean(fs) # a large mu should use this one, while a smaller one should use the following one
        fs_baseline = loss
        ws = (fs - fs_baseline) / self.mu  # shape: (num_queries,)

        grad = torch.sum(ws.unsqueeze(-1) * flat_noises, dim=0)

        grad.div_(self.num_queries - 1)

        return loss, grad
