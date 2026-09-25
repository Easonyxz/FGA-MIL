
import torch


def zeropower_via_newton_schulz(gradient: torch.Tensor, steps: int) -> torch.Tensor:
    if gradient.ndim < 2:
        raise ValueError("Muon parameters must have at least two dimensions")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = gradient.bfloat16()
    transposed = gradient.size(-2) > gradient.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        covariance = x @ x.mT
        x = a * x + (b * covariance + c * covariance @ covariance) @ x
    return x.mT if transposed else x


def muon_update(
    gradient: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
) -> torch.Tensor:
    momentum.lerp_(gradient, 1 - beta)
    update = gradient.lerp_(momentum, beta)
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newton_schulz(update, ns_steps)
    return update * max(1, gradient.size(-2) / gradient.size(-1)) ** 0.5


def adam_update(gradient, first_moment, second_moment, step, betas, eps):
    first_moment.lerp_(gradient, 1 - betas[0])
    second_moment.lerp_(gradient.square(), 1 - betas[1])
    first_unbiased = first_moment / (1 - betas[0] ** step)
    second_unbiased = second_moment / (1 - betas[1] ** step)
    return first_unbiased / (second_unbiased.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each parameter group must define use_muon")
            group.setdefault("weight_decay", 0.0)
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if group["use_muon"]:
                    if not state:
                        state["momentum"] = torch.zeros_like(parameter)
                    update = muon_update(
                        parameter.grad, state["momentum"], beta=group["momentum"]
                    )
                else:
                    if not state:
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        parameter.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.reshape(parameter.shape), alpha=-group["lr"])
        return loss
