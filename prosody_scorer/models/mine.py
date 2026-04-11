import torch
import torch.nn as nn


class MINEModule(nn.Module):
    def __init__(
        self, x_dim: int, z_dim: int, hidden_dim: int = 64, ema_decay: float = 0.99
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.t_net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("ema_log_denom", torch.tensor(0.0))

    def _flatten_if_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.reshape(-1, x.shape[-1])
        return x

    def forward(
        self, x: torch.Tensor, z: torch.Tensor, update_ema: bool = True
    ) -> torch.Tensor:
        return self.mi_lower_bound(x, z, update_ema=update_ema)

    def mi_lower_bound(
        self, x: torch.Tensor, z: torch.Tensor, update_ema: bool = True
    ) -> torch.Tensor:

        if x.shape[0] != z.shape[0]:
            raise ValueError(f"MINE input size mismatch: {x.shape} vs {z.shape}")

        # 1. Create the marginal Z by rolling the BATCH dimension of the original (potentially 3D) tensor
        if z.shape[0] > 1:
            z_marginal_unflattened = torch.roll(z, shifts=1, dims=0)
        else:
            z_marginal_unflattened = z

        # 2. NOW flatten the tensors for the linear layers
        x_flat = self._flatten_if_sequence(x)
        z_flat = self._flatten_if_sequence(z)
        z_marginal = self._flatten_if_sequence(z_marginal_unflattened)

        # 3. Calculate Joint
        joint_input = torch.cat([x_flat, z_flat], dim=-1)
        t_joint = self.t_net(joint_input)

        # 4. Calculate Marginal
        marginal_input = torch.cat([x_flat, z_marginal], dim=-1)
        t_marginal = self.t_net(marginal_input)

        # 5. Calculate Denominator
        n_samples = t_marginal.shape[0]
        log_denom = torch.logsumexp(t_marginal, dim=0) - torch.log(
            torch.tensor(float(n_samples), device=t_marginal.device)
        )
        log_denom = log_denom.squeeze()

        # 6. Final Bound
        mi_lb = t_joint.mean() - log_denom

        if update_ema and self.training:
            self.ema_log_denom.copy_(
                self.ema_decay * self.ema_log_denom
                + (1.0 - self.ema_decay) * log_denom.detach()
            )

        return mi_lb
