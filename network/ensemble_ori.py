from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def mlp(inp_dim, out_dim, hidden_sizes, bias=True):
    if len(hidden_sizes) == 0 and out_dim == 0:
        return nn.Identity()
    if len(hidden_sizes) == 0:
        return nn.Linear(inp_dim, out_dim, bias=bias)
    model = [nn.Linear(inp_dim, hidden_sizes[0], bias=bias)]
    model += [nn.ReLU(inplace=True)]
    for i in range(1, len(hidden_sizes)):
        model += [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], bias=bias)]
        model += [nn.ReLU(inplace=True)]
    if out_dim != 0:
        model += [nn.Linear(hidden_sizes[-1], out_dim, bias=bias)]
    return nn.Sequential(*model)


class EnsembleNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int] = (),
        ensemble_sizes: Sequence[int] = (),
        action_num: int = 1,
        noise_dim: int = 2,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        feature_dim = hidden_sizes[-1] if len(hidden_sizes) > 0 else in_features
        self.basedmodel = mlp(in_features, 0, hidden_sizes)
        self.out = nn.ModuleList(
            [
                mlp(feature_dim, action_num, ensemble_sizes)
                for _ in range(noise_dim)
            ]
        )
        if prior_scale > 0:
            self.priormodel = mlp(in_features, 0, hidden_sizes)
            self.prior_out = nn.ModuleList(
                [
                    mlp(feature_dim, action_num, ensemble_sizes)
                    for _ in range(noise_dim)
                ]
            )
            for param in self.priormodel.parameters():
                param.requires_grad = False
            for param in self.prior_out.parameters():
                param.requires_grad = False

        self.ensemble_num = noise_dim
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.out.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.xavier_normal_(param, gain=1.0)
        if self.prior_scale > 0:
            for name, param in self.prior_out.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.xavier_normal_(param, gain=1.0)

    def forward(self, z, x):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        logits = self.basedmodel(x)
        if z.shape[0] == 1:
            ensemble_index = int(np.where(z == 1)[1])
            out = self.out[ensemble_index](logits)
            if self.prior_scale > 0:
                prior_logits = self.priormodel(x)
                prior_out = self.prior_out[ensemble_index](prior_logits)
                out = self.posterior_scale * out + self.prior_scale * prior_out
        else:
            out = [self.out[k](logits) for k in range(self.ensemble_num)]
            out = torch.stack(out, dim=1)
            if self.prior_scale > 0:
                prior_logits = self.priormodel(x)
                prior_out = [
                    self.prior_out[k](prior_logits) for k in range(self.ensemble_num)
                ]
                prior_out = torch.stack(prior_out, dim=1)
                out = self.posterior_scale * out + self.prior_scale * prior_out
        return out.squeeze(-1)
