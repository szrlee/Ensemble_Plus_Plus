from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(input_dim, hidden_sizes, linear_layer=nn.Linear):
    model = []
    if len(hidden_sizes) > 0:
        hidden_sizes = [input_dim] + list(hidden_sizes)
        for i in range(1, len(hidden_sizes)):
            model += [linear_layer(hidden_sizes[i - 1], hidden_sizes[i])]
            model += [nn.ReLU(inplace=True)]
    model = nn.Sequential(*model)
    return model


class Ensemble_exp_Layer(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int,
        action_dim: int = 1,
        prior_std: float = 1.0,
        use_bias: bool = True,
        trainable: bool = True,
        out_type: str = "weight",
        weight_init: str = "xavier_normal",
        bias_init: str = "sphere-sphere",
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        assert out_type in ["weight", "bias"], f"No out type {out_type} in Ensemble_exp_Layer"
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.prior_std = prior_std
        self.use_bias = use_bias
        self.trainable = trainable
        self.out_type = out_type
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.device = device

        self.in_features = noise_dim
        if out_type == "weight":
            self.out_features = action_dim * hidden_dim
        elif out_type == "bias":
            self.out_features = action_dim

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        if not self.trainable:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def reset_parameters(self) -> None:
        # init weight
        if self.weight_init == "sDB":
            weight = np.random.randn(self.out_features, self.in_features).astype(
                np.float32
            )
            weight = weight / np.linalg.norm(weight, axis=1, keepdims=True)
            self.weight = nn.Parameter(
                torch.from_numpy(self.prior_std * weight).float()
            )
        elif self.weight_init == "gDB":
            weight = np.random.randn(self.out_features, self.in_features).astype(
                np.float32
            )
            self.weight = nn.Parameter(
                torch.from_numpy(self.prior_std * weight).float()
            )
        elif self.weight_init == "trunc_normal":
            bound = 1.0 / np.sqrt(self.in_features)
            nn.init.trunc_normal_(self.weight, std=bound, a=-2 * bound, b=2 * bound)
        elif self.weight_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight, gain=1.0)
        elif self.weight_init == "xavier_normal":
            nn.init.xavier_normal_(self.weight, gain=1.0)
        else:
            weight = [
                nn.init.xavier_normal_(
                    torch.zeros((self.action_dim, self.hidden_dim))
                ).flatten()
                for _ in range(self.in_features)
            ]
            self.weight = nn.Parameter(torch.stack(weight, dim=1))
            # nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # init bias
        if self.use_bias:
            if self.bias_init == "default":
                bound = 1.0 / np.sqrt(self.in_features)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                weight_bias_init, bias_bias_init = self.bias_init.split("-")
                if self.out_type == "weight":
                    if weight_bias_init == "zeros":
                        nn.init.zeros_(self.bias)
                    elif weight_bias_init == "sphere":
                        bias = np.random.randn(self.out_features).astype(np.float32)
                        bias = bias / np.linalg.norm(bias)
                        self.bias = nn.Parameter(
                            torch.from_numpy(self.prior_std * bias).float()
                        )
                    elif weight_bias_init == "xavier":
                        bias = nn.init.xavier_normal_(
                            torch.zeros((self.action_dim, self.hidden_dim))
                        )
                        self.bias = nn.Parameter(bias.flatten())
                elif self.out_type == "bias":
                    if bias_bias_init == "zeros":
                        nn.init.zeros_(self.bias)
                    elif bias_bias_init == "sphere":
                        bias = np.random.randn(self.out_features).astype(np.float32)
                        bias = bias / np.linalg.norm(bias)
                        self.bias = nn.Parameter(
                            torch.from_numpy(self.prior_std * bias).float()
                        )
                    elif bias_bias_init == "uniform":
                        bound = 1 / np.sqrt(self.hidden_dim)
                        nn.init.uniform_(self.bias, -bound, bound)
                    elif bias_bias_init == "pos":
                        bias = 1 * np.ones(self.out_features)
                        self.bias = nn.Parameter(torch.from_numpy(bias).float())
                    elif bias_bias_init == "neg":
                        bias = -1 * np.ones(self.out_features)
                        self.bias = nn.Parameter(torch.from_numpy(bias).float())

    def forward(self, z: torch.Tensor):
        z = z.to(self.device)
        return F.linear(z, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class Ensemble_exp_Linear(nn.Module):
    def __init__(
        self,
        noise_dim,
        out_features,
        action_num: int = 1,
        prior_std: float = 1.0,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        use_bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        Ensemble_exp_Layer_params = dict(
            noise_dim=noise_dim,
            hidden_dim=out_features,
            action_dim=action_num,
            prior_std=prior_std,
            out_type="weight",
            use_bias=use_bias,
            device=device,
        )
        self.weight = Ensemble_exp_Layer(
            **Ensemble_exp_Layer_params, trainable=True, weight_init="xavier_normal"
        )
        self.prior_weight = Ensemble_exp_Layer(
            **Ensemble_exp_Layer_params, trainable=False, weight_init="sDB"
        )

        self.hidden_dim = out_features
        self.action_num = action_num
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

    def forward(self, z, x, prior_x):
        theta = self.weight(z)
        prior_theta = self.prior_weight(z)

        theta = theta.view(theta.shape[0], -1, self.action_num, self.hidden_dim)
        prior_theta = prior_theta.view(prior_theta.shape[0], -1, self.action_num, self.hidden_dim)

        out = torch.einsum("bd,bnad -> bna", x, theta)
        prior_out = torch.einsum("bd,bnad -> bna", prior_x, prior_theta)

        # if len(x.shape) > 2:
        #     # compute feel-good term
        #     out = torch.einsum("bd,bad -> ba", theta, x)
        #     prior_out = torch.einsum("bd,bad -> ba", prior_theta, prior_x)
        # elif x.shape[0] != z.shape[0]:
        #     # compute action value for one action set
        #     out = torch.mm(theta, x.T).squeeze(0)
        #     prior_out = torch.mm(prior_theta, prior_x.T).squeeze(0)
        # elif x.shape == theta.shape:
        #     out = torch.sum(x * theta, -1)
        #     prior_out = torch.sum(prior_x * prior_theta, -1)
        # else:
        #     # compute predict reward in batch
        #     out = torch.bmm(theta, x.unsqueeze(-1)).squeeze(-1)
        #     prior_out = torch.bmm(prior_theta, prior_x.unsqueeze(-1)).squeeze(-1)

        out = self.posterior_scale * out + self.prior_scale * prior_out
        # if self.action_num == 1:
        #     out = out.squeeze(2)
        return out

    def get_thetas(self, z):
        theta = self.weight(z)
        prior_theta = self.prior_weight(z)
        theta = self.posterior_scale * theta + self.prior_scale * prior_theta
        return theta


class Ensemble_exp_Net(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int] = (),
        noise_dim: int = 2,
        action_num: int = 1,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        based_prior: float = False,
        feature_sg: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        self.basedmodel = mlp(in_features, hidden_sizes)
        if based_prior:
            self.priormodel = mlp(in_features, hidden_sizes)
            for param in self.priormodel.parameters():
                param.requires_grad = False

        feature_dim = in_features if len(hidden_sizes) == 0 else hidden_sizes[-1]

        if feature_sg:
            self.based_out = nn.Linear(feature_dim, action_num, bias=False)

        self.uncertainty_out = Ensemble_exp_Linear(
            noise_dim,
            feature_dim,
            action_num=action_num,
            prior_scale=prior_scale,
            posterior_scale=posterior_scale,
            use_bias=not feature_sg,
            device=device,
        )

        self.based_prior = based_prior
        self.feature_sg = feature_sg
        self.device = device

    def forward(self, z, x):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, device=self.device)
        if isinstance(z, np.ndarray):
            z = torch.as_tensor(z, device=self.device)
        logits = self.basedmodel(x)
        if self.based_prior:
            prior_logits = self.priormodel(x)
        else:
            prior_logits = logits.detach()

        if self.feature_sg:
            based_out = self.based_out(logits)
            # if len(z.shape) == 2:
            #     based_out = based_out.squeeze(-1)
            logits = logits.detach()
            uncertainty_out = self.uncertainty_out(z, logits, prior_logits)
            if uncertainty_out.shape[1] == 1:
                uncertainty_out = uncertainty_out.squeeze(1)
            else:
                based_out = based_out.unsqueeze(1)
            out = based_out + uncertainty_out
            out = out.squeeze(-1)
        else:
            uncertainty_out = self.uncertainty_out(z, logits, prior_logits)
            if uncertainty_out.shape[1] == 1:
                uncertainty_out = uncertainty_out.squeeze(1)
            out = uncertainty_out
        return out
