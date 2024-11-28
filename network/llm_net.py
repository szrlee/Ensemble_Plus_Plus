import numpy as np
import json
import torch
import torch.nn as nn
import loralib as lora

from transformers import GPT2LMHeadModel, GPT2Config, GPTNeoXForCausalLM, GPTNeoXConfig
from .ensemble_exp import Ensemble_exp_Layer
from .ensemble_exp import mlp


class LinearLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        use_bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.basedmodel = nn.Linear(in_features, out_features, use_bias)
        if prior_scale > 0:
            self.priormodel = nn.Linear(in_features, out_features, use_bias)
            for param in self.priormodel.parameters():
                param.requires_grad = False
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.basedmodel.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.xavier_normal_(param, gain=1.0)
        if self.prior_scale > 0:
            for name, param in self.priormodel.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.xavier_normal_(param, gain=1.0)

    def forward(self, x, prior_x):
        out = self.basedmodel(x)
        if self.prior_scale > 0:
            prior_out = self.priormodel(prior_x)
            out = self.posterior_scale * out + self.prior_scale * prior_out
        return out.unsqueeze(1)


class Ensemble_exp_Linear(nn.Module):
    def __init__(
        self,
        noise_dim,
        out_features,
        action_dim: int = 2,
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
            action_dim=action_dim,
            prior_std=prior_std,
            out_type="weight",
            use_bias=use_bias,
            device=device,
        )
        self.weight = Ensemble_exp_Layer(
            **Ensemble_exp_Layer_params, trainable=True, weight_init="xavier_normal", bias_init="zeros-zeros"
        )
        self.prior_weight = Ensemble_exp_Layer(
            **Ensemble_exp_Layer_params, trainable=False, weight_init="xavier_normal", bias_init="zeros-zeros"
        )

        self.hidden_dim = out_features
        self.action_dim = action_dim
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.device = device

    def forward(self, z, x, prior_x):
        if isinstance(z, np.ndarray):
            z = torch.as_tensor(z, device=self.device, dtype=x.dtype)
        # x: [batch_size, seq_len, hidden_dim]
        theta = self.weight(z)
        theta = theta.view(theta.shape[0], -1, self.action_dim, self.hidden_dim)
        out = torch.einsum("bsd,bnad -> bnsa", x, theta)

        if self.prior_scale > 0:
            prior_theta = self.prior_weight(z)
            prior_theta = prior_theta.view(
                prior_theta.shape[0], -1, self.action_dim, self.hidden_dim
            )
            prior_out = torch.einsum("bsd,bnad -> bnsa", prior_x, prior_theta)

            out = self.posterior_scale * out + self.prior_scale * prior_out
        return out


class EnsembleLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        ensemble_size: int = 2,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        use_bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.basedmodel = nn.ModuleList(
            [mlp(in_features, out_features, [], use_bias) for _ in range(ensemble_size)]
        )
        if prior_scale > 0:
            self.priormodel = nn.ModuleList(
                [
                    mlp(in_features, out_features, [], use_bias)
                    for _ in range(ensemble_size)
                ]
            )
            for model in self.priormodel:
                for param in model.parameters():
                    param.requires_grad = False

        self.ensemble_size = ensemble_size
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.basedmodel.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.xavier_normal_(param, gain=1.0)
        if self.prior_scale > 0:
            for name, param in self.priormodel.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.xavier_normal_(param, gain=1.0)

    def forward(self, z, logits, prior_logits):
        if z.shape[0] == 1:
            ensemble_index = int(np.where(z == 1)[1])
            out = self.basedmodel[ensemble_index](logits)
            if self.prior_scale > 0:
                prior_out = self.priormodel[ensemble_index](prior_logits)
                out = self.posterior_scale * out + self.prior_scale * prior_out
            out = out.unsqueeze(1)
        elif z.shape[0] == logits.shape[0] and len(z.shape) == 2:
            out = [
                self.basedmodel[int(np.where(z_ == 1)[0])](logits[i])
                for i, z_ in enumerate(z)
            ]
            out = torch.stack(out)
            if self.prior_scale > 0:
                prior_out = [
                    self.priormodel[int(np.where(z_ == 1)[0])](prior_logits[i])
                    for i, z_ in enumerate(z)
                ]
                prior_out = torch.stack(prior_out)
                out = self.posterior_scale * out + self.prior_scale * prior_out
            out = out.unsqueeze(1)
        else:
            out = [self.basedmodel[k](logits) for k in range(self.ensemble_size)]
            out = torch.stack(out, dim=1)
            if self.prior_scale > 0:
                prior_out = [
                    self.priormodel[k](prior_logits) for k in range(self.ensemble_size)
                ]
                prior_out = torch.stack(prior_out, dim=1)
                out = self.posterior_scale * out + self.prior_scale * prior_out
        return out


class LLM(nn.Module):
    def __init__(
        self,
        action_num: int = 2,
        noise_dim: int = 2,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        feature_sg: bool = True,
        head_name: str = "ensemble++",
        llm_name: str = "gpt2",
        use_pretrained: bool = True,
        use_lora: bool = False,
        fine_tune: bool = False,
        last_token: bool = True,
        embed_init: bool = False,
        hidden_transform: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        model_path = f"/your/path/for/{llm_name}"
        if use_pretrained:
            if "gpt2" in llm_name:
                self.transformer_model = GPT2LMHeadModel.from_pretrained(model_path)
                self.PAD_ID = 50256
            elif llm_name == "pythia-14m":
                self.transformer_model = GPTNeoXForCausalLM.from_pretrained(model_path)
                self.PAD_ID = 0
        else:
            config = json.load(open(f"{model_path}/config.json", "rb"))
            if "gpt2" in llm_name:
                self.transformer_model = GPT2LMHeadModel(GPT2Config(**config))
                self.PAD_ID = 50256
            elif llm_name == "pythia-14m":
                self.transformer_model = GPTNeoXForCausalLM(GPTNeoXConfig(**config))
                self.PAD_ID = 0

        if not fine_tune:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        if hidden_transform:
            self.gain = nn.Parameter(torch.randn(1,))
            self.bias = nn.Parameter(torch.randn(1,))

        feature_dim = self.transformer_model.config.hidden_size
        if head_name == "linear":
            self.out = LinearLayer(
                feature_dim,
                action_num,
                prior_scale=prior_scale,
                posterior_scale=posterior_scale,
                device=device,
            )
        elif head_name == "ensemble++":
            if feature_sg:
                if embed_init:
                    self.init_head_params(feature_dim, action_num)
                else:
                    self.based_out = nn.Linear(feature_dim, action_num, bias=False)
            self.out = Ensemble_exp_Linear(
                noise_dim,
                feature_dim,
                action_dim=action_num,
                prior_scale=prior_scale,
                posterior_scale=posterior_scale,
                use_bias=not feature_sg,
                device=device,
            )
        elif head_name == "ensemble+":
            self.out = EnsembleLayer(
                feature_dim,
                action_num,
                noise_dim,
                prior_scale=prior_scale,
                posterior_scale=posterior_scale,
                device=device,
            )
        else:
            raise ValueError(f"Unknown head_name: {head_name}")

        self.feature_sg = feature_sg
        self.num_padding_at_beginning = 0
        self.fine_tune = fine_tune
        self.last_token = last_token
        self.hidden_transform = hidden_transform
        self.llm_name = llm_name
        self.head_name = head_name
        self.device = device

    def init_head_params(self, feature_dim, action_num):
        output_embeddings = self.transformer_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings.mean(dim=0, keepdim=True).repeat(action_num, 1)

        self.based_out = nn.Linear(feature_dim, action_num, bias=False)
        self.based_out.weight = nn.Parameter(output_embeddings_avg)

    def transformer(self, input_ids, attention_mask):
        if "gpt2" in self.llm_name:
            transformer_out = self.transformer_model.transformer(
                input_ids=input_ids, attention_mask=attention_mask
            )
        elif self.llm_name == "pythia-14m":
            transformer_out = self.transformer_model.gpt_neox(
                input_ids=input_ids, attention_mask=attention_mask
            )
        logits = transformer_out.last_hidden_state
        if not self.fine_tune:
            logits = logits.detach()
        if self.hidden_transform:
            logits = self.gain * logits + self.bias
        return logits

    def forward(self, noise, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        logits = self.transformer(input_ids, attention_mask)
        prior_logits = logits.detach()
        if self.head_name == "linear":
            out = self.out(logits, prior_logits)
        elif self.head_name == "ensemble++":
            if self.feature_sg:
                based_out = self.based_out(logits)
                based_out = based_out.unsqueeze(1)
                logits = logits.detach()
                uncertainty_out = self.out(noise, logits, logits)
                out = based_out + uncertainty_out
            else:
                out = self.out(noise, logits, prior_logits)
        elif self.head_name == "ensemble":
            out = self.out(noise, logits, prior_logits)
        # out: [batch_size, NpS, seq_len, action_num]
        out = out.permute(0, 2, 1, 3)  # [batch_size, seq_len, NpS, action_num]
        if not self.last_token:
            return out
        else:
            bs, seq_len, NpS, action_num = out.shape
            values = torch.zeros(bs, NpS, action_num, dtype=out.dtype, device=self.device)
            for i in range(bs):
                input_id = input_ids[i]
                c_inds = (input_id == self.PAD_ID).nonzero()
                # assert self.PAD_ID == 0
                c_ind = (
                    c_inds[self.num_padding_at_beginning].item()
                    if len(c_inds) > self.num_padding_at_beginning
                    else seq_len
                )
                # Fill the values tensor with the end scores
                values[i] = out[i][c_ind - 1]
            return values.squeeze(1)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLM(device=device).to(device)
    print(model)
