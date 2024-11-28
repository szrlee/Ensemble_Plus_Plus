from functools import partial

import sys

sys.path.append("..")

import numpy as np
import torch
import torch.nn.functional as F

from logger import Logger
from utils import sample_action_noise, sample_update_noise, sample_buffer_noise
from network import LLM


def _random_argmax(vals, scale=1e-7):
    """Select max with additional random noise."""
    noise = torch.distributions.Uniform(0, 1).sample(vals.shape).to(vals.device)
    index = torch.max(vals + scale * noise, dim=-1)[1]
    return index


class ReplayBuffer:
    def __init__(self, buffer_size, buffer_shape, noise_type="sp"):
        self.buffers = {
            key: np.empty([buffer_size, *shape], dtype=np.float32)
            for key, shape in buffer_shape.items()
        }
        self.store_z = "z" in buffer_shape.keys()
        if self.store_z:
            self.noise_dim = buffer_shape["z"][-1]
        self.buffer_size = buffer_size
        self.current_size = 0
        self.point = 0
        self.set_buffer_noise(noise_type)

    def set_buffer_noise(self, noise_type):
        args = {"M": self.noise_dim}
        if noise_type == "gs":
            self.gen_noise = partial(sample_action_noise, "Gaussian", **args)
        elif noise_type == "sp":
            self.gen_noise = partial(sample_action_noise, "Sphere", **args)
        elif noise_type == "pn":
            self.gen_noise = partial(sample_action_noise, "UnifCube", **args)
        elif noise_type == "pm":
            self.gen_noise = partial(sample_action_noise, "PMCoord", **args)
        elif noise_type == "oh":
            self.gen_noise = partial(sample_action_noise, "OH", **args)
        elif noise_type == "sps":
            self.gen_noise = partial(sample_action_noise, "Sparse", **args)
        elif noise_type == "spc":
            self.gen_noise = partial(sample_action_noise, "SparseConsistent", **args)

    def __len__(self):
        return self.current_size

    def _sample(self, index):
        input_ids = self.buffers["input_ids"][index]
        attention_mask = self.buffers["attention_mask"][index]
        a_data = self.buffers["a"][index]
        r_data = self.buffers["r"][index]
        z_data = self.buffers["z"][index] if self.store_z else None
        return input_ids, attention_mask, a_data, r_data, z_data

    def reset(self):
        self.current_size = 0

    def put(self, transition):
        batch_size = len(transition["r"])
        idx = self._get_ordered_storage_idx(batch_size)
        for k, v in transition.items():
            self.buffers[k][idx] = v
        if self.store_z:
            z = self.gen_noise(dim=batch_size) / np.sqrt(self.noise_dim)
            self.buffers["z"][idx] = z

    def get(self, shuffle=True):
        # get all data in buffer
        index = list(range(self.current_size))
        if shuffle:
            np.random.shuffle(index)
        return self._sample(index)

    def sample(self, n):
        # get n data in buffer
        index = np.random.randint(low=0, high=self.current_size, size=n)
        return self._sample(index)

    def sample_all(self):
        return self._sample(range(self.current_size))

    # if full, insert in order
    def _get_ordered_storage_idx(self, inc=None):
        inc = inc or 1  # size increment
        assert inc <= self.buffer_size, "Batch committed to replay is too large!"

        if self.point + inc <= self.buffer_size - 1:
            idx = np.arange(self.point, self.point + inc)
        else:
            overflow = inc - (self.buffer_size - self.point)
            idx_a = np.arange(self.point, self.buffer_size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])

        self.point = (self.point + inc) % self.buffer_size

        # update replay size, don't add when it already surpass self.size
        if self.current_size < self.buffer_size:
            self.current_size = min(self.buffer_size, self.current_size + inc)

        if inc == 1:
            idx = idx[0]
        return idx


class LLMSolution:
    def __init__(
        self,
        n_arm: int,
        n_feature: int,
        action_num: int=2,
        threshold: float=0.5,
        noise_dim: int=4,
        NpS: int = 20,
        noise_coef: float = 0.01,
        buffer_noise: str = "sp",
        action_noise: str = "sgs",
        update_noise: str = "pn",
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        feature_sg: bool = True,
        optim: str = "Adam",
        lr: float = 0.01,
        batch_size: int = 32,
        weight_decay: float = 0.01,
        buffer_size: int = 10000,
        model_type: str = "ensemble++",
        llm_name: str = "gpt2",
        use_pretrained: bool = True,
        use_lora: bool = False,
        fine_tune: bool = False,
        last_token: bool = True,
        embed_init: bool = False,
        hidden_transform: bool = False,
        logger: Logger = None,
    ):
        self.n_arm = n_arm
        self.n_feature = n_feature
        self.action_num = action_num
        self.threshold = threshold

        self.noise_dim = noise_dim
        self.NpS = NpS
        self.noise_coef = noise_coef
        self.buffer_noise = buffer_noise
        self.action_noise = action_noise
        self.update_noise = update_noise
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.feature_sg = feature_sg

        self.optim = optim
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.llm_name = llm_name
        self.use_pretrained = use_pretrained
        self.use_lora = use_lora
        self.fine_tune = fine_tune
        self.last_token = last_token
        self.embed_init = embed_init
        self.hidden_transform = hidden_transform

        self.buffer_size = buffer_size
        self.model_type = model_type
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.init_model_optimizer()
        self.init_buffer()
        self.set_update_noise()
        self.set_action_noise()

    def init_model_optimizer(self):
        # init model
        model_param = {
            "noise_dim": self.noise_dim,
            "action_num": self.action_num,
            "prior_scale": self.prior_scale,
            "posterior_scale": self.posterior_scale,
            "feature_sg": self.feature_sg,
            "head_name": self.model_type,
            "llm_name": self.llm_name,
            "use_pretrained": self.use_pretrained,
            "use_lora": self.use_lora,
            "fine_tune": self.fine_tune,
            "last_token": self.last_token,
            "embed_init": self.embed_init,
            "hidden_transform": self.hidden_transform,
            "device": self.device,
        }
        self.model = LLM(**model_param).to(self.device)
        param_dict = {"Trainable": [], "Frozen": []}
        trainable_param_size, frozen_param_size = 0, 0
        for name, param in self.model.named_parameters():
            # if "transformer" not in name: continue
            if param.requires_grad:
                trainable_param_size += param.numel()
                param_dict["Trainable"].append(name)
            else:
                frozen_param_size += param.numel()
                param_dict["Frozen"].append(name)
        self.logger.info(param_dict)
        self.logger.info(f"\nNetwork structure:\n{str(self.model)}")
        self.logger.info(
            f"Network parameters: Trainable {trainable_param_size}, Frozen {frozen_param_size}"
        )
        # init optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.optim == "Adam":
            self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim == "SGD":
            self.optimizer = torch.optim.SGD(trainable_params, lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise NotImplementedError

    def init_buffer(self):
        # init replay buffer
        buffer_shape = {
            "input_ids": (self.n_feature,),
            "attention_mask": (self.n_feature,),
            "a": {},
            "r": (),
            "z": (self.noise_dim,),
        }
        self.buffer = ReplayBuffer(self.buffer_size, buffer_shape, self.buffer_noise)

    def update(self):
        input_ids, attention_mask, a_batch, r_batch, z_batch = self.buffer.sample(
            self.batch_size
        )
        results = self.learn(input_ids, attention_mask, a_batch, r_batch, z_batch)
        return results

    def put(self, transition):
        self.buffer.put(transition)

    def learn(self, input_ids, attention_mask, a_batch, r_batch, z_batch):
        r_batch = torch.FloatTensor(r_batch).to(self.device)
        a_batch = torch.FloatTensor(a_batch).to(dtype=torch.int64, device=self.device)
        input_ids = torch.FloatTensor(input_ids).to(
            dtype=torch.int64, device=self.device
        )
        attention_mask = torch.from_numpy(attention_mask).to(
            dtype=torch.int64, device=self.device
        )

        if self.model_type == "linear":
            predict = self.model(None, input_ids, attention_mask)
            target = r_batch
            if self.action_num > 1:
                if self.last_token:
                    predict = predict[np.arange(self.batch_size), a_batch]
                else:
                    a_one_hot = F.one_hot(a_batch, self.action_num).to(predict.dtype)
                    predict = torch.einsum("bsk,ba->bs", predict.squeeze(-2), a_one_hot)
                    target = target.unsqueeze(1).expand_as(predict)
            else:
                predict = predict.squeeze(-1)
                if not self.last_token:
                    predict = predict.squeeze(-1)
                    target = target.unsqueeze(1).expand_as(predict)
            diff = (target - predict).pow(2)
            if not self.last_token:
                diff = diff.sum(1)
            loss = diff.mean(0)
        else:
            # perturbation noise
            z_batch = torch.FloatTensor(z_batch).to(self.device)
            # noise for update
            update_noise = torch.from_numpy(self.gen_update_noise()).to(self.device)
            # noise for target
            target_noise = torch.bmm(update_noise, z_batch.unsqueeze(-1)) * self.noise_coef
            target = target_noise.squeeze(-1) + r_batch.unsqueeze(-1)
            predict = self.model(update_noise, input_ids, attention_mask)
            if self.action_num > 1:
                a_one_hot = F.one_hot(a_batch, self.action_num).to(predict.dtype)  # (None, n_a)
                if self.last_token:
                    predict = torch.einsum("bka,ba->bk", predict, a_one_hot)  # (None, NpS)
                else:
                    predict = torch.einsum("bska,ba->bsk", predict, a_one_hot)
                    target = target.unsqueeze(1).expand_as(predict)
            else:
                predict = predict.squeeze(-1)
                if not self.last_token:
                    target = target.unsqueeze(1).expand_as(predict)
            diff = (target - predict).pow(2)
            if not self.last_token:
                diff = diff.sum(1)
            loss = diff.mean(-1).mean(0)

        for param_group in self.optimizer.param_groups:
            param_group["weight_decay"] = self.weight_decay / len(self.buffer)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        results = {"loss": loss.item()}
        return results

    def predict(self, input_ids, attention_mask, num=1):
        if self.model_type == "linear":
            action_noise = None
        else:
            action_noise = self.gen_action_noise(dim=num)
        with torch.no_grad():
            p_a = self.model(action_noise, input_ids, attention_mask)  # .cpu().numpy()
            if not self.last_token:
                p_a = p_a[:, -1].squeeze(1)
            if self.action_num > 1:
                a = _random_argmax(p_a)
            else:
                a = torch.where(p_a > self.threshold, torch.ones_like(p_a), torch.zeros_like(p_a)).squeeze(-1)
        return a.cpu().numpy()

    def set_action_noise(self):
        args = {"M": self.noise_dim}
        if self.action_noise == "gs":
            self.gen_action_noise = partial(sample_action_noise, "Gaussian", **args)
        elif self.action_noise == "sp":
            self.gen_action_noise = partial(sample_action_noise, "Sphere", **args)
        elif self.action_noise == "pn":
            self.gen_action_noise = partial(sample_action_noise, "UnifCube", **args)
        elif self.action_noise == "pm":
            self.gen_action_noise = partial(sample_action_noise, "PMCoord", **args)
        elif self.action_noise == "oh":
            self.gen_action_noise = partial(sample_action_noise, "OH", **args)
        elif self.action_noise == "sps":
            self.gen_action_noise = partial(sample_action_noise, "Sparse", **args)
        elif self.action_noise == "spc":
            self.gen_action_noise = partial(
                sample_action_noise, "SparseConsistent", **args
            )

    def set_update_noise(self):
        args = {"M": self.noise_dim, "dim": self.NpS, "batch_size": self.batch_size}
        if self.update_noise == "gs":
            self.gen_update_noise = partial(sample_update_noise, "Gaussian", **args)
        elif self.update_noise == "sp":
            self.gen_update_noise = partial(sample_update_noise, "Sphere", **args)
        elif self.update_noise == "pn":
            self.gen_update_noise = partial(sample_update_noise, "UnifCube", **args)
        elif self.update_noise == "pm":
            self.gen_update_noise = partial(sample_update_noise, "PMCoord", **args)
        elif self.update_noise == "oh":
            self.gen_update_noise = partial(sample_update_noise, "OH", **args)
        elif self.update_noise == "sps":
            self.gen_update_noise = partial(sample_update_noise, "Sparse", **args)
        elif self.update_noise == "spc":
            self.gen_update_noise = partial(
                sample_update_noise, "SparseConsistent", **args
            )
