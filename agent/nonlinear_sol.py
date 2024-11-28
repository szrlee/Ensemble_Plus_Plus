from typing import Sequence
from functools import partial

import sys

sys.path.append("..")

import numpy as np
import torch
import torch.nn.functional as F

from logger import Logger
from utils import sample_action_noise, sample_update_noise, sample_buffer_noise
from network import Ensemble_exp_Net, EnsembleNet, EpiNet


class ReplayBuffer:
    def __init__(
        self, buffer_size, buffer_shape, noise_type="sp"
    ):
        self.buffers = {
            key: np.empty([buffer_size, *shape], dtype=np.float32)
            for key, shape in buffer_shape.items()
        }
        self.buffer_size = buffer_size
        self.current_size = 0
        self.point = 0
        self.store_z = "z" in buffer_shape.keys()
        if self.store_z:
            self.noise_dim = buffer_shape["z"][-1]
            self.set_buffer_noise(noise_type)

    def set_buffer_noise(self, noise_type):
        args = {"M": self.noise_dim}
        if noise_type == "gs":
            self.gen_noise = partial(sample_buffer_noise, "Gaussian", **args)
        elif noise_type == "sp":
            self.gen_noise = partial(sample_buffer_noise, "Sphere", **args)
        elif noise_type == "pn":
            self.gen_noise = partial(sample_buffer_noise, "UnifCube", **args)
        elif noise_type == "pm":
            self.gen_noise = partial(sample_buffer_noise, "PMCoord", **args)
        elif noise_type == "oh":
            self.gen_noise = partial(sample_buffer_noise, "OH", **args)
        elif noise_type == "sps":
            self.gen_noise = partial(sample_buffer_noise, "Sparse", **args)
        elif noise_type == "spc":
            self.gen_noise = partial(sample_buffer_noise, "SparseConsistent", **args)

    def __len__(self):
        return self.current_size

    def _sample(self, index):
        f_data = self.buffers["f"][index]
        r_data = self.buffers["r"][index]
        if self.store_z:
            z_data = self.buffers["z"][index]
        else:
            z_data = None
        return f_data, r_data, z_data

    def reset(self):
        self.current_size = 0

    def put(self, transition):
        transition.pop("a")
        batch_size = 1
        idx = self._get_ordered_storage_idx(batch_size)
        for k, v in transition.items():
            self.buffers[k][idx] = v
        if self.store_z:
            z = self.gen_noise()
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


class Ensemble_exp_Solution:
    def __init__(
        self,
        n_action: int,
        n_feature: int,
        noise_dim: int,
        NpS: int = 20,
        noise_coef: float = 0.01,
        buffer_noise: str = "sp",
        action_noise: str = "sgs",
        update_noise: str = "pn",
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        based_prior: bool = False,
        feature_sg: bool = True,
        hidden_sizes: Sequence[int] = (),
        ensemble_sizes: Sequence[int] = (),
        class_num: int = 1,
        optim: str = "Adam",
        lr: float = 0.01,
        batch_size: int = 32,
        weight_decay: float = 0.01,
        buffer_size: int = 10000,
        model_type: str = "ensemble++",
        logger: Logger = None,
    ):
        self.action_dim = n_action
        self.feature_dim = n_feature

        self.noise_dim = noise_dim
        self.NpS = NpS
        self.noise_coef = noise_coef
        self.buffer_noise = buffer_noise
        self.action_noise = action_noise
        self.update_noise = update_noise
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.based_prior = based_prior
        self.feature_sg = feature_sg

        self.hidden_sizes = hidden_sizes
        self.ensemble_sizes = ensemble_sizes
        self.class_num = class_num

        self.optim = optim
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

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
            "in_features": self.feature_dim,
            "hidden_sizes": self.hidden_sizes,
            "action_num": self.class_num,
            "noise_dim": self.noise_dim,
            "prior_scale": self.prior_scale,
            "posterior_scale": self.posterior_scale,
            "device": self.device,
        }
        if self.model_type == "ensemble++":
            Net = Ensemble_exp_Net
            model_param.update({
                "feature_sg": self.feature_sg,
                "based_prior": self.based_prior,
            })
        elif self.model_type == "epinet":
            Net = EpiNet
        elif self.model_type == "ensemble":
            model_param.update({"ensemble_sizes": self.ensemble_sizes})
            Net = EnsembleNet
        else:
            raise NotImplementedError
        self.model = Net(**model_param).to(self.device)
        param_dict = {"Trainable": [], "Frozen": []}
        trainable_param_size, frozen_param_size = 0, 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_param_size += param.numel()
                param_dict["Trainable"].append(name)
            else:
                frozen_param_size += param.numel()
                param_dict["Frozen"].append(name)
        self.logger.info(f"Trainable parameters:\n", "\n".join(param_dict['Trainable']))
        self.logger.info(f"Frozen parameters:\n", "\n".join(param_dict['Frozen']))
        self.logger.info(f"Network structure:\n{str(self.model)}")
        self.logger.info(
            f"Network parameters: {sum(param.numel() for param in self.model.parameters() if param.requires_grad)}"
        )
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.optim == "Adam":
            self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim == "SGD":
            self.optimizer = torch.optim.SGD(trainable_params, lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise NotImplementedError

    def init_buffer(self):
        buffer_shape = {"f": (self.feature_dim,), "r": (), "z": (self.noise_dim,)}
        self.buffer = ReplayBuffer(self.buffer_size, buffer_shape, self.buffer_noise)

    def update(self):
        if self.batch_size is None:
            f_batch, r_batch, z_batch = self.buffer.sample_all()
        else:
            f_batch, r_batch, z_batch = self.buffer.sample(self.batch_size)
        self.learn(f_batch, r_batch, z_batch)

    def put(self, transition):
        self.buffer.put(transition)

    def learn(self, f_batch, r_batch, z_batch):
        z_batch = torch.FloatTensor(z_batch).to(self.device)
        f_batch = torch.FloatTensor(f_batch).to(self.device)
        r_batch = torch.FloatTensor(r_batch).to(self.device)

        # noise for update
        update_noise = torch.from_numpy(self.gen_update_noise(batch_size=len(r_batch))).to(self.device)
        # noise for target
        target_noise = torch.bmm(update_noise, z_batch.unsqueeze(-1)) * self.noise_coef

        predict = self.model(update_noise, f_batch)
        if self.class_num > 1:
            NpS = predict.shape[1]
            r_batch = r_batch.unsqueeze(-1).repeat(1, NpS).to(torch.int64)
            r_batch = r_batch.view(-1)
            predict = predict.view(-1, self.class_num)
            loss = F.cross_entropy(predict, r_batch)
        else:
            diff = target_noise.squeeze(-1) + r_batch.unsqueeze(-1) - predict
            diff = diff.pow(2).mean(-1)
            loss = diff.mean()

        for param_group in self.optimizer.param_groups:
            param_group["weight_decay"] = self.weight_decay / len(self.buffer)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_thetas(self, num=1):
        assert len(self.hidden_sizes) == 0, f"hidden size > 0"
        action_noise = self.gen_action_noise(dim=num)
        with torch.no_grad():
            thetas = self.model.out.get_thetas(action_noise).cpu().numpy()
        return thetas

    def predict(self, features, num=1):
        action_noise = self.gen_action_noise(dim=num)
        with torch.no_grad():
            p_a = self.model(action_noise, features).cpu().numpy()
        return p_a

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
        elif self.action_noise == "hoh":
            self.gen_action_noise = partial(sample_action_noise, "HOH", **args)
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
        elif self.update_noise == "hoh":
            self.gen_update_noise = partial(sample_update_noise, "HOH", **args)
        elif self.update_noise == "sps":
            self.gen_update_noise = partial(sample_update_noise, "Sparse", **args)
        elif self.update_noise == "spc":
            self.gen_update_noise = partial(
                sample_update_noise, "SparseConsistent", **args
            )
