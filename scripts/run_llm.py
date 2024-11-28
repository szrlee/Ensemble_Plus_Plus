# %%
""" Packages import """
import os, sys

sys.path.append(os.getcwd())
import json
import argparse
import expe as exp
import numpy as np

import pickle as pkl
import utils
import time

# random number generation setup
np.random.seed(46)

# configurations
from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument("--game", type=str, default="hatespeech")
    parser.add_argument("--time-period", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=512)
    parser.add_argument("--n-arms", type=int, default=1)
    parser.add_argument("--env-threshold", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.1)
    # algorithm config
    parser.add_argument("--action-num", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--method", type=str, default="LLM")
    parser.add_argument("--noise-dim", type=int, default=4)
    parser.add_argument("--NpS", type=int, default=8)
    parser.add_argument("--z-coef", type=float, default=0.01)
    parser.add_argument("--action-noise", type=str, default="sp")
    parser.add_argument("--update-noise", type=str, default="pm")
    parser.add_argument("--buffer-noise", type=str, default="sp")
    parser.add_argument("--prior-scale", type=float, default=0.2)
    parser.add_argument("--posterior-scale", type=float, default=0.1)
    parser.add_argument("--feature-sg", type=int, default=1, choices=[0, 1])
    # model config
    parser.add_argument(
        "--model-type",
        type=str,
        default="ensemble++",
        choices=["linear", "ensemble+", "ensemble++"],
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        default="pythia-14m",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "pythia-14m"],
    )
    parser.add_argument("--use-pretrained", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-lora", type=int, default=0, choices=[0, 1])
    parser.add_argument("--fine-tune", type=int, default=1, choices=[0, 1])
    parser.add_argument("--last-token", type=int, default=1, choices=[0, 1])
    parser.add_argument("--embed-init", type=int, default=0, choices=[0, 1])
    parser.add_argument("--hidden-transform", type=int, default=0, choices=[0, 1])
    # optimizer config
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    # buffer config
    parser.add_argument("--buffer-size", type=int, default=10000)
    # update config
    parser.add_argument("--update-start", type=int, default=20)
    parser.add_argument("--update-num", type=int, default=1)
    parser.add_argument("--update-freq", type=int, default=10)
    # other config
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--n-expe", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--log-dir", type=str, default="~/results/bandit")
    args = parser.parse_known_args()[0]
    return args


args = get_args()

dir = f"{args.game.lower()}_{args.model_type}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
path = os.path.expanduser(os.path.join(args.log_dir, args.game, dir))
os.makedirs(path, exist_ok=True)

noise_param = {
    "linear": {
        "prior_scale": 0.0,
    },
    "ensemble++": {
        "action_noise": args.action_noise,
        "update_noise": args.update_noise,
        "buffer_noise": args.buffer_noise,
    },
    "ensemble+": {
        "action_noise": "oh",
        "update_noise": "oh",
        "buffer_noise": "gs",
    },
}

param = {
    "log_interval": args.log_interval,
    "action_num": args.action_num,
    "threshold": args.threshold,
    "noise_dim": args.noise_dim,
    "NpS": args.NpS,
    "z_coef": args.z_coef,
    "action_noise": args.action_noise,
    "update_noise": args.update_noise,
    "buffer_noise": args.buffer_noise,
    "prior_scale": args.prior_scale,
    "posterior_scale": args.posterior_scale,
    "feature_sg": args.feature_sg,
    "optim": args.optim,
    "lr": args.lr,
    "batch_size": args.batch_size,
    "weight_decay": args.weight_decay,
    "update_start": args.update_start,
    "update_num": args.update_num,
    "update_freq": args.update_freq,
    "buffer_size": args.buffer_size,
    "model_type": args.model_type,
    "llm_name": args.llm_name,
    "use_pretrained": args.use_pretrained,
    "use_lora": args.use_lora,
    "fine_tune": args.fine_tune,
    "last_token": args.last_token,
    "embed_init": args.embed_init,
    "hidden_transform": args.hidden_transform,
    **noise_param[args.model_type],
}


game_config = {
    "n_features": args.n_features,
    "n_arms": args.n_arms,
    "llm_name": args.llm_name,
    "threshold": args.env_threshold,
}

with open(os.path.join(path, "config.json"), "wt") as f:
    f.write(
        json.dumps(
            {
                "methods_param": param,
                "game_config": game_config,
                "user_config": vars(args),
            },
            indent=4,
        )
        + "\n"
    )
    f.flush()
    f.close()


# %%
# Regret
expe_params = {
    "n_expe": args.n_expe,
    "T": args.time_period,
    "method": args.method,
    "param_dic": param,
    "path": path,
    "problem": args.game,
    "seed": args.seed,
    **game_config,
}
lin = exp.Textual_expe(**expe_params)
# %%
