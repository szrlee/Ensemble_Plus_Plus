# %%
""" Packages import """
import os, sys
sys.path.append(os.getcwd())
import json
import time
import numpy as np
import argparse
import expe as exp

np.random.seed(2024)

def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument("--game", type=str, default="Russo")
    parser.add_argument("--time-period", type=int, default=1000)
    parser.add_argument("--n-context", type=int, default=1)
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--n-arms", type=int, default=50)
    parser.add_argument("--all-arms", type=int, default=1000)
    parser.add_argument("--freq-task", type=int, default=1, choices=[0, 1])
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0)
    # algorithm config
    parser.add_argument("--method", type=str, default="Ensemble++", choices=["Ensemble++", "EpiNet", "Ensemble+"])
    parser.add_argument("--noise-dim", type=int, default=8)
    parser.add_argument("--NpS", type=int, default=16)
    parser.add_argument("--z-coef", type=float, default=0.01)
    parser.add_argument("--action-noise", type=str, default="sp")
    parser.add_argument("--update-noise", type=str, default="pm")
    parser.add_argument("--buffer-noise", type=str, default="sp")
    parser.add_argument("--prior-scale", type=float, default=0.1)
    parser.add_argument("--posterior-scale", type=float, default=0.1)
    parser.add_argument("--based-prior", type=int, default=0, choices=[0, 1])
    parser.add_argument("--feature-sg", type=int, default=1, choices=[0, 1])
    parser.add_argument("--lmcts-beta", type=float, default=0.01)
    parser.add_argument("--NUCB-nu", type=float, default=1.0)
    # model config
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-layer", type=int, default=2)
    parser.add_argument("--ensemble-size", type=int, default=64)
    parser.add_argument("--ensemble-layer", type=int, default=0)
    # optimizer config
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    # buffer config
    parser.add_argument("--buffer-size", type=int, default=10000)
    # update config
    parser.add_argument("--update-start", type=int, default=128)
    parser.add_argument("--update-num", type=int, default=1)
    parser.add_argument("--update-freq", type=int, default=1)
    # other config
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--n-expe", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="~/results/bandit")
    args = parser.parse_known_args()[0]
    if args.game == "quadratic":
        args.prior_scale = 5.0
        args.posterior_scale = 1.0
        args.update_num = 4
    elif args.game == "neural_classification":
        args.prior_scale = 0.1
        args.posterior_scale = 0.1
        args.update_num = 1
        args.update_freq = 10
    elif args.game == "neural_regression":
        args.prior_scale = 0.2
        args.posterior_scale = 0.1
        args.update_num = 10
    elif args.game == "quadratic_lmcts":
        args.prior_scale = 5.0
        args.posterior_scale = 1.0
        args.update_num = 20
    elif args.game == "shuttle":
        args.prior_scale = 0.2
        args.posterior_scale = 0.1
        args.update_num = 50
    elif args.game == "mushroom":
        args.prior_scale = 1.0
        args.posterior_scale = 0.1
        args.update_num = 50
        args.eta = 1.0
        args.sigma = 10.0
        args.n_context = args.all_arms // args.n_arms
    elif "Russo" in args.game:
        args.n_context = 20
        args.prior_scale = 5.0
        args.posterior_scale = 1.0
        args.update_num = 10
        args.eta = 1.0
        args.sigma = 10.0
        args.n_context = args.all_arms // args.n_arms
    return args


args = get_args()

tag = f"{args.game.lower()}_{args.method}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
path = os.path.expanduser(os.path.join(args.log_dir, args.game, tag))
os.makedirs(path, exist_ok=True)

args.hidden_sizes = [args.hidden_size] * args.hidden_layer
based_param = {
    "noise_dim": args.noise_dim,
    "NpS": args.NpS,
    "z_coef": args.z_coef,
    "action_noise": args.action_noise,
    "update_noise": args.update_noise,
    "buffer_noise": args.buffer_noise,
    "prior_scale": args.prior_scale,
    "posterior_scale": args.posterior_scale,
    "feature_sg": args.feature_sg,
    "hidden_sizes": args.hidden_sizes,
    "class_num": 2 if args.game.endswith("classification") else 1,
    "optim": args.optim,
    "lr": args.lr,
    "batch_size": args.batch_size,
    "weight_decay": args.weight_decay,
    "update_start": args.update_start,
    "update_num": args.update_num,
    "update_freq": args.update_freq,
    "buffer_size": args.buffer_size,
}

param = {
    "Ensemble++": {
        **based_param,
        "based_prior": args.based_prior,
    },
    "EpiNet": {
        **based_param,
        "action_noise": "gs",
        "update_noise": "gs",
    },
    "Ensemble+": {
        **based_param,
        "action_noise": "oh",
        "update_noise": "oh",
        "buffer_noise": "gs",
        "ensemble_sizes": [args.ensemble_size] * args.ensemble_layer,
    },
    "LMCTS": {
        **based_param,
        "prior_scale": 0.0,
        "batch_size": 0,
        "beta_inv": args.lmcts_beta,
    },
    "NeuralUCB": {
        **based_param,
        "prior_scale": 0.0,
        "batch_size": 0,
        "nu": args.NUCB_nu,
    },
}

game_config = {
    "n_features": args.n_features,
    "n_arms": args.n_arms,
    "T": args.time_period,
    "freq_task": args.freq_task,
    "eta": args.eta,
    "sigma": args.sigma,
    "all_arms": args.all_arms
}

with open(os.path.join(path, "config.json"), "wt") as f:
    f.write(
        json.dumps(
            {
                "methods_param": param[args.method],
                "game_config": game_config,
                "user_config": vars(args),
            },
            indent=4,
        )
        + "\n"
    )
    f.flush()
    f.close()

method_names = {
    "Ensemble++": "Ensemble_exp",
    "EpiNet": "EpiNet",
    "Ensemble+": "Ensemble_ori",
    "LMCTS": "LMCTS",
    "NeuralUCB": "NeuralUCB",
}
# %%
# Regret
expe_params = {
    "problem": args.game,
    "n_expe": args.n_expe,
    "method": method_names[args.method],
    "param_dic": param[args.method],
    "path": path,
    "seed": args.seed,
    **game_config,
}
if args.game in ["quadratic", "neural_classification", "neural_regression", "quadratic_lmcts"]:
    results = exp.Synthetic_expe(**expe_params)
elif args.game in ["shuttle", "mushroom"]:
    results = exp.UCIBandit_expe(**expe_params)
elif args.game in ["Russo", "FreqRusso"]:
    results = exp.LinearBandit_expe(n_context=args.n_context, **expe_params)

# %%
