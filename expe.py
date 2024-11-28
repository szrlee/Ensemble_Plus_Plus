""" Packages import """

import numpy as np

from env.finitecontext import (
    FiniteContextPaperLinModel,
    FiniteContextFreqPaperLinModel,
)
from env.synthetic import SyntheticNonlinModel
from env.data_multi import Bandit_multi
from env.nlp_dataset import HateSpeechEnv
from agent.solution import Solution
from utils import storeRegret


def Synthetic_expe(
    problem,
    n_expe,
    n_features,
    n_arms,
    T,
    method,
    param_dic,
    path,
    all_arms,
    freq_task=True,
    seed=2022,
    eta=0.1,
    sigma=1.0,
    **kwargs,
):
    models = [
        Solution(
            SyntheticNonlinModel(
                n_features,
                n_arms,
                all_actions=all_arms,
                reward_name=problem,
                freq_task=freq_task,
                eta=eta,
                sigma=sigma,
            )
        )
        for _ in range(n_expe)
    ]
    title = f"Synthetic Bandit Model  - n_arms: {n_arms} - n_features: {n_features} - reward: {problem}"\

    print("Begin experiments on '{}'".format(title))
    results = storeRegret(
        models, method, param_dic, n_expe, T, path, seed, use_torch=True
    )
    return results


def UCIBandit_expe(
    problem,
    n_expe,
    T,
    method,
    param_dic,
    path,
    freq_task=True,
    seed=2022,
    eta=0.1,
    sigma=1.0,
    **kwargs,
):
    models = [
        Solution(
            Bandit_multi(
                problem, freq_task=freq_task, eta=eta, sigma=sigma,
            )
        )
        for _ in range(n_expe)
    ]
    title = f"Real Bandit Model  - {problem}"

    print("Begin experiments on '{}'".format(title))
    results = storeRegret(
        models, method, param_dic, n_expe, T, path, seed, use_torch=True
    )
    return results


def LinearBandit_expe(
    problem,
    n_expe,
    n_context,
    n_features,
    n_arms,
    T,
    method,
    param_dic,
    path,
    seed=2022,
    eta=0.1,
    sigma=1.0,
    **kwargs,
):
    if problem == "FreqRusso":
        u = 1 / np.sqrt(5)
        models = [
            Solution(
                FiniteContextFreqPaperLinModel(
                    u, n_context, n_features, n_arms, eta=eta, sigma=sigma
                )
            )
            for _ in range(n_expe)
        ]
        title = "Linear Gaussian Model (Freq MOD, Russo and Van Roy, 2018) - n_arms: {} - n_features: {}".format(
            n_arms, n_features
        )
    elif problem == "Russo":
        u = 1 / np.sqrt(5)
        models = [
            Solution(
                FiniteContextPaperLinModel(
                    u, n_context, n_features, n_arms, eta=eta, sigma=sigma
                )
            )
            for _ in range(n_expe)
        ]
        title = "Linear Gaussian Model (Bayes MOD, Russo and Van Roy, 2018) - n_arms: {} - n_features: {}".format(
            n_arms, n_features
        )
    else:
        raise NotImplementedError

    print("Begin experiments on '{}'".format(title))
    results = storeRegret(
        models, method, param_dic, n_expe, T, path, seed, use_torch=True
    )
    return results


def Textual_expe(
    n_expe,
    n_features,
    n_arms,
    T,
    method,
    param_dic,
    path,
    problem="hatespeech",
    llm_name="gpt2",
    threshold=0.5,
    seed=2022,
):
    if problem == "hatespeech":
        models = [Solution(HateSpeechEnv(n_features, n_arms, 
                                         llm_name=llm_name, threshold=threshold)) for _ in range(n_expe)]
        title = f"HateSpeech  - n_arms: {n_arms} - n_features: {n_features}"
    else:
        raise NotImplementedError

    print("Begin experiments on '{}'".format(title))
    results = storeRegret(
        models, method, param_dic, n_expe, T, path, seed, use_torch=True
    )
    return results
