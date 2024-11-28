import numpy as np
import time

from tqdm import tqdm
from utils import rd_argmax
from agent.nonlinear_sol import Ensemble_exp_Solution
from agent.llm_sol import LLMSolution


class Solution:
    def __init__(self, env):
        self.env = env
        self.expect_regret, self.n_a, self.d, self.features = (
            env.expect_regret,
            env.n_actions,
            env.n_features,
            env.features,
        )
        self.reward, self.eta = env.reward, env.eta

    def set_context(self):
        self.env.set_context()
        self.features = self.env.features

    def LLM(
        self,
        T,
        logger,
        log_interval=10,
        action_num=2,
        threshold=0.5,
        noise_dim=2,
        NpS=20,
        z_coef=None,
        action_noise="pn",
        update_noise="gs",
        buffer_noise="sp",
        prior_scale=1.0,
        posterior_scale=1.0,
        feature_sg=True,
        optim="Adam",
        lr=0.01,
        batch_size=32,
        weight_decay=0.0,
        buffer_size=None,
        update_num=2,
        update_start=32,
        update_freq=1,
        model_type="ensemble++",
        llm_name="gpt2",
        use_pretrained=True,
        use_lora=False,
        fine_tune=False,
        last_token=True,
        embed_init=False,
        hidden_transform=False,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = LLMSolution(
            self.n_a,
            self.d,
            action_num=action_num,
            threshold=threshold,
            noise_dim=noise_dim,
            NpS=NpS,
            noise_coef=z_coef,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            prior_scale=prior_scale,
            posterior_scale=posterior_scale,
            feature_sg=feature_sg,
            optim=optim,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            buffer_size=buffer_size,
            model_type=model_type,
            llm_name=llm_name,
            use_pretrained=use_pretrained,
            use_lora=use_lora,
            fine_tune=fine_tune,
            last_token=last_token,
            embed_init=embed_init,
            hidden_transform=hidden_transform,
            logger=logger,
        )

        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
        history_action = np.zeros((T, self.env.n_actions), dtype=np.int32)
        true_action = np.zeros((T, self.env.n_actions), dtype=np.int32)
        for t in tqdm(range(T)):
            self.set_context()
            input_ids, attention_mask = self.env.get_feature()
            a_t = model.predict(input_ids, attention_mask, num=self.n_a)
            r_t = self.reward(a_t)
            regret_t = self.expect_regret(a_t, self.features)
            reward[t], expected_regret[t] = r_t.mean(), regret_t.mean()
            history_action[t] = a_t
            true_action[t] = self.env.true_action

            transitions = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "a": a_t,
                "r": r_t,
            }
            model.put(transitions)
            # update model
            update_results = {}
            if t >= update_start and (t + 1) % update_freq == 0:
                for _ in range(update_num):
                    update_results = model.update()
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.record("accuracy", np.sum(history_action[:t+1] == true_action[:t+1]) / ((t + 1) * self.env.n_actions))
                logger.record("action", np.sum(history_action) / ((t + 1) * self.env.n_actions))
                logger.record("reward", reward[t])
                logger.record("regret", expected_regret[t])
                for key, value in update_results.items():
                    logger.record(key, value)
                logger.dump(t)
        return reward, expected_regret

    def Ensemble_exp(
        self,
        T,
        logger,
        noise_dim=2,
        NpS=20,
        z_coef=None,
        action_noise="pn",
        update_noise="gs",
        buffer_noise="sp",
        prior_scale=1.0,
        posterior_scale=1.0,
        based_prior=False,
        feature_sg=True,
        hidden_sizes=(),
        class_num=1,
        optim="Adam",
        lr=0.01,
        batch_size=32,
        weight_decay=0.0,
        buffer_size=None,
        update_num=2,
        update_start=32,
        update_freq=1,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = Ensemble_exp_Solution(
            self.n_a,
            self.d,
            noise_dim=noise_dim,
            NpS=NpS,
            noise_coef=z_coef,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            prior_scale=prior_scale,
            posterior_scale=posterior_scale,
            based_prior=based_prior,
            feature_sg=feature_sg,
            hidden_sizes=hidden_sizes,
            class_num=class_num,
            optim=optim,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            buffer_size=buffer_size,
            model_type="ensemble++",
            logger=logger,
        )

        log_interval = T // 1000
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
        start_time = time.time()
        for t in range(T):
            time_start = time.time()
            self.set_context()
            value = model.predict(self.features)
            if class_num > 1:
                value = value[:, 1]
            a_t = rd_argmax(value)
            f_t, r_t = self.features[a_t], self.reward(a_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"f": f_t, "r": r_t, "a": a_t}
            model.put(transitions)
            # update model
            if t >= update_start and (t + 1) % update_freq == 0:
                for _ in range(update_num):
                    model.update()
            time_end = time.time()
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("reward", reward[t])
                logger.record("regret", expected_regret[t])
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.record("time", time_end - time_start)
                logger.dump(t)
        logger.info("Total time: {}".format(time.time() - start_time))
        return reward, expected_regret

    def EpiNet(
        self,
        T,
        logger,
        noise_dim=2,
        NpS=20,
        z_coef=None,
        action_noise="pn",
        update_noise="gs",
        buffer_noise="sp",
        prior_scale=1.0,
        posterior_scale=1.0,
        feature_sg=True,
        hidden_sizes=(),
        class_num=1,
        optim="Adam",
        lr=0.01,
        batch_size=32,
        weight_decay=0.0,
        buffer_size=None,
        update_num=2,
        update_start=32,
        update_freq=1,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = Ensemble_exp_Solution(
            self.n_a,
            self.d,
            noise_dim=noise_dim,
            NpS=NpS,
            noise_coef=z_coef,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            prior_scale=prior_scale,
            posterior_scale=posterior_scale,
            feature_sg=feature_sg,
            hidden_sizes=hidden_sizes,
            class_num=class_num,
            optim=optim,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            buffer_size=buffer_size,
            model_type="epinet",
            logger=logger,
        )

        log_interval = T // 1000
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
        start_time = time.time()
        for t in range(T):
            time_start = time.time()
            self.set_context()
            value = model.predict(self.features)
            if class_num > 1:
                value = value[:, 1]
            a_t = rd_argmax(value)
            f_t, r_t = self.features[a_t], self.reward(a_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"f": f_t, "r": r_t, "a": a_t}
            model.put(transitions)
            # update model
            if t >= update_start and (t + 1) % update_freq == 0:
                for _ in range(update_num):
                    model.update()
            time_end = time.time()
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("reward", reward[t])
                logger.record("regret", expected_regret[t])
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.record("time", time_end - time_start)
                logger.dump(t)
        logger.info("Total time: {}".format(time.time() - start_time))
        return reward, expected_regret

    def Ensemble_ori(
        self,
        T,
        logger,
        noise_dim=2,
        NpS=20,
        z_coef=None,
        action_noise="pn",
        update_noise="gs",
        buffer_noise="sp",
        prior_scale=1.0,
        posterior_scale=1.0,
        feature_sg=True,
        hidden_sizes=(),
        class_num=1,
        ensemble_sizes=(),
        optim="Adam",
        lr=0.01,
        batch_size=32,
        weight_decay=0.0,
        buffer_size=None,
        update_num=2,
        update_start=32,
        update_freq=1,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = Ensemble_exp_Solution(
            self.n_a,
            self.d,
            noise_dim=noise_dim,
            NpS=NpS,
            noise_coef=z_coef,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            prior_scale=prior_scale,
            posterior_scale=posterior_scale,
            feature_sg=feature_sg,
            hidden_sizes=hidden_sizes,
            class_num=class_num,
            ensemble_sizes=ensemble_sizes,
            optim=optim,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            buffer_size=buffer_size,
            model_type="ensemble",
            logger=logger,
        )

        log_interval = T // 1000
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
        start_time = time.time()
        for t in range(T):
            time_start = time.time()
            self.set_context()
            value = model.predict(self.features)
            a_t = rd_argmax(value)
            f_t, r_t = self.features[a_t], self.reward(a_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"f": f_t, "r": r_t, "a": a_t}
            model.put(transitions)
            # update model
            if t >= update_start and (t + 1) % update_freq == 0:
                for _ in range(update_num):
                    model.update()
            time_end = time.time()
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("reward", reward[t])
                logger.record("regret", expected_regret[t])
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.record("time", time_end - time_start)
                logger.dump(t)
        logger.info("Total time: {}".format(time.time() - start_time))
        return reward, expected_regret
