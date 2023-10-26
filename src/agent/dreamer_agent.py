# ==
# Default agent class
# ==
import os

import numpy as np
import torch
from torch import nn
from torch import distributions as torchd

import models  # TODO: move these files to src too
import tools
import exploration as expl


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, logger, dataset, 
                 log_every, batch_size, batch_length, train_ratio, reset_every, 
                 explore_until, expl_behavior_type, action_repeat, pretrain,
                 video_pred_log, eval_state_mean, actor_dist, expl_amount, 
                 eval_noise, behavior_stop_grad, collect_dyn_sample, 
                 device, compile,
                 wm_config):
        super(Dreamer, self).__init__()

        self._logger = logger
        self._dataset = dataset

        self._should_log = tools.Every(log_every)
        batch_steps = batch_size * batch_length
        self._should_train = tools.Every(batch_steps / train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(reset_every)
        self._should_expl = tools.Until(int(explore_until / action_repeat))

        self._num_actions = act_space.n if hasattr(act_space, "n") \
            else act_space.shape[0]

        self._expl_behavior_type = expl_behavior_type
        self._device = device

        self._collect_dyn_sample = collect_dyn_sample
        self._pretrain = pretrain
        self._video_pred_log = video_pred_log
        self._action_repeat = action_repeat
        self._eval_state_mean = eval_state_mean
        self._actor_dist = actor_dist
        self._expl_amount = expl_amount
        self._eval_noise = eval_noise

        self._metrics = {}

        # this is update step
        self._step = logger.step // action_repeat
        self._update_count = 0
        
        # set up
        wm_config.num_actions = self._num_actions
        self._wm = models.WorldModel(obs_space, act_space, self._step, wm_config)
        self._task_behavior = models.ImagBehavior(
            wm_config, self._wm, behavior_stop_grad,
        )

        if (
            compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            #random=lambda: expl.Random(config, act_space),
            #plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[expl_behavior_type]().to(self._device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if self._should_reset(step):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]
        if training:
            steps = (
                self._pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            batch_size = len(obs["image"])
            latent = self._wm.dynamics.initial(len(obs["image"]))
            action = torch.zeros((batch_size, self._num_actions)).to(
                self._device
            )
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self._collect_dyn_sample
        )
        if self._eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._actor_dist == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._num_actions
            )
        action = self._exploration(action, training)
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _exploration(self, action, training):
        amount = self._expl_amount if training else self._eval_noise
        if amount == 0:
            return action
        if "onehot" in self._actor_dist:
            probs = amount / self._num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._expl_behavior_type != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


