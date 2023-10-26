# 

import os
import functools
from pathlib import Path

import hydra
import torch 
import torch.distributions as torchd

import envs.wrappers as wrappers
import tools
import src.error_logging as errlog
from parallel import Parallel, Damy


DB_LOGGER = errlog.get_debug_logger("train_smodel_data_collect")

to_np = lambda x: x.detach().cpu().numpy()

def count_steps(folder):
    """Count the nuumber of steps for episodes saved in folder"""
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_env(env_cfg, mode):
    suite, task = env_cfg.suite, env_cfg.task
    config = None  # TODO: dummy only to prevent bugs, delete this 
    
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, **env_cfg.env_kwargs)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        raise NotImplementedError
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        raise NotImplementedError
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        raise NotImplementedError
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, **env_cfg.env_kwargs)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        raise NotImplementedError
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)

    env = wrappers.TimeLimit(env, env_cfg.env_frames_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)

    return env


def make_dataset(episodes_cache, batch_length, batch_size):
    generator = tools.sample_episodes(episodes_cache, batch_length)
    dataset = tools.from_generator(generator, batch_size)
    return dataset

class Workspace:
    def __init__(self, cfg):
        # Set up workspace
        self.work_dir = Path.cwd()
        DB_LOGGER.info(f'workspace: {self.work_dir}')
        self.cfg = cfg
        DB_LOGGER.debug(f"cfg: {cfg}")

        # Reproducitiblity
        tools.set_seed_everywhere(cfg.seed)
        if cfg.deterministic_run:
            tools.enable_deterministic_run()

        # Directories
        # self.log_dir = self.work_dir / "log_dir"
        self.train_eps_dir = self.work_dir / "train_eps"  # storing episodes
        self.eval_eps_dir = self.work_dir / "eval_eps"

        self.total_train_steps = self.cfg.train_steps // self.cfg.action_repeat
        self.eval_every_steps = self.cfg.eval_every // self.cfg.action_repeat
        self.log_every_steps = self.cfg.log_every // self.cfg.action_repeat
        self.env_steps_limit = self.cfg.env_frames_limit // self.cfg.action_repeat
        
        # Make directories for episodes
        self.work_dir = Path(self.work_dir).expanduser()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        DB_LOGGER.info(f"working dir: {self.work_dir}")

        self.train_eps_dir.mkdir(parents=True, exist_ok=True)
        self.eval_eps_dir.mkdir(parents=True, exist_ok=True)
        step = count_steps(self.train_eps_dir)

        # step in logger is environmental step
        self.logger = tools.Logger(self.work_dir, self.cfg.action_repeat * step)
        DB_LOGGER.debug(f'logger: {self.logger}')

        # Loading caches of episodes
        if self.cfg.offline_traindir:
            # directory = config.offline_traindir.format(**vars(config))
            raise NotImplementedError  # TODO: not sure what happens here 
        else:
            directory = self.train_eps_dir 
        if self.cfg.offline_evaldir:
            # directory = config.offline_evaldir.format(**vars(config))
            raise NotImplementedError  # TODO: not sure what happens here 
        else:
            directory = self.eval_eps_dir
        
        self.train_eps_cache = tools.load_episodes(
            directory, limit=self.cfg.dataset_size)
        self.eval_eps_cache = tools.load_episodes(
            directory, limit=1)

        # Create environments
        DB_LOGGER.info(f'Creating envs.')
        make = lambda mode: make_env(self.cfg.envs, mode)  # TODO make this
        train_envs = [make("train") for _ in range(cfg.n_parallel_envs)]
        eval_envs = [make("eval") for _ in range(cfg.n_parallel_envs)]

        if self.cfg.use_parallel_env:
            self.train_envs = [Parallel(env, "process") for env in train_envs]
            self.eval_envs = [Parallel(env, "process") for env in eval_envs]
        else:
            self.train_envs = [Damy(env) for env in train_envs]
            self.eval_envs = [Damy(env) for env in eval_envs]
        
        DB_LOGGER.debug(f"train_envs: {self.train_envs}")
        DB_LOGGER.debug(f"eval_envs: {self.eval_envs}")
        
        # Action space
        self.action_space = self.train_envs[0].action_space
        self.num_actions = self.action_space.n if \
            hasattr(self.action_space, "n") else self.action_space.shape[0]
        DB_LOGGER.debug(f"action space: {self.action_space}")

        pass

    def _prefill_dataset(self):
        prefill = max(0, self.cfg.prefill_steps - 
                      count_steps(self.train_eps_dir))
        DB_LOGGER.info(
           f"Prefilling {prefill} steps of random actions into dataset path "
           f"{self.train_eps_dir} (up to {self.cfg.prefill_steps} steps)."
        )

        # Create random agent 
        if hasattr(self.action_space, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(self.num_actions).repeat(
                    self.cfg.n_parallel_envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(self.action_space.low).repeat(
                        self.cfg.n_parallel_envs, 1),
                    torch.Tensor(self.action_space.high).repeat(
                        self.cfg.n_parallel_envs, 1),
                ), 1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        # Simulate
        DB_LOGGER.debug("Starting simulating with random agent")
        state = tools.simulate(
            random_agent,
            self.train_envs,
            self.train_eps_cache,
            self.train_eps_dir,
            self.logger,
            limit=self.cfg.dataset_size,
            steps=prefill,
        )
        self.logger.step += prefill * self.cfg.action_repeat
        DB_LOGGER.debug(f"Logger: at ({self.logger.step} steps).") 

    def train(self):
        # Optionally pre-fill
        state = None
        if not self.cfg.offline_traindir:
            state = self._prefill_dataset()

        # Making datasets
        train_dataset = make_dataset(self.train_eps_cache, 
                                     self.cfg.data.batch_length, 
                                     self.cfg.data.batch_size)
        eval_dataset = make_dataset(self.eval_eps_cache, 
                                    self.cfg.data.batch_length, 
                                    self.cfg.data.batch_size)

        # Create agent
        agent = hydra.utils.instantiate(
            self.cfg.agent, 
            obs_space=self.train_envs[0].observation_space,
            act_space=self.train_envs[0].action_space,
            logger=self.logger, 
            dataset=train_dataset,
        ).to(self.cfg.device)
        agent.requires_grad_(requires_grad=False)
        DB_LOGGER.debug(f"Created agent: {agent}.") 

        # Load checkpoint?
        if (self.work_dir / "latest.pt").exists():
            checkpoint = torch.load(self.work_dir / "latest.pt")
            agent.load_state_dict(checkpoint["agent_state_dict"])
            tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
            agent._should_pretrain._once = False
    
        # make sure eval will be executed once after config.steps
        while agent._step < self.total_train_steps + self.eval_every_steps:
            self.logger.write()
            if self.cfg.eval_episode_num > 0:
                print("Start evaluation.")
                eval_policy = functools.partial(agent, training=False)
                tools.simulate(
                    eval_policy,
                    self.eval_envs,
                    self.eval_eps_cache,
                    self.eval_eps_dir,
                    self.logger,
                    is_eval=True,
                    episodes=self.cfg.eval_episode_num,
                )
                if self.cfg.video_pred_log:
                    video_pred = agent._wm.video_pred(next(eval_dataset))
                    self.logger.video("eval_openl", to_np(video_pred))
            print("Start training.")
            state = tools.simulate(
                agent,
                self.train_envs,
                self.train_eps_cache,
                self.train_eps_dir,
                self.logger,
                limit=self.cfg.dataset_size,
                steps=self.eval_every_steps,
                state=state,
            )
            items_to_save = {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            }
            torch.save(items_to_save, self.work_dir / "latest.pt")
        for env in self.train_envs + self.eval_envs:
            try:
                env.close()
            except Exception:
                pass
            



@hydra.main(config_path='.', config_name='train_dreamer', version_base=None)
def main(cfg):
    # log.info(cfg)  # TODO delete?
    os.environ["HYDRA_FULL_ERROR"] = "1"  # NOTE: for debugging hydra only
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # NOTE: for debugging cuda only

    workspace = Workspace(cfg)

    #if cfg.load.from_ckpt:
    #    workspace.load_snapshot()
    
    workspace.train()

if __name__ == '__main__':
    main()