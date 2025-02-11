import numpy as np
import torch
from gymnasium import ObservationWrapper, spaces
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from utils import compute_frontier_scores


class CnnWrapper(ObservationWrapper):
    def __init__(self, env, size):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, size, size), dtype=np.float32)

    def observation(self, obs):
        # Transform the observation into a tensor with the necessary channels
        grid = obs['map']
        agents = obs['agents']
        scores = compute_frontier_scores(grid)

        size = grid.shape[0]
        agent_tensor = np.zeros((size, size), dtype=float)
        for x, y in agents:
            agent_tensor[x, y] = 1

        map_tensor = torch.from_numpy(grid).float()
        score_tensor = torch.from_numpy(scores).float()
        agent_tensor = torch.from_numpy(agent_tensor).float()

        state_tensor = torch.stack([map_tensor, score_tensor, agent_tensor,])

        return state_tensor.numpy()


# create larger grid environments incrementally
class CurriculumCallback(BaseCallback):
    def __init__(self, check_freq, grid_size_start=5, grid_size_max=25, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.grid_size_start = grid_size_start
        self.grid_size_current = grid_size_start
        self.grid_size_max = grid_size_max

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if self.grid_size_current < self.grid_size_max:
                self.grid_size_current += 1
                self.training_env.env_method("update_grid_size",
                                             self.grid_size_current)
        return True


class TQDMCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.episode_count = 0  # To keep track of the number of episodes
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        timesteps_done = self.num_timesteps
        if self.locals['dones'][0]:
            self.episode_count += 1

        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])

        mean_reward = sum(self.episode_rewards[-10:]) / len(self.episode_rewards[-10:]) if self.episode_rewards else 0
        mean_length = sum(self.episode_lengths[-10:]) / len(self.episode_lengths[-10:]) if self.episode_lengths else 0

        self.pbar.n = timesteps_done
        self.pbar.set_postfix(
            Episodes=f"{self.episode_count}",
            Episode_Length=f"{mean_length:.2f}",
            Mean_Reward=f"{mean_reward:.2f}"
        )
        self.pbar.refresh()
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()