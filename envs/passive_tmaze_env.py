# envs/passive_tmaze_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PassiveTMazeEnv(gym.Env):

    def __init__(self, maze_size=5):
        super().__init__()
        self.maze_size = maze_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=self.maze_size - 1, shape=(2,), dtype=np.int32
        )
        self.agent_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        if action == 0: 
            self.agent_pos[0] -= 1
        elif action == 1: 
            self.agent_pos[0] += 1
        elif action == 2:
            self.agent_pos[1] -= 1
        elif action == 3:
            self.agent_pos[1] += 1

        self.agent_pos = np.clip(self.agent_pos, 0, self.maze_size - 1)

        reward = 0.0
        done = False

        if np.array_equal(self.agent_pos, [self.maze_size//2, self.maze_size//2]):
            reward = 1.0
            done = True

        obs = self._get_obs()
        info = {}
        return obs, reward, done, False, info

    def _get_obs(self):
        return self.agent_pos.copy()

    def render(self):
        pass
