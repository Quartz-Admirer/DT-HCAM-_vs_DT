# envs/minigrid_memory_env.py

import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Goal
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace

class MiniGridMemoryEnv(MiniGridEnv):
    def __init__(self, size=7, seed=None):
        super().__init__(
            mission_space=MissionSpace(
                mission_func=lambda: "reach the goal" 
            ),
            width=size,
            height=size,
            max_steps=4*size*size,
            see_through_walls=False,
            agent_view_size=7
        )
        self.custom_seed = seed

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self.custom_seed
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        goal_pos = (width // 2, height // 2)
        self.put_obj(Goal(), *goal_pos)

        self.agent_pos = (1, 1)
        self.agent_dir = 0
        
        self.mission = "reach the goal"
