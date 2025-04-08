# File: envs/snake_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=(5, 5)):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        
        #Action states: up, down, left, right (0, 1, 2, 3)
        self.action_space = spaces.Discrete(4) 
        
        #Observation space: a grid where each cell can represent snake, reward, or empty
        self.observation_space = spaces.Box(low=0,high=2, shape=(grid_size[0], grid_size[1]), dtype=np.int32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        #Make all states 0
        self.state = np.zeros(self.grid_size, dtype=np.int32)
        self.snake = [(self.grid_size[0] // 2, self.grid_size[1] // 2)]
        self._place_reward()
        # Clear info dict
        return self.state, {}
