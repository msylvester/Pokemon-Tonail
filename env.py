import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game_controller import GameController
from collections import defaultdict
from actions import Actions
from config import ROM_PATH, EMULATION_SPEED
import random

class env_red(gym.Env):  # Inherit from gym.Env for compatibility
    def __init__(self, learning_rate=0.05, discount_factor=0.9):
        super(env_red, self).__init__()
        
        self.controller = GameController(ROM_PATH, EMULATION_SPEED)
        self.q_table = defaultdict(lambda: np.zeros(len(Actions.list())))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Define action and observation space
        self.action_space = spaces.Discrete(len(Actions.list()))
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=0, high=500, shape=(3,), dtype=np.int32),
            "battle": spaces.Discrete(2),
        })

    def reset(self, seed=None, options=None):
        # Set the seed if provided for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize the game state
        self.controller.load_state()
        self.visited_coords = set()
        self.battle = self.controller.is_in_battle()
        self.battle_reward_applied = False
        self.current_step = 0
        self.total_reward = 0
        self.steps_to_battle = None
        self.last_distance_reward = None

        # Get initial position with the correct shape
        position = self.controller.get_global_coords()
        if len(position) == 2:
            position = (*position, 0)  # Add a third coordinate if missing

        initial_state = {
            "position": np.array(position, dtype=np.int32),
            "battle": 1 if self.battle else 0,
        }
        self.previous_state = initial_state
        return initial_state, {}

    def step(self, action):
        self.current_step += 1

        # Perform the action in the game environment
        self.controller.perform_action(action)
        self.controller.pyboy.tick()

        # Check battle status
        self.battle = self.controller.is_in_battle()
        position = self.controller.get_global_coords()
        if len(position) == 2:
            position = (*position, 0)  # Ensure position has shape (3,)

        step_reward = self.calculate_reward(position)
        self.total_reward += step_reward

        next_state = {
            "position": np.array(position, dtype=np.int32),
            "battle": 1 if self.battle else 0,
        }

        done = self.current_step >= 300 or self.battle
        return next_state, step_reward, done, False, {}

    def calculate_reward(self, position):
        # # Example reward calculation
        # target_position = (100, 100)  # Example target coordinates
        # distance = np.linalg.norm(np.array(position) - np.array(target_position))

        # # Calculate a reward inversely proportional to the distance to target
        # reward = max(100 - distance, 0)  # Reward decreases as distance increases

        # Always return a numerical value (even if it’s 0)
        return 1

    def close(self):
        self.controller.close()
