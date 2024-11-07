import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game_controller import GameController
from collections import defaultdict
from actions import Actions
from config import ROM_PATH, EMULATION_SPEED
import random
import os
import pickle
import json

class env_red(gym.Env):  # Inherit from gym.Env for compatibility
    def __init__(self, learning_rate=0.05, discount_factor=0.9):
        super(env_red, self).__init__()
        
        self.controller = GameController(ROM_PATH, EMULATION_SPEED)
        self.q_table = defaultdict(lambda: np.zeros(len(Actions.list())))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.seen_coords = set()
        self.visited_coords = set()  # Initialize visited_coords here
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
    
        self.battle = self.controller.is_in_battle()
        self.battle_reward_applied = False
        self.current_step = 0
        self.total_reward = 0
        self.steps_to_battle = None
        self.last_distance_reward = None
        self.seen_coords = set()
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

    def step(self, action, manual=False):
        if not manual: 
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
        self.current_step += 1
        self.controller.pyboy.tick()
        position = self.controller.get_global_coords()
        step_reward = self.calculate_reward(position)
        self.total_reward += step_reward

        next_state = {
            "position": np.array(position, dtype=np.int32),
            "battle": 1 if self.battle else 0,
        }

        done = self.current_step >= 300 or self.battle
        return next_state, step_reward, False, False


    def save_episode_stats(self, episode_id):
        # Define the stats you want to save for each episode
        stats = {
            "total_reward": self.total_reward,
            "steps": self.current_step,
            "visited_coords": list(self.visited_coords),
            # Add any other relevant stats
        }
        
        # Ensure the directory exists
        os.makedirs("episodes", exist_ok=True)
        
        # Save stats as a JSON file
        with open(f"episodes/episode_{episode_id}.json", "w") as f:
            json.dump(stats, f, indent=4)

    def battle_reward(self):
        """
        Returns a reward if the agent has entered a new battle.
        The reward is given only once per battle encounter.
        """
        if self.battle and not self.battle_reward_applied:
            self.battle_reward_applied = True  # Mark that reward has been applied
            return 100  # Example battle reward
        elif not self.battle:
            # Reset battle reward flag when not in battle
            self.battle_reward_applied = False
        return 0
    
    def exploration_reward(self):
        position = tuple(self.controller.get_global_coords())
        
        if position not in self.seen_coords:
            self.seen_coords.add(position)
         
            return .1  # Reward for discovering a new location
        
        return 0  # No reward if the location was already visited
    def directional_reward(self, position):
        """
        Calculates a reward based on the distance to a target position (339, 95).
        The closer the agent is to this target, the higher the reward.
        """
    
        target_position = np.array([338,94])  # Assuming a 3D position with z = 0

        current_position = np.array(position)

        # Calculate Euclidean distance to target
        distance = np.linalg.norm(current_position - target_position)
       
        # Define the maximum reward and a scaling factor for proximity
        max_reward = 10  # Max reward when at target
        reward = max(max_reward - distance * 0.1, 0)  # Reward decreases with distance
 
        return reward
     
    def calculate_reward(self, position):
        # Sum up all the rewards
        battle_reward = self.battle_reward()
        exploration_reward = self.exploration_reward()

        dir_reward =self.directional_reward(self.controller.get_global_coords())
        #directional_reward = self.directional_reward(self.controller.get_global_coords())
        # Print each reward for better diagnostics
      
        # Return the total reward for the current step
        return battle_reward + dir_reward


    def close(self):
        self.controller.close()
