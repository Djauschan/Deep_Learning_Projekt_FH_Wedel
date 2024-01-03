import numpy as np
import random
from utils.read_config import Config_reader


class QLearningAgent:
    def __init__(self, agent_type, state_size, config_path='config/config.yml'):
        self.config = Config_reader(config_path)
        self.agent_type = agent_type
        self.state_size = state_size

        # Read parameters from the configuration
        self.learning_rate = self.config.get_parameter('learning_rate', 'train_parameters')
        self.discount_factor = self.config.get_parameter('discount_factor', 'train_parameters')
        self.exploration_rate = self.config.get_parameter('exploration_rate', 'train_parameters')
        self.exploration_decay = self.config.get_parameter('exploration_decay', 'train_parameters')
        self.exploration_min = self.config.get_parameter('exploration_min', 'train_parameters')
        self.action_size = self.config.get_parameter('action_size', 'train_parameters')

        # Anpassung für Aggregationsagenten
        if agent_type == 'aggregation':
            self.q_table = np.zeros((self.config.get_parameter('aggregation_state_size', 'aggregation'), 3))  # 4 Zustände, 3 Aktionen
        else:
            self.q_table = np.zeros((state_size, self.action_size))
  

    def act(self, state):
        """
        Determines an action based on the current state.
        :param state: The current state.
        :return: The chosen action.
        """
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """
        Updates the Q-table based on the action, reward, and the next state.
        :param state: Current state.
        :param action: Performed action.
        :param reward: Received reward.
        :param next_state: State after the action.
        :param done: Whether the episode is finished.
        """
        q_update = reward
        if not done:
            q_update += self.discount_factor * np.amax(self.q_table[next_state])
        
        self.q_table[state, action] += self.learning_rate * (q_update - self.q_table[state, action])

        # Update the exploration rate
        if done:
            self.exploration_rate *= self.exploration_decay
            self.exploration_rate = max(self.exploration_min, self.exploration_rate)
