import numpy as np
import random

def calculate_state_from_std(price, ma_value, std_dev):
    deviation_in_std = (price - ma_value) / std_dev
    if deviation_in_std < -3:
        return 0
    elif deviation_in_std < -2:
        return 1
    elif deviation_in_std < -1:
        return 2
    elif deviation_in_std < 1:
        return 3
    elif deviation_in_std < 2:
        return 4
    elif deviation_in_std < 3:
        return 5
    else:
        return 6

class QLearningAgent:
    def __init__(self, agent_type, state_size, action_size=2, learning_rate=0.9, discount_factor=0.95, exploration_rate=0.9, exploration_decay=0.99, exploration_min=0.01, ma_values=None, std_devs=None):
        self.agent_type = agent_type
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.q_table = np.zeros((state_size, action_size))
        self.weight = 1
        self.ma_values = ma_values
        self.std_devs = std_devs

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.choice([-1, 1])  # Gibt entweder -1 oder 1 zurück
        return np.argmax(self.q_table[state])

    #
    def rsi_decision(rsi_value):
    if rsi_value < 30:
        return 1  # Kaufsignal
    elif rsi_value > 70:
        return -1  # Verkaufssignal
    else:
        return 0  # Keine Aktion


    def learn(self, state, action, reward, next_state, done):
        q_update = reward
        if not done:
            q_update += self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (q_update - self.q_table[state, action])

        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)
    
    def calculate_state(self, price):
        if self.ma_values and self.std_devs:
            for ma_value, std_dev in zip(self.ma_values, self.std_devs):
                state = calculate_state_from_std(price, ma_value, std_dev)
                return state
        else:
            raise ValueError("MA values and Standard Deviations are not set for the agent.")
