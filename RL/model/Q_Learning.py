import numpy as np
import random

class QLearningAgent:
    def __init__(self, agent_type, state_size, action_size=3, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.99, exploration_min=0.01):
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

    def rsi_decision(self, rsi_value):
        # spezifische Logik für RSI-Entscheidungen 
        if rsi_value > 70:
            return 1  # Verkaufen
        elif rsi_value < 30:
            return -1  # Kaufen
        else:
            return 0  # Halten
        
    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        q_update = reward
        if not done:
            q_update += self.discount_factor * np.amax(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (q_update - self.q_table[state, action])

        if done:
            self.exploration_rate *= self.exploration_decay
            self.exploration_rate = max(self.exploration_min, self.exploration_rate)
