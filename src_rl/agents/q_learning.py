import numpy as np
import random
import sys

class QLearningAgent:
    def __init__(self, agent_type, state_size, action_size=3, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.99, exploration_min=0.01):
        """
        Initialisiert einen Q-Learning-Agenten.
        :param agent_type: Bezeichnung des Agententyps.
        :param state_size: Größe des Zustandsraums.
        :param action_size: Anzahl der möglichen Aktionen.
        :param learning_rate: Lernrate (Schrittgröße der Aktualisierung).
        :param discount_factor: Diskontierungsfaktor für zukünftige Belohnungen.
        :param exploration_rate: Anfangswert für die Erkundungsrate.
        :param exploration_decay: Zerfallsrate der Erkundungsrate.
        :param exploration_min: Minimale Erkundungsrate.
        """
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
        """
        Entscheidet Aktion basierend auf RSI-Wert.
        :param rsi_value: RSI-Wert.
        :return: Aktion basierend auf RSI-Wert.
        """
        if rsi_value > 70:
            return 1  # Verkaufen
        elif rsi_value < 30:
            return -1  # Kaufen
        else:
            return 0  # Halten

    def act(self, state):
        """
        Bestimmt Aktion basierend auf dem aktuellen Zustand.
        :param state: Der aktuelle Zustand.
        :return: Die gewählte Aktion.
        """
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)  # Zufällige Aktion zur Erkundung
        return np.argmax(self.q_table[state])  # Beste Aktion basierend auf der Q-Tabelle

    def learn(self, state, action, reward, next_state, done):
        """
        Aktualisiert die Q-Tabelle basierend auf der Aktion, Belohnung und dem nächsten Zustand.
        :param state: Aktueller Zustand.
        :param action: Durchgeführte Aktion.
        :param reward: Erhaltene Belohnung.
        :param next_state: Zustand nach der Aktion.
        :param done: Ob die Episode beendet ist.
        """
        q_update = reward
        # print(f'{q_update=}')
        # if not done:
        #     q_update += self.discount_factor * np.amax(self.q_table[next_state])  # Aktualisierung der Q-Werte
        #     print(f'{self.discount_factor * np.amax(self.q_table[next_state])=}')
        # print(f'{q_update=}')
        # print(f'{self.learning_rate=}')
        # print(f'{q_update - self.q_table[state, action]=}')
        # print(f'{self.learning_rate * (q_update - self.q_table[state, action])=}')
    
        self.q_table[state, action] += self.learning_rate * (q_update - self.q_table[state, action])

        # Aktualisieren der Erkundungsrate
        if done:
            self.exploration_rate *= self.exploration_decay
            self.exploration_rate = max(self.exploration_min, self.exploration_rate)
            
