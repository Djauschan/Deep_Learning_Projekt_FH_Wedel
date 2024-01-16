import numpy as np


def aggregate_actions(aggregation_agent, actions):
    weighted_actions = np.zeros(3)  # Angenommen, es gibt 3 mögliche Aktionen
    for i, action in enumerate(actions):
        weighted_actions += aggregation_agent.q_table[i, action]

    # Wenn alle gewichteten Aktionen gleich sind, wählen Sie zufällig
    if np.all(weighted_actions == weighted_actions[0]):
        return np.random.choice(len(weighted_actions))
    else:
        return np.argmax(weighted_actions)
  


"""
#Aggregationfunction

def state_to_index(state, max_value=99):
    # Angenommen, jeder Wert im Zustand ist bereits ein diskretisierter Wert
    # Diese Funktion kombiniert die Werte zu einem Index, der innerhalb der Größe der Q-Tabelle liegt
    # max_value ist der höchste Wert, den ein diskretisierter Zustand annehmen kann
    index = 0
    for i, value in enumerate(state):
        index += value * (max_value + 1) ** i
    return index % 100  # Begrenzung des Index auf die Größe der Q-Tabelle

def aggregate_q_values(agents, states, agent_types):
    combined_q_values = np.zeros(3)  # Für 3 Aktionen
    for agent, agent_type in zip(agents, agent_types):
        if agent_type == 'rsi':
            rsi_value = states[agent_type][0]  # Annahme: RSI-Wert ist der erste Eintrag in der Liste
            action = agent.rsi_decision(rsi_value)
            if action != 0:
                return action
        else:
            state = states[agent_type]
            state_index = state_to_index(state)  # Konvertieren Sie den Zustand in einen Index
            q_values = agent.q_table[state_index] * agent.weight
            combined_q_values += q_values
    return np.argmax(combined_q_values)
"""

