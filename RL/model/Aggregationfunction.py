import numpy as np

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
    combined_q_values = np.zeros(2)  # Für 2 Aktionen
    for agent, agent_type in zip(agents, agent_types):
        state = states[agent_type]
        state_index = state_to_index(state)  # Konvertieren Sie den Zustand in einen Index
        weight = 15 if agent_type == 'rsi' else agent.weight  # Standardgewichtung für RSI
        q_values = agent.q_table[state_index] * weight
        combined_q_values += q_values
    return np.argmax(combined_q_values)
