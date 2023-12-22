
# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from Aggregationfunction import aggregate_q_values 
from Q_Learning import QLearningAgent
import sys
directory_path = r'C:\Users\Joel\Desktop\FH-Wedel\Deep Learning Projekt\Git\Deep_Learning-2\Simulation'
sys.path.append(directory_path)
from Traidingenvirement import TradingEnvironment


# Pfadangaben
directory_path = r'C:\Users\Joel\Desktop\FH-Wedel\Deep Learning Projekt\Git\Deep_Learning-2\Simulation'
train_data_path = r'C:\Users\Joel\Desktop\FH-Wedel\Deep Learning Projekt\Git\Deep_Learning-2\test_data.csv'

# Einlesen der Daten
train_data = pd.read_csv(train_data_path)
env = TradingEnvironment(train_data)

# Erstellen der Agenten
ma5_agent = QLearningAgent('ma5', state_size=10, action_size=2)
ma30_agent = QLearningAgent('ma30', state_size=10, action_size=2)
ma200_agent = QLearningAgent('ma200', state_size=10, action_size=2)
rsi_agent = QLearningAgent('rsi', state_size=10, action_size=2)

# Initialisierung des RSI-Agenten (Handcrafted Feature)
# 0-30 kaufen; 7-100 verkaufen
rsi_agent.q_table = np.array([[0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float64)
rsi_agent.exploration_rate = 0.0  # Keine Exploration

ql_agents = [ma5_agent, ma30_agent, ma200_agent, rsi_agent]
agent_types = ['ma5', 'ma30', 'ma200', 'rsi']
performance_metrics = {agent_type: [] for agent_type in agent_types}

def berechne_neue_gewichtung(leistungen):
    alpha = 0.05  # Glättungsfaktor
    gewichtetes_mittel = 0
    normierungsfaktor = 0

    for i, leistung in enumerate(reversed(leistungen)):
        gewicht = (1 - alpha) ** i
        gewichtetes_mittel += leistung * gewicht
        normierungsfaktor += gewicht

    if normierungsfaktor > 0:
        gewichtetes_mittel /= normierungsfaktor

    return 1.001 if gewichtung > 0 else 0.999

# Trainingsprozess
NUM_EPISODES = 100
for episode in tqdm(range(NUM_EPISODES)):
    states = env.reset()
    done = False

    while not done:
        final_action = aggregate_q_values(ql_agents, states, agent_types)
        next_states, reward, done = env.step(final_action)

        if not done:
            for agent, agent_type in zip(ql_agents, agent_types):
                state = states[agent_type]
                next_state = next_states[agent_type]
                agent.learn(state, final_action, reward, next_state, done)

    if env.done:
        episode_performance = env.messen_der_leistung()
        for agent_type in agent_types:
            performance_metrics[agent_type].append(episode_performance)

    #Q-Table überwachen
    print(f"Episode {episode + 1}/{NUM_EPISODES}")
    for agent in ql_agents:
        print(f"Q-Tabelle für {agent.agent_type}:")
        print(agent.q_table)

# Durchschnittliche Leistung und Anpassung der Gewichtungen
for agent_type, leistungen in performance_metrics.items():
    gewichtung = berechne_neue_gewichtung(leistungen)
    for agent in ql_agents:
        if agent.agent_type == agent_type:
            agent.weight = gewichtung

# Speichern der trainierten Modelle
for agent, agent_type in zip(ql_agents, agent_types):
    model_file_path = f'{directory_path}\\{agent_type}_agent_model.pkl'
    with open(model_file_path, 'wb') as file:
        pickle.dump(agent, file)

print("Modelle wurden erfolgreich gespeichert.")