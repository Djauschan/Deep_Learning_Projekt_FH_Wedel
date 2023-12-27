
# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from utils.aggregationfunction import aggregate_q_values 
from agents.q_learning import QLearningAgent
import matplotlib.pyplot as plt
import sys
from environments.TraidingEnvironment import TradingEnvironment
from utils.read_config import Config_reader
    
def main():
    # Read config file
    config = Config_reader('config/config.yml')
    
    # Pfadangaben
    train_data_path = config.get_parameter('train_data', 'directories')

    # Einlesen der Daten
    train_data = pd.read_csv(train_data_path)
    env = TradingEnvironment(train_data)  # Erzeugt ein TradingEnvironment-Objekt mit den Trainingsdaten

    # Erstellen der Q-Learning-Agenten für verschiedene Indikatoren
    ma5_agent = QLearningAgent('ma5', state_size=100, action_size=3)
    ma30_agent = QLearningAgent('ma30', state_size=100, action_size=3)
    ma200_agent = QLearningAgent('ma200', state_size=100, action_size=3)
    rsi_agent = QLearningAgent('rsi', state_size=100, action_size=3)  # Q-Tabelle nicht erforderlich

    ql_agents = [ma5_agent, ma30_agent, ma200_agent, rsi_agent]  # Liste aller Agenten
    agent_types = ['ma5', 'ma30', 'ma200', 'rsi']  # Bezeichnungen der Agententypen
    performance_metrics = {agent_type: [] for agent_type in agent_types}  # Initialisiert Leistungsmetriken

    # Trainingsprozess der Agenten
    NUM_EPISODES = config.get_parameter('epochs', 'train_parameters')  # Anzahl der Trainingsepisoden
    cumulative_rewards = {agent_type: np.zeros(NUM_EPISODES) for agent_type in agent_types}  # Kumulative Belohnungen

    for episode in tqdm(range(NUM_EPISODES)):  # Trainingsschleife
        states = env.reset()  # Zustand der Umgebung zurücksetzen
        done = False
        episode_rewards = {agent_type: 0 for agent_type in agent_types}  # Initialisiert Belohnungen für die Episode

        while not done:
            final_action = aggregate_q_values(ql_agents, states, agent_types)  # Bestimmt die endgültige Aktion basierend auf aggregierten Q-Werten
            states, reward, done = env.step(final_action)  # Führt die Aktion aus und erhält Belohnungen und nächsten Zustand

            if not done:
                for agent, agent_type in zip(ql_agents, agent_types):
                    state = states[agent_type]
                    agent.learn(state, final_action, reward, state, done)  # Trainiert jeden Agenten
                    episode_rewards[agent_type] += reward  # Summiert die Belohnungen

        if env.done:
            episode_performance = env.messen_der_leistung()  # Misst die Leistung der Episode
            for agent_type in agent_types:
                performance_metrics[agent_type].append(episode_performance)  # Speichert die Leistung
                cumulative_rewards[agent_type][episode] = episode_rewards[agent_type]  # Speichert kumulative Belohnungen

        # Ausgabe der Q-Tabellen für Überwachungszwecke
        print(f"Episode {episode + 1}/{NUM_EPISODES}")
        for agent in ql_agents:
            print(f"Q-Tabelle für {agent.agent_type}:")
            print(agent.q_table)

    # Anpassung der Gewichtung basierend auf der Leistung
    for agent_type in agent_types:
        agent_performance = performance_metrics[agent_type][-1]
        for agent in ql_agents:
            if agent.agent_type == agent_type:
                agent.weight = max(min(agent.weight + agent_performance * 0.01, 1), 0)


    # Speichern der trainierten Modelle
    for agent, agent_type in zip(ql_agents, agent_types):
        model_file_path = f'models\\{agent_type}_agent_model.pkl'
        with open(model_file_path, 'wb') as file:
            pickle.dump(agent, file)

    print("Modelle wurden erfolgreich gespeichert.")

    # Visualisierung der Metriken
    for agent_type in agent_types:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(cumulative_rewards[agent_type])
        plt.title(f"Kumulative Belohnungen pro Episode für {agent_type}")
        plt.xlabel("Episode")
        plt.ylabel("Kumulative Belohnung")

        plt.show()

if __name__ == '__main__':
    main()