import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from environments.TraidingEnvironment_q_learning import TradingEnvironment
from utils.read_config import Config_reader
from agents.q_learning import QLearningAgent
from utils.aggregationfunction import aggregate_actions
from main import Preprocess


def main():
 
    # Konfigurationsdatei und Daten einlesen
    config = Config_reader('config/config.yml')
    train_data_path = config.get_parameter('train_data', 'directories')
    train_data = pd.read_csv(train_data_path)
    env = TradingEnvironment(train_data)

    # Initialisierung der Q-Learning-Agenten und des Aggregationsagenten
    state_size = config.get_parameter('state_size', 'train_parameters')
    aggregation_state_size = config.get_parameter('aggregation_state_size', 'aggregation')
    agent_types = config.get_parameter('agent_types')
    ql_agents = [QLearningAgent(agent_type, state_size, 'config/config.yml') for agent_type in agent_types]
    aggregation_agent = QLearningAgent('aggregation', aggregation_state_size, 'config/config.yml')

    # Trainingsparameter
    NUM_EPISODES = config.get_parameter('epochs', 'train_parameters')

    # Initialisierung der Sammelvariablen für individuelle Agenten und Aggregationsagent
    cumulative_rewards = {agent_type: np.zeros(NUM_EPISODES) for agent_type in agent_types}
    cumulative_rewards['aggregation'] = np.zeros(NUM_EPISODES)  # Für den Aggregationsagenten
    portfolio_values = {agent_type: [[] for _ in range(NUM_EPISODES)] for agent_type in agent_types}
    portfolio_values['aggregation'] = [[] for _ in range(NUM_EPISODES)]  # Für den Aggregationsagenten

    # Trainingsprozess für individuelle Agenten
    for agent, agent_type in zip(ql_agents, agent_types):
        train_individual_agent(agent, agent_type, NUM_EPISODES, env, config, cumulative_rewards, portfolio_values)

    # Trainingsprozess für den Aggregationsagenten
    train_aggregation_agent(train_data, aggregation_agent, ql_agents, agent_types, NUM_EPISODES, env, config, cumulative_rewards, portfolio_values)

    # Speichern der Modelle
    save_models(ql_agents, agent_types, aggregation_agent)

    # Visualisierung der Performance
    visualize_performance(agent_types, cumulative_rewards, portfolio_values, config)

def train_individual_agent(agent, agent_type, num_episodes, env, config, cumulative_rewards, portfolio_values):
    for episode in tqdm(range(num_episodes), desc=f"Training {agent_type}"):
        states = env.reset()
        done = False
        episode_rewards = 0
        while not done:
            state = states[agent_type]
            action = agent.act(state)
            next_states, reward, done = env.step(action, agent_type)
            if not done:
                next_state = next_states[agent_type]
                agent.learn(state, action, reward, next_state, done)
            episode_rewards += reward
            states = next_states
        cumulative_rewards[agent_type][episode] = episode_rewards
        portfolio_values[agent_type][episode].append(env.calculate_portfolio_value(agent_type))

def get_actions_from_other_agents():
    return 0

def train_aggregation_agent(price_train_data, aggregation_agent, ql_agents, agent_types, num_episodes, env, config, cumulative_rewards, portfolio_values):
    for episode in tqdm(range(num_episodes), desc="Training Aggregation Agent"):
        states = env.reset()
        done = False
        aggregation_state = [0] * config.get_parameter('aggregation_state_size', 'aggregation')
        episode_rewards = 0

        while not done:
            individual_actions = [agent.act(states[agent_type]) for agent, agent_type in zip(ql_agents, agent_types)]

            # Berechnung des RSI für den aktuellen Zeitpunkt
            current_step = env.current_step
            current_data = price_train_data.iloc[:current_step + 1]  # Daten bis zum aktuellen Schritt
            RSI = Preprocess.calculate_RSI(current_data)
            rsi_action = Preprocess.determine_action_based_on_RSI(RSI)
        

            # Hinzufügen der RSI-Aktion zu den Agentenaktionen
            all_actions = individual_actions + [rsi_action]

            aggregated_action = aggregate_actions(aggregation_agent, all_actions)
            _, aggregation_reward, done = env.step(aggregated_action, 'aggregation', calculate_value=True)
            new_aggregation_state = update_aggregation_state(aggregation_state, individual_actions, rsi_action)
            aggregation_agent.learn(aggregation_state, aggregated_action, aggregation_reward, new_aggregation_state, done)
            episode_rewards += aggregation_reward
            aggregation_state = new_aggregation_state

        # Aktualisieren der Sammelvariablen für den Aggregationsagenten
        cumulative_rewards['aggregation'][episode] = episode_rewards
        portfolio_values['aggregation'][episode].append(env.calculate_portfolio_value('aggregation'))


def save_models(ql_agents, agent_types, aggregation_agent):
    for agent, agent_type in zip(ql_agents, agent_types):
        model_file_path = f'models/{agent_type}_agent_model.pkl'
        with open(model_file_path, 'wb') as file:
            pickle.dump(agent, file)
    with open('models/aggregation_agent_model.pkl', 'wb') as file:
        pickle.dump(aggregation_agent, file)

def visualize_performance(agent_types, cumulative_rewards, portfolio_values, config):
    NUM_EPISODES = config.get_parameter('epochs', 'train_parameters')

    # Visualisierung für individuelle Agenten
    for agent_type in agent_types:
        plt.figure(figsize=(12, 5))
        plt.plot([pv[-1] for pv in portfolio_values[agent_type] if pv], label="Portfolio-Werte")
        cumulative_average_end_values = np.cumsum([pv[-1] for pv in portfolio_values[agent_type] if pv]) / np.arange(1, len(portfolio_values[agent_type]) + 1)
        plt.plot(cumulative_average_end_values, label="Iterativer Durchschnitt der Endwerte", color='g')
        plt.axhline(y=config.get_parameter('start_cash_monitoring', 'train_parameters'), color='r', linestyle='--', label="Anfangswert des Portfolios")
        plt.title(f"Performance pro Episode für {agent_type}")
        plt.xlabel("Episode")
        plt.ylabel("Wert")
        plt.legend()
        plt.show()

    # Visualisierung für die Aggregationsfunktion
    plt.figure(figsize=(12, 5))
    plt.plot([pv[-1] for pv in portfolio_values['aggregation'] if pv], label="Portfolio-Werte (Aggregation)")
    cumulative_average_end_values = np.cumsum([pv[-1] for pv in portfolio_values['aggregation'] if pv]) / np.arange(1, len(portfolio_values['aggregation']) + 1)
    plt.plot(cumulative_average_end_values, label="Iterativer Durchschnitt der Endwerte (Aggregation)", color='g')
    plt.axhline(y=config.get_parameter('start_cash_monitoring', 'train_parameters'), color='r', linestyle='--', label="Anfangswert des Portfolios")
    plt.title("Performance pro Episode (Aggregationsagent)")
    plt.xlabel("Episode")
    plt.ylabel("Wert")
    plt.legend()
    plt.show()

"""
def update_aggregation_state(current_state, agent_actions, rsi_action):
    print("aktuell", current_state)
    new_state = list(current_state)
    # Aktualisieren mit Agentenaktionen
    for i, action in enumerate(agent_actions):
        new_state[i] = action
    
    # Hinzufügen der der Aktionen weiterer Agenten--------------------------------------------------------
    #RSI-Aktion
    new_state.append(rsi_action)
    return new_state
"""
def update_aggregation_state(current_state, agent_actions, rsi_action):
    # Erstellen eines neuen Zustands mit der aktuellen Aktion jedes Agenten und der RSI-Aktion
    new_state = agent_actions.copy() # Kopiert die aktuellen Aktionen der Agenten
    new_state.append(rsi_action)     # Fügt die RSI-Aktion hinzu
    return new_state


"""
def update_aggregation_state(current_state, actions):
    new_state = list(current_state)
    for i, action in enumerate(actions):
        new_state[i] = action
    return new_state
"""

if __name__ == "__main__":
    main()

