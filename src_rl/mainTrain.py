
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from environments.TraidingEnvironment_q_learning import TradingEnvironment
from utils.read_config import Config_reader
from agents.q_learning import QLearningAgent
from utils.aggregationfunction import aggregate_actions
from main import Preprocess

def main():
    # Konfigurationsdatei einlesen
    config = Config_reader('config/config.yml')
    train_data_directory = config.get_parameter('train_data_directory', 'directories')
    
    # Finde alle "train_" Dateien im Verzeichnis
    train_files = [file for file in os.listdir(train_data_directory) if file.startswith("train_")]
    
    for file in train_files:
        train_data_path = os.path.join(train_data_directory, file)
        train_data = pd.read_csv(train_data_path)
        
        # Initialisiere die Umgebung und Agenten mit den neuen Daten
        env = TradingEnvironment(train_data)
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
    #for agent, agent_type in zip(ql_agents, agent_types):
    #    train_individual_agent(agent, agent_type, NUM_EPISODES, env, config, cumulative_rewards, portfolio_values)

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
            next_state = next_states[agent_type]
            if not done:
                next_state = next_states[agent_type]
                agent.learn([state], action, reward, next_state, True, done)
            episode_rewards += reward
            states = next_states
        cumulative_rewards[agent_type][episode] = episode_rewards
        portfolio_values[agent_type][episode].append(env.calculate_portfolio_value(agent_type))

"""
weitere Agenten
"""
# Weitere Modelle laden
def load_q_tables(config):
    q_tables = {}
    q_tables['RF'] = np.load(config.get_parameter('rf', 'q_models'))
    q_tables['GBM'] = np.load(config.get_parameter('gbm', 'q_models'))
    q_tables['transformer'] = np.load(config.get_parameter('transformer', 'q_models'))
    return q_tables

def action_from_q_table(prediction, last_price, q_table):
    # Bestimmen des Zustands basierend auf der Preisveränderung
    price_diff = prediction - last_price
    state = min(int(abs(price_diff) / last_price * 10), 9)  # Zustand basierend auf relativer Preisänderung
    # Aktion mit dem höchsten Q-Wert für diesen Zustand auswählen
    action = np.argmax(q_table[state])
    return action

"""
Aggregationsfunktion
"""

def train_aggregation_agent(price_train_data, aggregation_agent, ql_agents, agent_types, num_episodes, env, config, cumulative_rewards, portfolio_values):
    q_tables_models = load_q_tables(config)  # Laden der Q-Tabellen

    for episode in tqdm(range(num_episodes), desc="Training Aggregation Agent"):
        states = env.reset()
        done = False
        aggregation_state = [0] * config.get_parameter('aggregation_state_size', 'aggregation')
        episode_rewards = 0

        while not done:
            individual_actions = [agent.act(states[agent_type]) for agent, agent_type in zip(ql_agents, agent_types)]

            """
            # Berechnung der Aktionen basierend auf den Modellvorhersagen
            model_actions = []
            for model_name, q_table in q_tables_models.items():
                last_price = env.stock_price
                prediction = price_train_data[f'{model_name}_pred'].iloc[env.current_step]
                model_action = action_from_q_table(prediction, last_price, q_table)
                model_actions.append(model_action)
                print(model_name,prediction,price_train_data[f'close'].iloc[env.current_step],model_actions)
            """

            
            # Direkte Berechnung der Aktionen basierend auf den Modellvorhersagen für t+1
            model_actions = []
            for model_name in q_tables_models.keys():  # Verwendung von .keys() ist optional
                # Hinweis: "last_price" repräsentiert hier eigentlich den Preis zum aktuellen Zeitpunkt "t"
                last_price = env.stock_price
                # Die "prediction" ist die Vorhersage für den Preis zum Zeitpunkt "t+1"
                prediction = price_train_data[f'{model_name}_pred'].iloc[env.current_step]
                
                # Bestimmen der Aktion basierend auf dem Vergleich der Vorhersage für t+1 mit dem aktuellen Preis bei t
                if prediction > last_price:
                    model_action = 1  # Kaufen, wenn die Vorhersage für t+1 höher ist als der Preis bei t
                elif prediction < last_price:
                    model_action = 2  # Verkaufen, wenn die Vorhersage für t+1 niedriger ist als der Preis bei t
                else:
                    model_action = 0  # Halten, wenn die Vorhersage für t+1 dem Preis bei t entspricht
                
                model_actions.append(model_action)


            # Berechnung des RSI für den aktuellen Zeitpunkt
            current_step = env.current_step
            current_data = price_train_data.iloc[:current_step + 1]  # Daten bis zum aktuellen Schritt
            RSI = Preprocess.calculate_RSI(current_data)
            rsi_action = Preprocess.determine_action_based_on_RSI(RSI)

            # Hinzufügen der RSI-Aktion zu den Agentenaktionen
            all_actions = individual_actions + [rsi_action] + model_actions
            #print(all_actions)
            aggregated_action = aggregate_actions(aggregation_agent, all_actions)
            _, aggregation_reward, done = env.step(aggregated_action, 'aggregation', calculate_value=True)
            aggregation_agent.learn(aggregation_state, aggregated_action, aggregation_reward, all_actions, False, done)
            aggregation_agent.learn(aggregation_state, aggregated_action, aggregation_reward, all_actions, False, done)
            episode_rewards += aggregation_reward
            aggregation_state = all_actions

        # Aktualisieren der Sammelvariablen für den Aggregationsagenten
        cumulative_rewards['aggregation'][episode] = episode_rewards
        portfolio_values['aggregation'][episode].append(env.calculate_portfolio_value('aggregation'))

    # Speichern der Q-Tabelle des Aggregationsagenten am Ende des Trainings
    model_file_path = 'models/aggregation_agent_q_table.npy'
    np.save(model_file_path, aggregation_agent.q_table)
    print(f"Q-Tabelle des Aggregationsagenten gespeichert unter: {model_file_path}")


def save_models(ql_agents, agent_types, aggregation_agent):
    for agent, agent_type in zip(ql_agents, agent_types):
        model_file_path = f'models/{agent_type}_agent_q_table_1h.npy'
        np.save(model_file_path, agent.q_table)
    
    #np.save('models/aggregation_agent_q_table.npy', aggregation_agent.q_table)

def visualize_performance(agent_types, cumulative_rewards, portfolio_values, config):
    NUM_EPISODES = config.get_parameter('epochs', 'train_parameters')
    """
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
    """
    
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
    


if __name__ == "__main__":
    main()





