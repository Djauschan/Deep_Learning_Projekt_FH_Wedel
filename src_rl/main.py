# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from utils.aggregationfunction import aggregate_actions
from environments.TraidingEnvironment_q_learning import TradingEnvironment
from utils.read_config import Config_reader

class Preprocess:
    #RSI---------------------------------------------------------------------
    def calculate_RSI(data, period=14):
        if len(data) < period:
            return 50  # Neutraler RSI-Wert bei unzureichenden Daten

        delta = data['close'].diff()
        gain = (delta.clip(lower=0)).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()

        RS = gain.fillna(0) / loss.fillna(0)
        RSI = 100 - (100 / (1 + RS))
        return RSI.iloc[-1]  # Rückgabe des letzten RSI-Wertes

    def determine_action_based_on_RSI(RSI, low_threshold=30, high_threshold=70):
        if RSI < low_threshold:
            return 1
        elif RSI > high_threshold:
            return 2
        else:
            return 0
        


class Portfolio:
    def __init__(self, cash, stocks):
        self.cash = cash
        self.stocks = stocks
        self.last_action = None

    def calculate_portfolio_change(self, action, current_price):
        if action == 1 and self.cash >= current_price and self.last_action != 'buy':  # Kaufen
            self.stocks += 1
            self.cash -= current_price
            self.last_action = 'buy'
        elif action == 2 and self.stocks > 0 and self.last_action != 'sell':  # Verkaufen
            self.stocks -= 1
            self.cash += current_price
            self.last_action = 'sell'
        return self.cash + self.stocks * current_price

def main():
    # Initialwert des Portfolios
    initial_portfolio_value = 10000

    portfolio = Portfolio(initial_portfolio_value, 0)  # Start mit Bargeld und 0 Aktien

    # Konfigurationsdatei lesen
    config = Config_reader('config/config.yml')

    # Laden der trainierten Modelle
    ma5_agent = pickle.load(open(config.get_parameter('ma5', 'q_models'), 'rb'))
    ma30_agent = pickle.load(open(config.get_parameter('ma30', 'q_models'), 'rb'))
    ma200_agent = pickle.load(open(config.get_parameter('ma200', 'q_models'), 'rb'))
    aggregation_agent = pickle.load(open(config.get_parameter('aggregation', 'q_models'), 'rb'))

    # Testparameter
    test_data_path = config.get_parameter('test_data', 'directories')
    test_data = pd.read_csv(test_data_path)
    env = TradingEnvironment(test_data)

    # Listen für Testergebnisse
    portfolio_values = []
    actions = []
    individual_agent_actions = {agent_type: [] for agent_type in ['ma5', 'ma30', 'ma200', 'rsi']}

    # Testdurchlauf
    NUM_EPISODES = 1
    for episode in tqdm(range(NUM_EPISODES), desc="Testdurchlauf"):
        state = env.reset()
        done = False

        while not done:
            individual_actions = [
                ma5_agent.act(state['ma5']),
                ma30_agent.act(state['ma30']),
                ma200_agent.act(state['ma200'])
            ]

            # RSI-basierte Aktion hinzufügen----------------------------------------------------
            rsi_value = Preprocess.calculate_RSI(test_data.iloc[:env.current_step + 1])
            rsi_action = Preprocess.determine_action_based_on_RSI(rsi_value)
            individual_actions.append(rsi_action)  # Füge RSI-basierte Aktion hinzu
            #-----------------------------------------------------------------------------------

            proposed_action = aggregate_actions(aggregation_agent, individual_actions)

            current_price = test_data['close'].iloc[env.current_step]
            #previous_price = test_data['close'].iloc[env.current_step - 1] if env.current_step > 0 else current_price
            
            # Überprüfen, ob die vorgeschlagene Aktion durchführbar ist
            if (proposed_action == 1 and portfolio.last_action == 'buy') or \
               (proposed_action == 2 and portfolio.last_action == 'sell'):
                final_action = 0  # Halten, wenn die letzte Aktion gleich war
            else:
                final_action = proposed_action

            actions.append(final_action)

            # Anpassen des Portfolio-Wertes basierend auf der Aktion
            portfolio_value = portfolio.calculate_portfolio_change(final_action, current_price)
            portfolio_values.append(portfolio_value)

            # Aktualisieren der Schritte und Überprüfung der Beendigungsbedingung
            env.current_step += 1
            if env.current_step >= len(test_data) - 1:
                done = True

            # Aktualisieren der individual_agent_actions
            for i, agent_type in enumerate(['ma5', 'ma30', 'ma200']): #, 'rsi'
                individual_agent_actions[agent_type].append(individual_actions[i])

    # Visualisierung
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Aktienchart und MAs zeichnen
    ax1.plot(test_data['close'], label='Schließkurse', color='blue')
    if 'ma5' in test_data.columns:
        ax1.plot(test_data['ma5'], label='MA 5 Tage', color='orange')
    if 'ma30' in test_data.columns:
        ax1.plot(test_data['ma30'], label='MA 30 Tage', color='green')
    if 'ma200' in test_data.columns:
        ax1.plot(test_data['ma200'], label='MA 200 Tage', color='red')

    # Kauf- und Verkaufsorders markieren
    for i, action in enumerate(actions):
        if action == 1:  # Kauf
            ax1.scatter(i, test_data['close'].iloc[i], color='green', marker='^')
        elif action == 2:  # Verkauf
            ax1.scatter(i, test_data['close'].iloc[i], color='red', marker='v')

    ax1.set_title('Aktienchart mit gleitenden Durchschnitten und Orders')
    ax1.set_xlabel('Zeit')
    ax1.set_ylabel('Preis')
    ax1.legend()

    # Wertentwicklung des Portfolios zeichnen
    ax2.plot(portfolio_values, label='Portfolio-Wert', color='purple')
    ax2.set_title('Wertentwicklung des Portfolios')
    ax2.set_xlabel('Zeit')
    ax2.set_ylabel('Wert')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
