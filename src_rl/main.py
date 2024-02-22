# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from agents.q_learning import QLearningAgent
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
    def __init__(self, cash, stocks, enable_stop_loss = True):
        self.cash = cash
        self.stocks = stocks
        self.buy_stop_loss = 0
        self.sell_stop_loss = float('inf')
        self.enable_stop_loss = enable_stop_loss


    def set_buy_stop_loss(self, current_price, stopp_under_buy_price):
            self.buy_stop_loss = current_price * (1 - 0.01 * stopp_under_buy_price) 

    def set_sell_stop_loss(self, current_price, stopp_under_sell_price):
            self.sell_stop_loss = current_price * (1 - 0.01 * stopp_under_sell_price) 

    def check_stop_loss(self, current_price):
            return current_price <= self.buy_stop_loss or current_price <= self.sell_stop_loss

    def calculate_portfolio_change(self, action, current_price, stopp_under_buy_price, stopp_under_sell_price):
        stop_loss_triggered = False
        if action == 1 and self.cash >= current_price:  # Kaufen
            self.stocks += 1
            self.cash -= current_price
            self.set_buy_stop_loss(current_price, stopp_under_buy_price)
            stop_loss_triggered = 'buy'
        elif action == 2 and self.stocks > 0:  # Verkaufen
            self.stocks -= 1
            self.cash += current_price
            self.buy_stop_loss = 0
            self.sell_stop_loss = float('inf')
            if current_price <= self.sell_stop_loss:
                stop_loss_triggered = 'sell'

        return self.cash + self.stocks * current_price, stop_loss_triggered


def main():

    # Konfigurationsdatei lesen
    config = Config_reader('config/config.yml')
    enable_stop_loss = config.get_parameter('use_stopp_loss', 'stopp_loss')
    stopp_under_buy_price = config.get_parameter('buy_stop_loss_under_buy_price', 'stopp_loss') 
    stopp_under_sell_price = config.get_parameter('stopp_under_sell_price', 'stopp_loss') 

    # Initialwerte und Konfigurationen
    initial_portfolio_value = config.get_parameter('startvalue','traiding')

    # Start mit Bargeld und 0 Aktien
    portfolio = Portfolio(initial_portfolio_value, 0, enable_stop_loss)
  

    # Laden der trainierten Modelle
    # Erstellen von Agenteninstanzen
    ma5_agent = QLearningAgent('ma5', len(np.load(config.get_parameter('ma5', 'q_models'))[0]), 'config/config.yml')
    ma30_agent = QLearningAgent('ma30', len(np.load(config.get_parameter('ma30', 'q_models'))[0]), 'config/config.yml')
    ma200_agent = QLearningAgent('ma200', len(np.load(config.get_parameter('ma200', 'q_models'))[0]), 'config/config.yml')
    aggregation_agent = QLearningAgent('aggregation', len(np.load(config.get_parameter('aggregation', 'q_models'))[0]), 'config/config.yml')

    # Laden der Q-Tabellen in die Agenteninstanzen
    ma5_agent.q_table = np.load(config.get_parameter('ma5', 'q_models'))
    ma30_agent.q_table = np.load(config.get_parameter('ma30', 'q_models'))
    ma200_agent.q_table = np.load(config.get_parameter('ma200', 'q_models'))
    aggregation_agent.q_table = np.load(config.get_parameter('aggregation', 'q_models'))


    # Testparameter
    test_data_path = config.get_parameter('test_data', 'directories')
    test_data = pd.read_csv(test_data_path)
    env = TradingEnvironment(test_data)

    # Listen für Testergebnisse
    stop_losses = []
    portfolio_values = []
    actions = []
    #individual_agent_actions = {agent_type: [] for agent_type in ['ma5', 'ma30', 'ma200', 'rsi']}

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

            rsi_value = Preprocess.calculate_RSI(test_data.iloc[:env.current_step + 1])
            rsi_action = Preprocess.determine_action_based_on_RSI(rsi_value)
            individual_actions.append(rsi_action)

            proposed_action = aggregate_actions(aggregation_agent, individual_actions)
            current_price = test_data['close'].iloc[env.current_step]

            final_action = 0  # Standardaktion ist Halten

            # Wenn Stop-Loss aktiviert ist
            if enable_stop_loss:
                if proposed_action == 2:
                    portfolio.set_sell_stop_loss(current_price, stopp_under_sell_price)  # Setze neuen Stop-Loss für Verkauf
                if portfolio.check_stop_loss(current_price) and portfolio.stocks > 0:
                    final_action = 2  # Verkaufen, wenn Stop-Loss durchbrochen wird
                if proposed_action == 1 and portfolio.stocks == 0:
                    portfolio.set_buy_stop_loss(current_price, stopp_under_buy_price)
                    final_action = 1
            else:
                # Wenn Stop-Loss deaktiviert ist
                if proposed_action == 1 and portfolio.stocks == 0:
                    final_action = 1  # Kaufen, wenn vorgeschlagen und keine Aktien im Bestand
                elif proposed_action == 2 and portfolio.stocks > 0:
                    final_action = 2  # Verkaufen, wenn vorgeschlagen und Aktien vorhanden

            actions.append(final_action)
        

            portfolio_value, stop_loss_triggered = portfolio.calculate_portfolio_change(final_action, current_price, stopp_under_buy_price, stopp_under_sell_price)
            portfolio_values.append(portfolio_value)

            if stop_loss_triggered:
                stop_losses.append((env.current_step, current_price, stop_loss_triggered))
        

            env.current_step += 1
            if env.current_step >= len(test_data) - 1:
                done = True



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

    for step, price, stop_type in stop_losses:
        if stop_type == 'buy':
            ax1.plot([step-5, step+5], [price, price], color='red', linestyle='--', linewidth=2)  # Kurze horizontale Linie für Kauf-Stop-Loss
        elif stop_type == 'sell':
            ax1.plot([step-5, step+5], [price, price], color='blue', linestyle='--', linewidth=2)  # Kurze horizontale Linie für Verkauf-Stop-Loss



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
