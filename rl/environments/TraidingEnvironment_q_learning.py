import numpy as np
from utils.read_config import Config_reader
import pdb

class TradingEnvironment:
    def __init__(self, data, config_path='config/config.yml'):
        if data.empty:
            raise ValueError("The provided data is empty!")
        self.config = Config_reader(config_path)
        self.data = data
        self.current_step = 0
        self.done = False
        self.std_devs = self.calculate_std_deviation(self.config.get_parameter('agent_types'))
        self.stock_price = self.data['close'].iloc[0] #erster close Wert

        # Test Wertentwicklung für Individuelle Agenten:
        self.agent_portfolios = {agent_type: {'cash': self.config.get_parameter('start_cash_monitoring', 'train_parameters'), 
                                              'stocks': self.config.get_parameter('start_stock_monitoring', 'train_parameters')} 
                                 for agent_type in self.config.get_parameter('agent_types')}
        
        # Portfolio für den Aggregationsagenten hinzufügen
        self.agent_portfolios['aggregation'] = {
            'cash': self.config.get_parameter('start_cash_monitoring', 'train_parameters'),
            'stocks': self.config.get_parameter('start_stock_monitoring', 'train_parameters')
        }

        self.reward_params = {
            'reward': self.config.get_parameter('reward', 'train_parameters'),
            'hold_reward': self.config.get_parameter('hold_reward', 'train_parameters'),
            'penalty': self.config.get_parameter('penalty', 'train_parameters')
        }

        # Letzter Portfolio-Wert für jede Episode
        self.last_portfolio_values = {agent_type: self.calculate_portfolio_value(agent_type) 
                                      for agent_type in self.config.get_parameter('agent_types')}
        # Letzter Portfolio-Wert für den Aggregationsagenten
        self.last_portfolio_values['aggregation'] = self.calculate_portfolio_value('aggregation')

    """
    #Das Training/Prognose auf eine neue Datei -> keine Prognose)
    def load_new_data(self, new_data):
        if new_data.empty:
            raise ValueError("The provided data is empty!")
        self.data = new_data
        self.std_devs = self.calculate_std_deviation(self.config.get_parameter('agent_types'))
        self.reset()  # Reset the environment with the new data
    """
    def load_new_data(self, new_data):
        if new_data.empty:
            raise ValueError("The provided data is empty!")
        self.data = new_data
        self.std_devs = self.calculate_std_deviation(self.config.get_parameter('agent_types'))
        # Ruft reset auf, aber behält dabei die aktuellen 'stocks'-Werte bei
        self.reset(retain_stocks=True)


    # Anpassung-> Standardabweichung beschränkt sich nur auf die letzten 220 Werten (Aufgrund verfügbarer Vorlauf in den DAten für Prediction)
    def calculate_std_deviation(self, ma_columns):
        std_devs = {}
        for ma_column in ma_columns:
            # Beschränkung auf die letzten 220 Werte
            recent_prices = self.data['close'].tail(220)
            recent_ma_values = self.data[ma_column].tail(220)
            
            # Berechnung der Abweichung auf den letzten 220 Werten
            deviation = (recent_prices - recent_ma_values) / recent_ma_values * 100
            std_devs[ma_column] = deviation.std()
        return std_devs

    def discretize_deviation(self, price, ma_value, ma_column, n_bins):
        std_dev = self.std_devs[ma_column]
        max_deviation = 2 * std_dev
        if ma_value == 0:
            ma_value = 0.001
        deviation = (price - ma_value) / ma_value * 100
        scaled_deviation = (deviation + max_deviation) / (2 * max_deviation)
        return max(0, min(int(scaled_deviation * n_bins), n_bins - 1))

    def reset(self, retain_stocks=False):
        self.current_step = 0
        self.done = False
        if not retain_stocks:
            # Zurücksetzen von 'cash' und 'stocks' zu den Anfangswerten, falls nicht beibehalten werden soll
            for agent_type in self.agent_portfolios:
                self.agent_portfolios[agent_type]['cash'] = self.config.get_parameter('start_cash_monitoring', 'train_parameters')
                self.agent_portfolios[agent_type]['stocks'] = self.config.get_parameter('start_stock_monitoring', 'train_parameters')
        # Immer den letzten Portfolio-Wert aktualisieren, unabhängig davon, ob 'stocks' beibehalten werden
        for agent_type in self.agent_portfolios:
            self.last_portfolio_values[agent_type] = self.calculate_portfolio_value(agent_type)

        return self.get_state()



    def step(self, action, agent_type, calculate_value=True):
            # Überprüfen, ob das Ende der Daten erreicht ist
        if self.current_step >= len(self.data) - 1:
            self.done = True
            # Da keine weiteren Daten vorhanden sind, kannst du neutrale Zustände zurückgeben
            neutral_states = {agent_type: 0 for agent_type in self.config.get_parameter('agent_types')}
            return neutral_states, 0, self.done

        reward = 0
        self.stock_price = current_price = self.data['close'].iloc[self.current_step]
        next_price = self.data['close'].iloc[self.current_step + 1] if self.current_step < len(self.data) - 1 else current_price

        if calculate_value:
            agent_cash = self.agent_portfolios[agent_type]['cash']
            agent_stocks = self.agent_portfolios[agent_type]['stocks']

        if action == 1:  # Kaufen
            if next_price > current_price:
                reward = self.reward_params['reward']
            elif next_price < current_price:
                reward = self.reward_params['penalty']
            elif next_price == current_price:
                reward = self.reward_params['hold_reward']

            if calculate_value:
                num_stocks_to_buy = int(agent_cash / current_price)
                if num_stocks_to_buy > 0:
                    self.agent_portfolios[agent_type]['stocks'] += num_stocks_to_buy
                    self.agent_portfolios[agent_type]['cash'] -= num_stocks_to_buy * current_price

        elif action == 2:  # Verkaufen
            if next_price > current_price:
                reward = self.reward_params['penalty']
            elif next_price < current_price:
                reward = self.reward_params['reward']
            elif next_price == current_price:
                reward = self.reward_params['penalty']

            if calculate_value:
                if agent_stocks > 0:
                    self.agent_portfolios[agent_type]['cash'] += agent_stocks * current_price
                    self.agent_portfolios[agent_type]['stocks'] = 0

        elif action == 0:  # Nichts tun
            if next_price > current_price:
                reward = self.reward_params['penalty']
            elif next_price < current_price:
                reward = self.reward_params['penalty']
            elif next_price == current_price:
                reward = self.reward_params['hold_reward']

        else:
            raise ValueError(f"The provided action ({action}) is invalid!")

        # Weitere Belohnung-/Bestrafungslogik
        new_portfolio_value = self.calculate_portfolio_value(agent_type) if calculate_value else 0
        change_percentage = ((new_portfolio_value - self.last_portfolio_values[agent_type]) / self.last_portfolio_values[agent_type]) * 100 if self.last_portfolio_values[agent_type] != 0 else 0

        # Überprüfen der Wertänderung
        if change_percentage > 10:
            reward += self.config.get_parameter('threshold_reward10', 'train_parameters')
        elif change_percentage < -10:
            reward -= self.config.get_parameter('threshold_penalty10', 'train_parameters')
        elif 10 > change_percentage > 5:
            reward += self.config.get_parameter('threshold_reward5', 'train_parameters')
        elif -10 < change_percentage < -5:
            reward -= self.config.get_parameter('threshold_penalty5', 'train_parameters')
        elif 5 > change_percentage > 2:
            reward += self.config.get_parameter('threshold_reward2', 'train_parameters')
        elif -5 < change_percentage < -2:
            reward -= self.config.get_parameter('threshold_penalty2', 'train_parameters')
        elif 2 > change_percentage > 1:
            reward += self.config.get_parameter('threshold_reward1', 'train_parameters')
        elif -2 < change_percentage < -1:
            reward -= self.config.get_parameter('threshold_penalty1', 'train_parameters')

        # Aktualisieren des letzten Portfolio-Wertes
        if calculate_value:
            self.last_portfolio_values[agent_type] = new_portfolio_value

        self.current_step += 1
        self.done = self.current_step >= len(self.data)
        next_states = self.get_state()

        return next_states, reward, self.done

    def calculate_portfolio_value(self, agent_type):
        portfolio = self.agent_portfolios[agent_type]
        return portfolio['cash'] + portfolio['stocks'] * self.stock_price

    def get_state(self):
        current_price = self.data['close'].iloc[self.current_step]
        state = {}
        for ma_column in self.config.get_parameter('agent_types'):
            ma_value = self.data[ma_column].iloc[self.current_step]
            state[ma_column] = self.discretize_deviation(current_price, ma_value, ma_column, self.config.get_parameter('state_size', 'train_parameters'))
        return state

