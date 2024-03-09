import numpy as np

class TradingEnvironment:
    def __init__(self, data, reward_params=None):
        if data.empty:
            raise ValueError("The provided data is empty!")
        self.data = data
        self.current_step = 0
        self.done = False
        self.balance = 100
        self.stock_owned = 0
        self.initial_portfolio_value = self.get_portfolio_value()
        self.std_devs = self.calculate_std_deviation(['ma5', 'ma30', 'ma200', 'rsi'])

        # Angepasstes Belohnungsschema
        if reward_params is None:
            reward_params = {
                'buy_reward': 5,
                'sell_reward': 5,
                'hold_reward': 1,
                'buy_penalty': -1,
                'sell_penalty': -1,
                'invalid_action_penalty': -5
            }
        self.reward_params = reward_params

    def calculate_std_deviation(self, ma_columns):
        std_devs = {}
        for ma_column in ma_columns:
            deviation = (self.data['close'] - self.data[ma_column]) / self.data[ma_column] * 100
            std_devs[ma_column] = deviation.std()
        return std_devs

    def discretize_deviation(self, price, ma_value, ma_column, n_bins = 20):
        std_dev = self.std_devs[ma_column]
        max_deviation = 2 * std_dev  # ±2 Standardabweichungen

        deviation = (price - ma_value) / ma_value * 100
        scaled_deviation = (deviation + max_deviation) / (2 * max_deviation)
        
        return max(0, min(int(scaled_deviation * n_bins), n_bins - 1))

    def get_portfolio_value(self):
        current_price = self.data['close'].iloc[self.current_step]
        return self.balance + (self.stock_owned * current_price)

    def messen_der_leistung(self):
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1

        final_portfolio_value = self.get_portfolio_value()
        if self.initial_portfolio_value == 0:
            return 0

        performance = ((final_portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value) * 100
        return performance


    def reset(self):
        self.current_step = 0
        self.done = False
        self.balance = 100
        self.stock_owned = 0
        self.initial_portfolio_value = self.get_portfolio_value()
        return self.get_state()

    def step(self, action):
        current_price = self.data['close'].iloc[self.current_step]
        next_price = self.data['close'].iloc[self.current_step + 1] if self.current_step < len(self.data) - 1 else current_price

        if action == 1:  # Kaufen
            reward = self.reward_params['buy_reward'] * (next_price - current_price) / current_price
            """
            if self.balance >= current_price:
                self.stock_owned += 1
                self.balance -= current_price
                # Belohnung basierend auf Marktbewegungen
                reward = self.reward_params['buy_reward'] if next_price > current_price else self.reward_params['buy_penalty']
            else:
                reward = self.reward_params['invalid_action_penalty']
            """
        elif action == 2:  # Verkaufen
            reward = self.reward_params['sell_reward'] * (current_price - next_price) / current_price
            """
            if self.stock_owned > 0:
                self.stock_owned -= 1
                self.balance += current_price
                # Belohnung basierend auf Marktbewegungen
                reward = self.reward_params['sell_reward'] if next_price < current_price else self.reward_params['sell_penalty']
            else:
                reward = self.reward_params['invalid_action_penalty']
            """
        elif action == 0:  # Halten
            # TODO: Belohnung anpassen. Agent hält aktuell permanent, ohne zu traden. Das ist nicht sinnvoll.
            reward = 0
        else:
            raise ValueError(f"The provided action ({action}) is invalid!")

        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True
            return [], 0, True

        next_states = self.get_state()

        #Kontrolle
        # print(f"Schritt: {self.current_step}, Aktion: {action}, Belohnung: {reward}, Neuer Zustand: {next_states}")

        return next_states, reward, self.done

    def get_state(self):
        current_price = self.data['close'].iloc[self.current_step]
        state = {}
        for ma_column in ['ma5', 'ma30', 'ma200', 'rsi']:
            ma_value = self.data[ma_column].iloc[self.current_step]
            state[ma_column] = [self.discretize_deviation(current_price, ma_value, ma_column), self.discretize_deviation(current_price, current_price, ma_column)]
        return state
