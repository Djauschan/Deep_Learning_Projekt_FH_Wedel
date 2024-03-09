Preprocessing

# MA: Berechnen der gleitenden Durchschnitte der letzten x-Tage   

    #Mittelwert der Preise von t-4 bis t ergeben den MA 5 in t 
    #Mittelwert der Preise von t-29 bis t ergeben den MA 30 in t 
    #Mittelwert der Preise von t-199 bis t ergeben den MA 200 in t 

    Bsp:
    def calculate_MA(data, period): #Mit period = 5, 30 oder 200
    if len(data) < period:
        # Nicht genügend Daten für den vollständigen MA
        return data.mean()
    else:
        # Berechnen des Durchschnitts der letzten 'period' Werte
        return data.iloc[-period:].mean()

    # Berechnung des MA5
    ma5 = calculate_MA(daily_data['close'], 5)




# Berechnung RSI 
def calculate_RSI(data, period=14):
    if len(data) < period:
        return 50  # Neutraler RSI-Wert bei unzureichenden Daten

    # Berechnung der Differenzen
    delta = data['close'].diff()

    # Ermittlung von Gewinnen und Verlusten über die letzten 'period' Tage
    gain = delta.clip(lower=0).iloc[-period:].mean()
    loss = -delta.clip(upper=0).iloc[-period:].mean()

    # Vermeidung der Division durch Null
    if loss == 0:
        return 100 if gain > 0 else 50

    # Berechnung des RSI
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def determine_action_based_on_RSI(RSI, low_threshold=30, high_threshold=70):
    if RSI < low_threshold:
        return 1  # Kaufen
    elif RSI > high_threshold:
        return 2  # Verkaufen
    else:
        return 0  # Halten







Für die Prediction der MA Werte ist eine Diskretisierungsfunktion erforderlich erforderlich

    # Laden der trainierten Modelle (Name aus config.yaml)
    ma5_agent = QLearningAgent('ma5', len(np.load(config.get_parameter('ma5', 'q_models'))[0]), 'config/config.yml')
    ma30_agent = QLearningAgent('ma30', len(np.load(config.get_parameter('ma30', 'q_models'))[0]), 'config/config.yml')
    ma200_agent = QLearningAgent('ma200', len(np.load(config.get_parameter('ma200', 'q_models'))[0]), 'config/config.yml')


    #State-Berechnungs-Funktion
    def calculate_state(current_price, ma_value):
        deviation = (current_price - ma_value) / ma_value * 100
        max_deviation = 2 * deviation.std() 
        scaled_deviation = (deviation + max_deviation) / (2 * max_deviation)
        state = max(0, min(int(scaled_deviation * 20), 20 - 1)) # n_bins = 20
        return state


    def make_prediction(current_price,ma_value, agent):
        state_index = calculate_state(current_price, ma_value)
        action = np.argmax(agent.q_table[state_index])
        return action


    # Beispiel der Verwendung
    current_price = ... # Aktueller Preis
    ma5_value = ...     # aktueller MA5-Wert
    ma5_agent = ...     # MA5-Agent

    prediction_ma5 = make_prediction(current_price, ma5_value, ma5_agent)


Achtung schlechtere Ergebnisse!!!!
#für RF, GBM und Transformer:
    def action_from_q_table(prediction_des_Modells, last_price, q_table_des_Modells):
        # Bestimmen des Zustands basierend auf der Preisveränderung
        price_diff = prediction_des_Modells - last_price
        state = min(int(abs(price_diff) / last_price * 10), 9)
        # Aktion mit dem höchsten Q-Wert für diesen Zustand
        action = np.argmax(q_table_des_Modells[state])
        return action

Besser wenn Modelle direkt Predictions in Aggregationsfunktion einfliessen lassen:

                if prediction > last_price:
                    model_action = 1  # Kaufen, wenn die Vorhersage für t+1 höher ist als der Preis bei t
                elif prediction < last_price:
                    model_action = 2  # Verkaufen, wenn die Vorhersage für t+1 niedriger ist als der Preis bei t
                else:
                    model_action = 0 



#Aggregationsmodell: 
"""
Action jeweils 0 (nichts tun),1 (kaufen),2 (verkaufen)
Reihenfolge der actions [MA5_Action, MA30_Action, MA200_Action, RSI_Action]
"""

aggregation_agent = QLearningAgent('aggregation', len(np.load(config.get_parameter('aggregation', 'q_models'))[0]), 'config/config.yml')

def aggregate_actions(aggregation_agent, actions):
    weighted_actions = np.zeros(3)  # Angenommen, es gibt 3 mögliche Aktionen (kaufen, nichts tun, und verkaufen)
    
    for i, action in enumerate(actions):
        weighted_actions += aggregation_agent.q_table[i, action]

    # Wenn alle gewichteten Aktionen gleich sind, wähle zufällig
    if np.all(weighted_actions == weighted_actions[0]):
        return np.random.choice(len(weighted_actions))
    else:
        return np.argmax(weighted_actions)


Beispiel:
proposed_action = aggregate_actions(aggregation_agent, [0,1,2,0,1,2,1])