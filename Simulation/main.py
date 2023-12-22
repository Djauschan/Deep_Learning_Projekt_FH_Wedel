#Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys
directory_path = r'C:\Users\Joel\Desktop\FH-Wedel\Deep Learning Projekt\Git\Deep_Learning-2\RL\model'
sys.path.append(directory_path)
from Aggregationfunction import state_to_index
from Aggregationfunction import aggregate_q_values
from Traidingenvirement import TradingEnvironment
from Q_Learning import QLearningAgent
from Traidingenvirement import TradingEnvironment  # Stellen Sie sicher, dass Sie die richtige Pfadangabe verwenden
import matplotlib.pyplot as plt  # Für die Darstellung der Ergebnisse


#trained Agents
# Laden der trainierten Modelle
ma5_agent = pickle.load(open(r'C:\Users\Joel\Desktop\FH-Wedel\Deep Learning Projekt\Git\Deep_Learning-2\Simulation\ma5_agent_model.pkl', 'rb'))
ma30_agent = pickle.load(open(r'C:\Users\Joel\Desktop\FH-Wedel\Deep Learning Projekt\Git\Deep_Learning-2\Simulation\ma30_agent_model.pkl', 'rb'))
ma200_agent = pickle.load(open(r'C:\Users\Joel\Desktop\FH-Wedel\Deep Learning Projekt\Git\Deep_Learning-2\Simulation\ma200_agent_model.pkl', 'rb'))
rsi_agent = pickle.load(open(r'C:\Users\Joel\Desktop\FH-Wedel\Deep Learning Projekt\Git\Deep_Learning-2\Simulation\rsi_agent_model.pkl', 'rb'))


# Agentenliste
ql_agents = [ma5_agent, ma30_agent, ma200_agent, rsi_agent]
agent_types = ['ma5', 'ma30', 'ma200', 'rsi']

#Testing
# Pfad zum Testdatensatz
test_data_path = r'C:\Users\Joel\Desktop\FH-Wedel\Deep Learning Projekt\Git\Deep_Learning-2\test_data.csv'
# Einlesen des Testdatensatzes
test_data = pd.read_csv(test_data_path)

# Erstellen der Handelsumgebung für den Testdatensatz
env = TradingEnvironment(test_data)

# Listen zur Aufzeichnung der Ergebnisse
portfolio_values = []
actions = []
individual_agent_actions = {agent_type: [] for agent_type in agent_types}  # Für individuelle Agentenaktionen

# Testparameter
NUM_EPISODES = 1

#Run Test
for episode in tqdm(range(NUM_EPISODES)):
    states = env.reset()
    done = False
    portfolio_value = []  # Portfolio-Werte für jede Episode

    for agent in ql_agents:
        print(agent.q_table.shape)  # Überprüfen Sie, ob dies (state_size, 2) für jeden Agenten anzeigt
    while not done:
        # Aktionen der Agenten auswählen
        individual_actions = [agent.act(states[agent.agent_type]) for agent in ql_agents]
        for agent_type, action in zip(agent_types, individual_actions):
            individual_agent_actions[agent_type].append(action)

        final_action = aggregate_q_values(ql_agents, states, agent_types)
        actions.append(final_action)  # Speichern der aggregierten Aktion

        # Aktionen durchführen und Belohnung erhalten
        next_states, reward, done = env.step(final_action)
        portfolio_value.append(env.balance + env.stock_owned * env.data['close'].iloc[env.current_step-1])

        states = next_states

    portfolio_values.append(portfolio_value)




# Ausgabe der individuellen Aktionen jedes Agenten -> zur Kontrolle
for agent_type, agent_actions in individual_agent_actions.items():
    print(f"Aktionen von {agent_type}: {agent_actions}")









# Plot der Portfolio-Werte

# Laden des Testdatensatzes
test_data_path = r'C:\Users\Joel\Desktop\FH-Wedel\Deep Learning Projekt\Git\Deep_Learning-2\test_data.csv'
test_data = pd.read_csv(test_data_path)

# Erstellen einer Figur und definieren von zwei Subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

# Zeichnen des Aktiencharts (Schließkurse) und der MAs auf dem ersten Subplot
ax1.plot(test_data['close'], label='Schließkurse', color='blue')
if 'ma5' in test_data.columns:
    ax1.plot(test_data['ma5'], label='MA 5 Tage', color='orange')
if 'ma30' in test_data.columns:
    ax1.plot(test_data['ma30'], label='MA 30 Tage', color='green')
if 'ma200' in test_data.columns:
    ax1.plot(test_data['ma200'], label='MA 200 Tage', color='red')

# Markierungen für Kauf- und Verkaufsorders

for i, action in enumerate(actions):
    #action
    if action == 1:  # Kauf
        ax1.scatter(i, test_data['close'].iloc[i], color='green', marker='^')  # Kein Label für Kauf
    elif action == -1:  # Verkauf
        ax1.scatter(i, test_data['close'].iloc[i], color='red', marker='v')  # Kein Label für Verkauf
    print(action)
ax1.set_title('Aktienchart mit gleitenden Durchschnitten und Orders')
ax1.set_xlabel('Zeit')
ax1.set_ylabel('Preis')
ax1.legend()

# Zeichnen des RSI auf dem zweiten Subplot
if 'rsi' in test_data.columns:
    ax2.plot(test_data['rsi'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='green', linestyle='--')
    ax2.set_ylim([0, 100])
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.set_xlabel('Zeit')
    ax2.set_ylabel('RSI')
    ax2.legend()

# Anzeigen des Plots
plt.tight_layout()
plt.show()
