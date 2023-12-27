import pandas as pd
from tqdm import tqdm
from torch import tensor, float32, int64
import matplotlib.pyplot as plt
import sys
import numpy as np

from agents.dqn import DQN_Agent
from environments.TraidingEnvironment import TradingEnvironment
from utils.read_config import Config_reader

def main():
    # Read config file
    config = Config_reader('config/config_dqn.yml')
    
    train_data_path = config.get_parameter('train_data', 'directories')
    
    train_data = pd.read_csv(train_data_path)
    env = TradingEnvironment(train_data)
    states = env.reset()
    ma30_agent = DQN_Agent(state_size=len(states['ma30']), action_size=3)
    
    NUM_EPISODES = config.get_parameter('epochs', 'train_parameters')
    
    performance = []
    cumulative_rewards = []
    
    for episode in tqdm(range(NUM_EPISODES)):
        states = env.reset()
        done = False
        episode_rewards = 0
        
        while not done:
            state = tensor(states['ma30'], dtype=float32).unsqueeze(0)
            final_action = ma30_agent.act(state)
            next_states, reward, done = env.step(final_action)
            try:
                next_state = tensor(next_states['ma30'], dtype=float32).unsqueeze(0)
            except TypeError:
                next_state = tensor([])
            
            if not done:
                ma30_agent.replay_memory.push(state, tensor([[final_action]]), tensor([reward]), next_state)
                states = next_states
                ma30_agent.train()
            
            if env.done:
                performance.append(env.messen_der_leistung)
                done = True
                ma30_agent.update_epsilon()
        
        states = env.reset()
        done = False
        while not done:
            state = tensor(states['ma30'], dtype=float32).unsqueeze(0)
            final_action = ma30_agent.validate(state)
            next_states, reward, done = env.step(final_action)
            states = next_states
            episode_rewards += reward
            
            if env.done:
                cumulative_rewards.append(episode_rewards)
                done = True
        
                
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cumulative_rewards)
    plt.title(f"Kumulative Belohnungen pro Episode für MA30 Agent")
    plt.xlabel("Episode")
    plt.ylabel("Kumulative Belohnung")

    plt.show()

if __name__ == '__main__':
    main()