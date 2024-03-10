import pandas as pd
from tqdm import tqdm
from torch import tensor, float32

from agents.dqn import DQN_Agent
from environments.TraidingEnvironment import TradingEnvironment
from utils.read_config import Config_reader
from utils.vis_results import plot_rewards

def main():
    # Read config file
    config = Config_reader('config/config_dqn.yml')
    
    train_data_path = config.get_parameter('train_data', 'directories')
    
    train_data = pd.read_csv(train_data_path)
    env = TradingEnvironment(train_data)
    states = env.reset()
    agent = DQN_Agent(state_size=len(states['ma200']), action_size=3)
    
    NUM_EPISODES = config.get_parameter('epochs', 'train_parameters')
    
    performance = []
    cumulative_rewards = []
    
    for episode in tqdm(range(NUM_EPISODES)):
        states = env.reset()
        done = False
        episode_rewards = 0
        
        while not done:
            state = tensor(states['ma200'], dtype=float32).unsqueeze(0)
            final_action = agent.act(state)
            next_states, reward, done = env.step(final_action)
            try:
                next_state = tensor(next_states['ma200'], dtype=float32).unsqueeze(0)
            except TypeError:
                next_state = tensor([])
            
            if not done:
                agent.replay_memory.push(state, tensor([[final_action]]), tensor([reward]), next_state)
                states = next_states
                agent.train()
            else:
                performance.append(env.messen_der_leistung)
                agent.update_epsilon()
        
        states = env.reset()
        done = False
        while not done:
            state = tensor(states['ma200'], dtype=float32).unsqueeze(0)
            final_action = agent.validate(state)
            next_states, reward, done = env.step(final_action)
            states = next_states
            episode_rewards += reward
            
            if done:
                cumulative_rewards.append(episode_rewards)
                # plot_rewards(cumulative_rewards)
    
    plot_rewards(cumulative_rewards, show_result=True)

if __name__ == '__main__':
    main()