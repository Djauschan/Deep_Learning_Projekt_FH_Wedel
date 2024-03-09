import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import sys

from utils.replay_memory import Replay_memory


class DQN(nn.Module):
    """Class for the DQN-Network.
    """
    def __init__(self, state_size : int, action_size : int) -> None:
        """The constructor of the DQN class.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Forward pass of the DQN-Network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DQN_Agent():
    """Class for the DQN-Agent.
    """
    def __init__(self, state_size : int,
                 action_size : int,
                 replay_memory_size : int = 10000,
                 batch_size : int = 128,
                 gamma : float = 0.95,
                 epsilon : float = 1.0,
                 epsilon_min : float = 0.01,
                 epsilon_decay : float = 0.995,
                 learning_rate : float = 0.001,
                 tau :float = 0.005):
        """The constructor of the DQN class.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
            replay_memory_size (int, optional): The maximum size of the replay_memory. Defaults to 10000.
            batch_size (int, optional): The size of the batches. Defaults to 32.
            gamma (float, optional): The discountfactor which is used for the training. Defaults to 0.95.
            epsilon (float, optional): The starting value of epsilon. Defaults to 1.0.
            epsilon_min (float, optional): The final value of epsilon. Defaults to 0.01.
            epsilon_decay (float, optional): Controlls the rate of epsilon decay. Defaults to 0.995.
            learning_rate (float, optional): The learning rate. Defaults to 0.001.
            tau (float, optional): The update rate of the target network. Defaults to 0.005.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.replay_memory = Replay_memory(replay_memory_size, self.transition)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.policy_network = DQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
        self.steps_done = 0
    
    def update_epsilon(self) -> None:
        """Updates epsilon.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        
    def act(self, state : torch.Tensor) -> int:
        """Returns an action based on the current state.

        Args:
            state (torch.Tensor): The current state.

        Returns:
            int: The action to be taken.
        """
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, self.action_size, (1,)).item()
        else:
            with torch.no_grad():
                return self.policy_network(state).argmax().item()
    
    def validate(self, state : torch.Tensor) -> int:
        """Returns an action based on the current state without exploration.

        Args:
            state (torch.Tensor): The current state.

        Returns:
            int: The action to be taken.
        """
        with torch.no_grad():
            return self.policy_network(state).argmax().item()
    
    def train(self):
        """Trains the DQN-Network.
        """
        if len(self.replay_memory) < self.batch_size:
            return
        
        transitions = self.replay_memory.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values
        excpected_state_action_values = (next_state_values * self.gamma) + reward_batch
  
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, excpected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()