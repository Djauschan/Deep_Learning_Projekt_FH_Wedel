from collections import namedtuple, deque
import random
import torch

class Replay_memory():
    """Class for the replay memory which is used for the DQN-Agent.
    """
    def __init__(self, capacity : int, transition : namedtuple) -> None:
        """The constructor of the Replay_memory class.

        Args:
            capacity (int): The maximum capacity of the replay memory.
        """
        self.memory = deque([], maxlen=capacity)
        self.transition = transition
    
    def push(self, *args : list) -> None:
        """Adds a transition to the replay memory.
        """
        self.memory.append(self.transition(*args))
    
    def sample(self, batch_size : int) -> list: 
        """Samples a batch of transitions from the replay memory.

        Args:
            batch_size (int): The size of the batch to be sampled.

        Returns:
            list: A list of transitions.
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        """Returns the length of the replay memory.

        Returns:
            int: The length of the replay memory.
        """
        return len(self.memory)