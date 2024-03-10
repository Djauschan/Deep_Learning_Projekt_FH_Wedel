import matplotlib.pyplot as plt
import numpy as np
        
def plot_rewards(rewards : list, show_result : bool = False) -> None:
    if len(rewards) < 2:
        return
    plt.ion()
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xticks(np.arange(len(rewards)), np.arange(1, len(rewards)+1))
    plt.xlabel('Episode')
    plt.ylabel('Kumulative Belohnung')
    plt.plot(rewards)
    
    plt.pause(1)
    
    if show_result:
        plt.ioff()
        plt.show()