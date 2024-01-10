import numpy as np
import pandas as pd
#end to end Walkthrough
from ray import tune

from neuralforecast.core import NeuralForecast
from neuralforecast.auto import AutoNHITS, AutoLSTM
from neuralforecast.losses.pytorch import MQLoss


from neuralforecast.models import NBEATS, NHITS


class Nbeats(): 
    
    def __init__(self, horizon, ar):
        self.nbeatsModel= NBEATS(h=horizon,input_size=ar,max_steps=1500, learning_rate=0.003,n_freq_downsample=[2, 1, 1])
        self.nf=NeuralForecast(models=[self.nbeatsModel],freq='B')

    def forecast(self,y_train):
        self.nf.fit(df=y_train)
        y_hat=self.nf.predict().reset_index()
        return y_hat

class Nhits(): 
    
    def __init__(self, horizon, ar):
        self.nhitsModel= NHITS(h=horizon,input_size=ar,learning_rate=0.003,max_steps=1500)
        self.nf=NeuralForecast(models=[self.nhitsModel],freq='B')

    def forecast(self,y_train):
        self.nf.fit(df=y_train)
        y_hat=self.nf.predict().reset_index()
        return y_hat

class Autoexp():

    def __init__(self,horizon):
        self.config_nhits = {
       "learning_rate": tune.choice([1e-3]),                                     # Initial Learning rate
       "max_steps": tune.choice([1000]),                                         # Number of SGD steps
       "input_size": tune.choice([5 * horizon]),                                 # input_size = multiplier * horizon
       "batch_size": tune.choice([7]),                                           # Number of series in windows
       "windows_batch_size": tune.choice([256]),                                 # Number of windows in batch
       "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),               # MaxPool's Kernelsize
       "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]), # Interpolation expressivity ratios
       "activation": tune.choice(['ReLU']),                                      # Type of non-linear activation
       "n_blocks":  tune.choice([[1, 1, 1]]),                                    # Blocks per each 3 stacks
       "mlp_units":  tune.choice([[[512, 512], [512, 512], [512, 512]]]),        # 2 512-Layers per block for each stack
       "interpolation_mode": tune.choice(['linear']),                            # Type of multi-step interpolation
       "val_check_steps": tune.choice([100]),                                    # Compute validation every 100 epochs
       "random_seed": tune.randint(1, 10),
        }
        
        self.nf=nf = NeuralForecast(
        models=[AutoNHITS(h=horizon, config=self.config_nhits, loss=MQLoss(), num_samples=5)],
        freq='B'
        )

    def forecast(self,y_train):
        self.nf.fit(df=y_train)
        fcst_df=self.nf.predict()
        fcst_df.columns = fcst_df.columns.str.replace('-median', '')
        return fcst_df






