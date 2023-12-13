import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from neuralforecast.core import NeuralForecast
from neuralforecast.models import Informer, Autoformer, FEDformer, PatchTST

# Erhalte den Pfad zum Verzeichnis der aktuellen Python-Datei
current_dir = os.path.dirname(os.path.abspath(__file__))

# Baue den Pfad zur CSV-Datei relativ zum aktuellen Verzeichnis
csv_path = os.path.join(current_dir, '..', '..', 'data', 'output', 'Multi_Symbol.csv')

df = pd.read_csv(csv_path)

# posix Zeiten in Zeitstempel umwandeln
df['posix_time'] = pd.to_datetime(df['posix_time'], unit='s')

df = df.rename(columns={'posix_time': 'ds'})
df = df.rename(columns={'close NIO': 'y'})

# nur Zeitstempel und close NIO Werte behalten
Y_df = df.drop(df.columns[[1,2,3,4,5]], axis=1)

Y_df.insert(0, 'unique_id', 'NIO')

print(Y_df)

n_time = len(Y_df.ds.unique())
val_size = int(.2 * n_time)
test_size = int(.2 * n_time)

Y_df.groupby('unique_id').head(2)


# Überprüfe, ob CUDA verfügbar ist
if torch.cuda.is_available():
    # Definiere die Modelle
    horizon = 90    # wie häufig 1 min in die Zukunft vorhersagen
    models = [Informer(h=horizon,                   # Forecasting horizon
                    input_size=horizon,             # Input size
                    max_steps=1000,                 # Number of training iterations
                    val_check_steps=100,            # Compute validation loss every 100 steps
                    early_stop_patience_steps=3),   # Stop training if validation loss does not improve
            """Autoformer(h=horizon,
                    input_size=horizon,
                    max_steps=1000,
                    val_check_steps=100,
                    early_stop_patience_steps=3),
            PatchTST(h=horizon,
                    input_size=horizon,
                    max_steps=1000,
                    val_check_steps=100,
                    early_stop_patience_steps=3)"""]

    # Bewege die Modelle auf die GPU
    for model in models:
        model.to('cuda')

    print("CUDA verfügbar. Modelle wurden auf die GPU verschoben.")
else:
    print("CUDA nicht verfügbar. Die Modelle werden auf der CPU ausgeführt.")

nf = NeuralForecast(
    models=models,
    freq='1min')

Y_hat_df = nf.cross_validation(df=Y_df,
                               val_size=val_size,
                               test_size=test_size,
                               n_windows=None)


Y_hat_df.head()

Y_plot = Y_hat_df[Y_hat_df['unique_id']=='NIO'] # NIO dataset
cutoffs = Y_hat_df['cutoff'].unique()[::horizon]
Y_plot = Y_plot[Y_hat_df['cutoff'].isin(cutoffs)]

plt.figure(figsize=(20,5))
plt.plot(Y_plot['ds'], Y_plot['y'], label='True')
plt.plot(Y_plot['ds'], Y_plot['Informer'], label='Informer')
#plt.plot(Y_plot['ds'], Y_plot['Autoformer'], label='Autoformer')
#plt.plot(Y_plot['ds'], Y_plot['PatchTST'], label='PatchTST')
plt.xlabel('Datestamp')
plt.ylabel('NIO')
plt.grid()
plt.legend()


# compute test error using MAE
from neuralforecast.losses.numpy import mae

mae_informer = mae(Y_hat_df['y'], Y_hat_df['Informer'])
#mae_autoformer = mae(Y_hat_df['y'], Y_hat_df['Autoformer'])
#mae_patchtst = mae(Y_hat_df['y'], Y_hat_df['PatchTST'])

print(f'Informer: {mae_informer:.3f}')
#print(f'Autoformer: {mae_autoformer:.3f}')
#print(f'PatchTST: {mae_patchtst:.3f})'

