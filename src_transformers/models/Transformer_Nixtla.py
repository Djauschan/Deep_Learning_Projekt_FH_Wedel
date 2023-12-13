import os
import pandas as pd

# Erhalte den Pfad zum Verzeichnis der aktuellen Python-Datei
current_dir = os.path.dirname(os.path.abspath(__file__))

# Baue den Pfad zur CSV-Datei relativ zum aktuellen Verzeichnis
csv_path = os.path.join(current_dir, '..', '..', 'data', 'output', 'Multi_Symbol.csv')

df = pd.read_csv(csv_path)

print(df)