import os
import pandas as pd
import matplotlib.pyplot as plt

# Pfad zum Ordner, der die TXT-Dateien enthält
ordner_pfad = 'C:\\Users\\Dell\\Downloads\\usindex_1min_u8d0l_(1)'


dataframes = {}
spaltennamen = ['Datum_Uhrzeit', 'Öffnungspreis', 'Höchstpreis', 'Tiefstpreis', 'Schlusspreis']

# Durchlaufen aller Dateien im angegebenen Ordner
for dateiname in os.listdir(ordner_pfad):
    if dateiname.endswith('.txt'):
        datei_pfad = os.path.join(ordner_pfad, dateiname)
        df = pd.read_csv(datei_pfad, sep=',', header=None, names=spaltennamen, parse_dates=['Datum_Uhrzeit'], index_col='Datum_Uhrzeit')
        dataframes[dateiname[:-4]] = df


# alles vor 2018 abschneiden
#gefilterte_dataframes = []
#for df in dataframes:
#    gefilterter_df = df[df.index.year > 2018]
#    gefilterte_dataframes.append(gefilterter_df)


# Erstellen eines vollständigen Datums-/Zeitindex für das Jahr 2019
start = '2020-01-01 00:00:00'
ende = '2020-12-31 23:59:00'
vollständiger_index = pd.date_range(start=start, end=ende, freq='T') # 'T' steht für minutengenaue Frequenz
for aktienname, df in dataframes.items():
    df_vollständig = df.reindex(vollständiger_index, fill_value=0)
    df_vollständig.index.name = 'Datum_Uhrzeit'
    dataframes[aktienname] = df_vollständig

# Überprüfen des Ergebnisses
#print(df_vollständig.head())

nullen_pro_aktie = {aktienname: df['Höchstpreis'].isin([0]).sum() for aktienname, df in dataframes.items()}

nullen_df = pd.DataFrame(list(nullen_pro_aktie.items()), columns=['Aktienname', 'Anzahl der Nullen'])

# Ausgeben des DataFrame
print(nullen_df)

# Erstellen und Anzeigen des Balkendiagramms
fig, ax = plt.subplots()
ax.bar(nullen_df['Aktienname'], nullen_df['Anzahl der Nullen'])
ax.set_title('Anzahl der Missing Values pro Indice')
ax.set_xlabel('Indice')
ax.set_ylabel('Anzahl Missing Values')
plt.xticks(rotation=45, ha="right")
plt.show()


