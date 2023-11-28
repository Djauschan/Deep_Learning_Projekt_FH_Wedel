import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
 
 
 
class VisualizeStatsModel():
 
    def __init__(self, data:pd.DataFrame, prediction, train, test):
        self.data=data
        self.prediction=prediction
        self.train= train
        self.test=test
 
    def simpleViz(self, index, horizon:int, test=None, prediction=None):
        plt.style.use('fivethirtyeight')
 
        if test is None:
            test = self.test
        if prediction is None:
            prediction = self.prediction
 
        fig, ax = plt.subplots()
        ax.plot_date(index, test[0:horizon], 'o--',color='red',label='Tatsächliche Daten')
        ax.plot_date(index, prediction, 'o:',color='blue', label='Forecast')
        ax.set_xlabel('Zeit')
        ax.set_ylabel('Aktienkurs in Dollar')
        plt.xticks(rotation=45)
        ax.legend()
        plt.show()
 
    def arimaCrossvalidation(self, test, Prognosedic):
        plt.style.use('fivethirtyeight')
        colors = ['b', 'g', 'r', 'c', 'm']
 
        for i,(key, values) in enumerate(Prognosedic.items()):
            plt.plot(values, label=key)
 
        # Achsentitel und Legende hinzufügen
        plt.plot(test,label='Echte Daten',color='k')
        plt.title('Prognosen')
        plt.xlabel('Zeitpunkt')
        plt.ylabel('Aktienkurs')
        plt.title('Vergleich der Modelle')
        plt.legend()
 
        plt.show()
