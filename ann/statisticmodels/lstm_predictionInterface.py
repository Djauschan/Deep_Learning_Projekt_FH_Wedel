import os

# from tensorflow.keras.models import load_model
import pandas as pd
import statisticmodels.lstmMVP_V2 as lstmMVP
from abstract_model import AbstractModel


class LstmInterface(AbstractModel):

    def predict(self, timestamp_start: pd.Timestamp, timestamp_end: pd.Timestamp, interval: int) -> pd.DataFrame:

        business_days = pd.bdate_range(
            start=timestamp_start, end=timestamp_end)
        days_difference = len(business_days)
        path = "statisticmodels/lstmMVP_V2.py"
        all_predictions = pd.DataFrame()

        # Erzeugen eines Datumsbereichs für Geschäftstage
        business_days = pd.bdate_range(
            start=timestamp_start, end=timestamp_end)
        # Konvertieren in eine pandas Series
        business_days_series = list(business_days)
        # all_predictions = prediction
        # print(all_predictions)
        # initialisierung der StockModel Klasse
        stock_model = lstmMVP.StockModel(
            "..\..\..\data\Aktien\AAPL_1min.txt", '2021-03-03')
        all_predictions = stock_model.predict_x_days(
            days_difference, pd.to_datetime(timestamp_start))
        return all_predictions

        # Da Nur Apple genommen wird, wird alles vorerst kommentiert
        # for filename in os.listdir(path):
        #     if filename.endswith(".h5"):
        #         model_path = os.path.join(path, filename)
        #         # LSTM modell laden
        #         model = lstmMVP.load_model(model_path)
        #         prediction = model.predict_x_days(2, "2021-01-04")

        #         prediction['ds'] =  business_days_series
        #         prediction['symbol'] = filename[:-4]

        #         all_predictions = pd.concat([all_predictions, prediction], ignore_index=True)
        #         print(all_predictions)

        return all_predictions

    def load_data(self) -> None:
        pass

    def preprocess(self) -> None:
        pass

    def load_model(self) -> None:
        pass


def main():
    test = LstmInterface()
    print(test.predict('2021-01-04', '2021-01-06', 120))


if __name__ == '__main__':
    main()
