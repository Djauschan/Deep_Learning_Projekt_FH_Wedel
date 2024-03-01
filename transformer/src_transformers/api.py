import pandas as pd
from fastapi import FastAPI
from src_transformers.prediction_interface import TransformerInterface
from src_transformers.abstract_model import resolution

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Prediction API for the Transformer group :)))"}


@app.get("/predict")
def predict_transformer(stock_symbol: str, start_date: str, end_date: str):
    transformer_interface = TransformerInterface()

    prediction = transformer_interface.predict(["AAPL", "NVDA"], pd.to_datetime(
        start_date), pd.to_datetime(end_date), resolution.TWO_HOURLY)

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction[f"close {stock_symbol.upper()}"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data


if __name__ == "__main__":
    symbols = ["AAPL", "AAL", "AMD", "C", "NVDA", "SNAP", "SQ", "TSLA"]
    start_date = "2019-01-30"
    end_date = "2021-01-30"
    predictions = []

    # Generate time range for every 2 hours between start_date and end_date
    date_range = pd.date_range(start_date, end_date, freq="2H")

    for timestamp in date_range:
        step_predictions = {}
        for symbol in symbols:
            prediction = predict_transformer(symbol, timestamp, None)
            step_predictions[symbol] = prediction[0]["value"]
        predictions.append(step_predictions)

    prediction = pd.DataFrame(predictions, index=date_range)

    prediction.to_csv("data/output/prediction_data_rl.csv")
