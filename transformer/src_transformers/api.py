import pandas as pd
from fastapi import FastAPI
from transformer_interface import TransformerInterface

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Prediction API for the Transformer group :)))"}


@app.get("/predict")
def predict_transformer(stock_symbol: str, start_date: str, end_date: str):
    transformer_interface = TransformerInterface()

    prediction = transformer_interface.predict(
        pd.to_datetime(start_date), pd.to_datetime(end_date))

    # Convert the prediction to a list of dictionaries
    prediction_data = prediction[f"close {stock_symbol.upper()}"]
    data = [{"date": date, "value": value}
            for date, value in prediction_data.items()]

    return data
