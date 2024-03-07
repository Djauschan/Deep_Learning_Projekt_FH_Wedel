import pandas as pd
from fastapi import FastAPI
from src_transformers.abstract_model import resolution
from src_transformers.prediction_interface import TransformerInterface

app = FastAPI()


@app.get("/")
async def root() -> dict:
    """
    This endpoint returns the description of the API.

    Returns:
        dict: Description of the API.
    """
    return {"message": "Prediction API for the Transformer group :)))"}


@app.get("/predict")
def predict_transformer(stock_symbols: str, start_date: str, end_date: str, resolution: resolution) -> dict:
    """
    This endpoint returns the stock predictions for the given stock symbols, start date, end date and resolution.

    Args:
        stock_symbols (str): Stock symbols for which the predictions should be made.
        start_date (str): Start date for the predictions.
        end_date (str): End date for the predictions.
        resolution (resolution): Time resolution for the predictions.

    Returns:
        dict: Predictions for the given stock symbols for the given time range and resolution.
    """
    # convert stock_symbols to list "[AAPL, AAL, AMD]" -> ["AAPL", "AAL", "AMD"]
    stock_symbols = stock_symbols[1:-1].split(", ")
    transformer_interface = TransformerInterface()

    prediction = transformer_interface.predict(stock_symbols, pd.to_datetime(
        start_date), pd.to_datetime(end_date), resolution.TWO_HOURLY)

    data = {}
    for symbol in stock_symbols:
        symbol_prediction = prediction[f"close {symbol}"]
        data[symbol] = [{"date": date, "value": value}
                        for date, value in symbol_prediction.items()]

    return data
