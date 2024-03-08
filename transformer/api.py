import pandas as pd
from fastapi import FastAPI
from src_transformers.abstract_model import resolution as resolution_enum
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
def predict_transformer(stock_symbols: str = "[AAPL, AAL, AMD]",
                        start_date: str = "2021-02-01",
                        end_date: str = "2021-05-01",
                        resolution: resolution_enum = resolution_enum.DAILY) -> dict:
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
    symbols_list = stock_symbols[1:-1].split(", ")
    transformer_interface = TransformerInterface(resolution)

    # TODO: Remove resolution from the function signature and use the resolution_enum.TWO_HOURLY directly
    # prediction = transformer_interface.predict(symbols_list, pd.to_datetime(
    #     start_date), pd.to_datetime(end_date), resolution.TWO_HOURLY)
    prediction = transformer_interface.predict(
        symbols_list, pd.to_datetime(start_date))

    data = {}
    for symbol in symbols_list:
        symbol_prediction = prediction[f"close {symbol}"]
        data[symbol] = [{"date": date, "value": value}
                        for date, value in symbol_prediction.items()]

    return data
