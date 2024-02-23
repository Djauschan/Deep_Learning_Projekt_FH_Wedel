from prediction_interface.rl_interface import RLInterface

from fastapi import FastAPI
import pandas as pd

interface = RLInterface()
app = FastAPI()

@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint for the RL-prediction API.

    Returns:
        dict[str, str]: A simple message.
    """
    return {"message": "Prediction API for the RL group :)))"}

@app.get("/predict/")
def predict_rl(stock_symbol: str = 'aapl', start_date: str = '2021-01-04', end_date: str = '2021-01-05') -> list[dict[object, dict[str, str]]]:
    """Predicts trading actions for a given stock symbol and time frame with every avialible model.

    Args:
        stock_symbol (str, optional): The stock symbol to predict trading actions for. Defaults to 'aapl'.
        start_date (str, optional): The start date of the time frame to predict trading actions for. Defaults to '2021-01-04'.
        end_date (str, optional): The end date of the time frame to predict trading actions for. Defaults to '2021-01-05'.

    Returns:
        list[dict[Timestamp, dict[str, str]]]: A list containing the predictions for every model for every hour in the given time frame.
    """
    return [interface.predict(stock_symbol, start_date, end_date)]