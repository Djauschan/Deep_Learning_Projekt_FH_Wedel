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
def predict_rl(stock_symbols: str = '[aapl, snap]', start_date: str = '2021-01-04', end_date: str = '2021-01-05') -> list[dict[str, dict[object, dict[str, str]]]]:
    """Predicts trading actions for a given stock symbol and time frame with every avialible model.

    Args:
        stock_symbols (str, optional): The stock symbols to predict trading actions for. Defaults to '[aapl, snap]'.
        start_date (str, optional): The start date of the time frame to predict trading actions for. Defaults to '2021-01-04'.
        end_date (str, optional): The end date of the time frame to predict trading actions for. Defaults to '2021-01-05'.

    Returns:
        list[dict[str, dict[object, dict[str, str]]]]: A list containing the predictions for every every stock symbol, every model for every 2hours in the given time frame.
    """
    # convert stock_symbols to list "[AAPL, AAL, AMD]" -> ["AAPL", "AAL", "AMD"]
    stock_symbols = stock_symbols[1:-1].split(", ")
    return_dict = {}
    for stock_symbol in stock_symbols:
        return_dict[stock_symbol] = interface.predict(stock_symbol, start_date, end_date)
        print(interface.predict(stock_symbol, start_date, end_date))
    return [return_dict]