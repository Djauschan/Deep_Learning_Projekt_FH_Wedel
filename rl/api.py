from prediction_interface.rl_interface import RLInterface
from prediction_interface.abstract_model import resolution as resolution_type

from fastapi import FastAPI

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
def predict_rl(stock_symbols: str = '[aapl, snap]',
               start_date: str = '2021-01-04',
               end_date: str = '2021-01-05',
               resolution: resolution_type = resolution_type.TWO_HOURLY) -> dict[str, dict[object, dict[str, str]]]:
    """Predicts trading actions for a given stock symbol and time frame with every avialible model.

    Args:
        stock_symbols (str, optional): The stock symbols to predict trading actions for. Defaults to '[aapl, snap]'.
        start_date (str, optional): The start date of the time frame to predict trading actions for. Defaults to '2021-01-04'.
        end_date (str, optional): The end date of the time frame to predict trading actions for. Defaults to '2021-01-05'.
        resolution (resolution_type, optional): The resolution of the stock data. Defaults to resolution_type.TWO_HOURLY.

    Returns:
        dict[str, dict[object, dict[str, str]]]: A dictionary containing the predictions for every model for every timestep in the given time frame.
    """
    # convert stock_symbols to list "[AAPL, AAL, AMD]" -> ["AAPL", "AAL", "AMD"]
    stock_symbols = stock_symbols[1:-1].split(", ")
    return_dict = {}
    for stock_symbol in stock_symbols:
        return_dict[stock_symbol] = interface.predict(stock_symbol, start_date, end_date, resolution)
    return return_dict