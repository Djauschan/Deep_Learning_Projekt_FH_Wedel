from fastapi import FastAPI
from prediction_interface.rl_interface import RLInterface

interface = RLInterface()
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Prediction API for the RL group :)))"}

@app.get("/predict/")
def predict_rl(stock_symbol: str = 'aapl', start_date: str = '2021-01-04', end_date: str = '2021-01-05'):
    
    prediction = interface.predict(stock_symbol, start_date, end_date)
    
    return [prediction]