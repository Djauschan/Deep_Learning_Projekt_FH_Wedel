import bcrypt
import crud
import models
import pandas as pd
import numpy as np
import requests
import schemas
import math
from database import SessionLocal, engine
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Create tables in the database based on the model definitions
models.Base.metadata.create_all(bind=engine)

# Create instance of FastAPI
app = FastAPI()


origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",  # Frontend
    "http://localhost:8000",  # Backend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB Dependency


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# test method to get Stock data for n days


@app.get("/getStock/")
def get_stock_days(stock_symbol: str, days_back: int, db: Session = Depends(get_db)):
    return crud.get_stock_days(stock_symbol=stock_symbol, n=days_back, db=db)

# post method to delete table "users"


@app.post("/deleteUsers/")
def delete_users(db: Session = Depends(get_db)):
    db_delete_users = crud.delete_users(db)
    if db_delete_users:
        raise HTTPException(status_code=400, detail="Not Found")
    # return crud.get_users(db=db, skip=0, limit=100)

# post method to delete a single user from table "users" by id


@app.post("/deleteUser/{username}")
def delete_user(username: str, db: Session = Depends(get_db)):
    db_delete_user = crud.delete_user(db, username)
    if db_delete_user is None:
        raise HTTPException(status_code=400, detail="User not found")
    return {"message": "User deleted successfully", "status_code": 200}

# post method to create a user into table "users"


@app.post("/createUser/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user_email = crud.get_user_by_email(db, email=user.email)
    db_user_username = crud.get_user_by_username(db, username=user.username)
    if db_user_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    if db_user_username:
        raise HTTPException(
            status_code=400, detail="Username already registered")
    return crud.create_user(db=db, user=user)


@app.get("/get_user/{username}")
async def get_user(username: str, db: Session = Depends(get_db)):
    return crud.get_user_by_username(db, username)


@app.get("/get_budget/{username}")
async def get_budget(username: str, db: Session = Depends(get_db)):
    return crud.get_budget_by_username(db, username)


@app.get("/get_user/{email}")
async def get_user(email: str, db: Session = Depends(get_db)):
    return crud.get_user_by_email(db, email)


@app.put("/update_user/{username}")
async def update_user(username: str, user_update: schemas.UserUpdate, db: Session = Depends(get_db)):
    updated_user = crud.update_user_by_username(db, username, user_update)
    return {"message": "User updated successfully"}


@app.put("/update_budget/{username}")
async def update_budget_by_user(username: str, budgetInput: int, db: Session = Depends(get_db)):
    updated_budget = crud.update_budget_by_user(
        db=db, username=username, new_budget=budgetInput)
    return {"message": "Budget updated successfully"}


@app.get("/getUsers/", response_model=list[schemas.User])
def read_users(query: str = '', limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, query=query, limit=limit)
    return users


@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.post("/login/")
def user_login(login_request: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, login_request.user)

    if not user:
        raise HTTPException(status_code=400, detail="Username does not exist")

    if not bcrypt.checkpw(login_request.password.encode('utf-8'), user.password):
        raise HTTPException(status_code=400, detail="Incorrect password")
    # Creating a login record after successful authentication
    # Removed schemas.LoginCreate()
    login_record = crud.create_login(db, owner_id=user.id)

    if not login_record:
        raise HTTPException(
            status_code=500, detail="Could not create login record")

    return {
        "username": user.username,
        "email": user.email,
        "user_id": user.id,
        "is_active": user.is_active
    }


@app.get("/getLogins/", response_model=list[schemas.Login])
def get_logins(query: str = '', limit: int = 100, db: Session = Depends(get_db)):
    logins = crud.get_logins(db, query=query, limit=limit)
    return logins


@app.get("/getLoginByUser_ID/", response_model=list[schemas.Login])
def get_logins(user_id: int, db: Session = Depends(get_db)):
    logins = crud.get_logins_by_user_id(db, owner_id=user_id)
    return logins


@app.get("/login/validation")
def check_login(email: str, password: str, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, email)

    if user:
        raise HTTPException(status_code=400, detail="Email does not exist")


def calculate_end_date(start_date: str, resolution: str):
    if resolution == "D":
        return (pd.Timestamp(start_date) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    elif resolution == "H":
        return (pd.Timestamp(start_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    elif resolution == "M":
        # We use 19 minutes because of the ann prediction interface implementation (it returns 1 value too much)
        return (pd.Timestamp(start_date) + pd.Timedelta(minutes=19)).strftime("%Y-%m-%d %H:%M:%S")
    else:
        raise ValueError("Invalid resolution")


@app.get("/predict/transformer")
def predict_transformer(stock_symbols: str = "[AAPL, NVDA]",
                        start_date: str = "2021-01-04",
                        resolution: str = "D"):
    if resolution == "M":
        start_date += " 10:01:00"

    end_date = calculate_end_date(start_date, resolution)
    data_to_send = {"stock_symbols": stock_symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "resolution": resolution}

    api_url = "http://predict_transformer:8000/predict"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/cnn")
def predict_cnn(stock_symbols: str = "[AAPL, NVDA]",
                start_date: str = "2021-01-04",
                resolution: str = "D"):
    end_date = calculate_end_date(start_date, resolution)
    data_to_send = {"stock_symbol": stock_symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "resolution": resolution}

    api_url = "http://predict_cnn:8000/predict"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()

@app.get("/predict/randomForest")
def predict_randomForest(stock_symbols: str = "[AAPL]",
                         start_date: str = "2021-01-04",
                         resolution: str = "H"):
    if resolution == "M":
        start_date += " 10:01:00"
    end_date = calculate_end_date(start_date, resolution)
    if resolution == "H":
        start_date += " 10:00:00"
        end_date += " 16:00:00"

    data_to_send = {"stock_symbols": stock_symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "resolution": resolution}

    api_url = "http://predict_ann:8000/predict/randomForest"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/gradientBoost")
def predict_gradientBoost(stock_symbols: str = "[AAPL]",
                          start_date: str = "2021-01-04",
                          resolution: str = "H"):
    if resolution == "M":
        start_date += " 10:01:00"
    end_date = calculate_end_date(start_date, resolution)
    if resolution == "H":
        start_date += " 10:00:00"
        end_date += " 18:00:00"

    data_to_send = {"stock_symbols": stock_symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "resolution": resolution}

    api_url = "http://predict_ann:8000/predict/gradientBoost"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()

@app.get("/predict/lstm")
def predict_lstm(stock_symbols: str = "[AAPL]",
                          start_date: str = "2021-01-04",
                          resolution: str = "H"):
    if resolution == "M":
        start_date += " 10:01:00"
    end_date = calculate_end_date(start_date, resolution)
    if resolution == "H":
        start_date += " 08:00:00"
        end_date += " 18:00:00"

    data_to_send = {"stock_symbols": stock_symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "resolution": resolution}

    api_url = "http://predict_ann:8000/predict/lstm"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()

@app.get("/predict/rl")
def predict_rl(stock_symbols: str = "[AAPL, NVDA]",
               start_date: str = "2021-01-04",
               resolution: str = "D"):
    """Predicts trading actions for a given stock symbol and time frame with every avialible model.

    Args:
        stock_symbols (str): The stock symbols to predict trading actions for.
        start_date (str): The start date of the time frame to predict trading actions for.
        resolution (str): The resolution for the prediction.

    Returns:
        dict[Timestamp, dict[str, str]]: A list containing the predictions for every model for every hour in the given time frame.
    """
    if resolution == "M":
        start_date += " 10:00:00"
    end_date = calculate_end_date(start_date, resolution)
    if resolution == "H":
        end_date = calculate_end_date(end_date, resolution)
    data_to_send = {"stock_symbols": stock_symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "resolution": resolution}
    api_url = "http://predict_rl:8000/predict"
    response = requests.get(api_url, params=data_to_send)
    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }
    return response.json()

@app.get("/simulate/rl")
def rl_simulation(stock_symbols: str = "[AAPL, NVDA]",
                  start_date: str = "2021-01-04",
                  resolution: str = "D",
                  username : str = '1',
                  db : Session = Depends(get_db)):
    stock_db = {}
    data_dict = {}
    data = load_data(stock_symbols, start_date, resolution)
    for symbol in stock_symbols[1:-1].split(", "):
        stock_db[symbol] = 0
        data_dict[symbol] = pd.DataFrame(data[symbol]).set_index("DateTime")
    actions = {}
    profit = 0
    start_budget = crud.get_budget_by_username(db, username)
    if start_budget == 0.0:
        return {'Message': 'Geld_Aufladen!'}
    budget = start_budget
    budget_per_trade = round(budget / 10, 2)
    if resolution == "M":
        start_date += " 10:00:00"
    rl_results = predict_rl(stock_symbols, start_date, resolution)
    dates = list(rl_results[symbol].keys())
    for date in dates:
        no_action = True
        actions[date] = {'assets' : {},
                         'actions' : {}}
        for symbol in stock_db.keys():
            actions[date]['actions'][symbol] = ''
            df = data_dict[symbol]
            try:
                price = float(df[df.index == date].Close[0])
            except IndexError:
                price = 0.0
            if price == 0.0:
                continue
            try:
                rl_results[symbol][date]
            except KeyError:
                continue
            if rl_results[symbol][date]['ensemble'] == 'buy' and budget >= price:
                stocks_to_buy = math.ceil(budget_per_trade / price)
                if budget < stocks_to_buy * price:
                    stocks_to_buy = 1
                no_action = False
                budget -= price * stocks_to_buy
                actions[date]['actions'][symbol] = f'{stocks_to_buy}xbuy'
                stock_db[symbol] += stocks_to_buy
            elif stock_db[symbol] != 0:
                if rl_results[symbol][date]['ensemble'] == 'hold':
                    no_action = False
                    actions[date]['actions'][symbol] = 'hold'
                elif rl_results[symbol][date]['ensemble'] == 'sell':
                    no_action = False
                    actions[date]['actions'][symbol] = 'sell'
                    budget += price * stock_db[symbol]
                    stock_db[symbol] = 0
            if stock_db[symbol] > 0:
                actions[date]['assets'][symbol] = round(stock_db[symbol] * price, 2)
        if no_action:
            actions.pop(date)
        else:
            actions[date]['assets']['liquid_money'] = round(budget, 2)
    assets = 0
    date = list(actions.keys())[-1]
    for key in actions[date]['assets'].keys():
        assets += round(actions[date]['assets'][key], 2)
    profit = round(assets - start_budget, 2)
    crud.update_budget_by_user(db, username, profit)
    return {'profit' : profit, 'actions' : actions}
    



@app.get("/load/data")
def load_data(stock_symbols: str = "[AAPL, NVDA]", start_date: str = '2021-01-04', resolution: str = "H"):
    if resolution == "M":
        start_date += " 10:01:00"
    allColumns = ["DateTime", "Open", "Close", "High", "Low", "a"]
    relevantColumns = ["DateTime", "Open", "Close", "High", "Low"]

    if(resolution == "H"):
        end_date = (pd.Timestamp(start_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    else:
        end_date = calculate_end_date(start_date, resolution)

    data = crud.loadDataFromFile(stock_symbols=stock_symbols,
                                 start_date=pd.Timestamp(start_date),
                                 end_date=end_date,
                                 interval=resolution,
                                 ALL_DATA_COLUMNS=allColumns,
                                 COLUMNS_TO_KEEP=relevantColumns)

    return data

@app.get("/get/MAE")
def get_MAE_for_model(stock_symbols: str = '[AAPL, SNAP]', start_date: str = '2021-01-04', resolution: str = 'H', model_type: str = 'transformer'):
    if model_type.lower() == "transformer":
        pred_model = predict_transformer
    elif model_type.lower() == "cnn":
        pred_model = predict_cnn
    elif model_type.lower() == "randomforest":
        pred_model = predict_randomForest
    elif model_type.lower() == "gradientboost":
        pred_model = predict_gradientBoost
    elif model_type.lower() == "lstm":
        pred_model = predict_lstm
    else:
        raise ValueError("Invalid model type")
    
    return_dict = {}

    y_true = load_data(stock_symbols, start_date, resolution)
    y_pred = pred_model(stock_symbols, start_date, resolution)
    print(y_pred)
    stock_symbols = stock_symbols[1:-1].split(", ")
    for stock_symbol in stock_symbols:
        try:
            df_true = pd.DataFrame(y_true[stock_symbol])
            df_pred = pd.DataFrame(y_pred[stock_symbol]).rename(columns={"value": "value_pred"})
            df_pred["date"] = pd.to_datetime(df_pred["date"])
            df_true["date"] = pd.to_datetime(df_true["DateTime"])
            
            df_complete = pd.concat([df_pred.set_index('date'), df_true.set_index('date')], axis=1)
            df_complete = df_complete[df_complete.Close != 0.0].dropna()
            
            return_dict[stock_symbol] = {'MAE' : mean_absolute_error(df_complete["value_pred"].values,df_complete["Close"].values),
                                        'ME' : mean_error(df_complete["value_pred"].values, df_complete["Close"].values),
                                        'profit' : get_model_profit(df_complete["value_pred"].values, df_complete["Close"].values)}
        except Exception as e:
            print(f"Error: {e}")
            return_dict[stock_symbol] = {'MAE' : -999.9,
                                         'ME' : -999.9,
                                         'profit' : -999.9}

    return return_dict

def mean_absolute_error(predictions, targets):
    """
    Calculate the mean absolute error between two NumPy arrays.

    Parameters:
    predictions (numpy.ndarray): A NumPy array of predicted values.
    targets (numpy.ndarray): A NumPy array of target (true) values.

    Returns:
    float: Mean Absolute Error (MAE)
    """
    if predictions.shape != targets.shape:
        raise ValueError("Shape of predictions and targets must be the same.")

    total_error = np.sum(np.abs(predictions - targets))

    return round(total_error / len(predictions), 2)


def mean_error(predictions, targets):
    """
    Calculate the mean error between two NumPy arrays.

    Parameters:
    predictions (numpy.ndarray): A NumPy array of predicted values.
    targets (numpy.ndarray): A NumPy array of target (true) values.

    Returns:
    float: Mean Error (ME)
    """
    if predictions.shape != targets.shape:
        raise ValueError("Shape of predictions and targets must be the same.")

    total_error = np.sum(predictions - targets)

    return round(total_error / len(predictions), 2)

def get_model_profit(predictions, targets) -> float:
    stocks = 0
    invested = 0.0
    cash_out = 0.0
    for real, pred in zip(targets, predictions):
        if real > pred:
            stocks += 5
            invested += (real * 5)
        else:
            cash_out += (real * stocks)
            stocks = 0
    
    profit = stocks * real + cash_out - invested
    
    return round(profit, 2)

""" Outdated code from the original main.py"""

# @app.get("/predict/svm")
# def predict_svm(stock_symbol: str, start_date: str, end_date: str):
#     data_to_send = {"stock_symbol": stock_symbol,
#                     "start_date": start_date,
#                     "end_date": end_date}
#     api_url = "http://predict_ann:8000/predict/svm"
#     response = requests.get(api_url, params=data_to_send)

#     if response.status_code != 200:
#         return {
#             "status_code": response.status_code,
#             "response_text": response.text
#         }

#     return response.json()

# @app.get("/predict/arima")
# def predict_arima(stock_symbol: str, start_date: str, end_date: str):
#     data_to_send = {"stock_symbol": stock_symbol,
#                     "start_date": start_date,
#                     "end_date": end_date}
#     api_url = "http://predict_ann:8000/predict/arima"
#     response = requests.get(api_url, params=data_to_send)

#     if response.status_code != 200:
#         return {
#             "status_code": response.status_code,
#             "response_text": response.text
#         }

#     return response.json()


# @app.get("/predict/ETS")
# def predict_ets(stock_symbol: str, start_date: str, end_date: str):
#     data_to_send = {"stock_symbol": stock_symbol,
#                     "start_date": start_date,
#                     "end_date": end_date}
#     api_url = "http://predict_ann:8000/predict/ETS"
#     response = requests.get(api_url, params=data_to_send)

#     if response.status_code != 200:
#         return {
#             "status_code": response.status_code,
#             "response_text": response.text
#         }

#     return response.json()


# @app.get("/predict/historicAverage")
# def predict_historicAverage(stock_symbol: str, start_date: str, end_date: str):
#     data_to_send = {"stock_symbol": stock_symbol,
#                     "start_date": start_date,
#                     "end_date": end_date}
#     api_url = "http://predict_ann:8000/predict/historicAverage"
#     response = requests.get(api_url, params=data_to_send)

#     if response.status_code != 200:
#         return {
#             "status_code": response.status_code,
#             "response_text": response.text
#         }

#     return response.json()


# @app.get("/predict/theta")
# def predict_theta(stock_symbol: str, start_date: str, end_date: str):
#     data_to_send = {"stock_symbol": stock_symbol,
#                     "start_date": start_date,
#                     "end_date": end_date}
#     api_url = "http://predict_ann:8000/predict/theta"
#     response = requests.get(api_url, params=data_to_send)

#     if response.status_code != 200:
#         return {
#             "status_code": response.status_code,
#             "response_text": response.text
#         }

#     return response.json()


# @app.get("/predict/naive")
# def predict_naive(stock_symbol: str, start_date: str, end_date: str):
#     data_to_send = {"stock_symbol": stock_symbol,
#                     "start_date": start_date,
#                     "end_date": end_date}
#     api_url = "http://predict_ann:8000/predict/naive"
#     response = requests.get(api_url, params=data_to_send)

#     if response.status_code != 200:
#         return {
#             "status_code": response.status_code,
#             "response_text": response.text
#         }

#     return response.json()


# @app.get("/predict/windowAverage")
# def predict_windowAverage(stock_symbol: str, start_date: str, end_date: str):
#     data_to_send = {"stock_symbol": stock_symbol,
#                     "start_date": start_date,
#                     "end_date": end_date}
#     api_url = "http://predict_ann:8000/predict/windowAverage"
#     response = requests.get(api_url, params=data_to_send)

#     if response.status_code != 200:
#         return {
#             "status_code": response.status_code,
#             "response_text": response.text
#         }

#     return response.json()


# @app.get("/predict/linearRegression")
# def predict_linearRegression(stock_symbol: str, start_date: str, end_date: str):
#     data_to_send = {"stock_symbol": stock_symbol,
#                     "start_date": start_date,
#                     "end_date": end_date}
#     api_url = "http://predict_ann:8000/predict/linearRegression"
#     response = requests.get(api_url, params=data_to_send)

#     if response.status_code != 200:
#         return {
#             "status_code": response.status_code,
#             "response_text": response.text
#         }

#     return response.json()