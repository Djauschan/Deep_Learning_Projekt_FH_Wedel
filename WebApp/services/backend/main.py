import bcrypt
import crud
import models
import pandas as pd
import requests
import schemas
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
    "http://localhost:8080",
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


@app.get("/get_user/{email}")
async def get_user(email: str, db: Session = Depends(get_db)):
    return crud.get_user_by_email(db, email)


@app.put("/update_user/{username}")
async def update_user(username: str, user_update: schemas.UserUpdate, db: Session = Depends(get_db)):
    updated_user = crud.update_user_by_username(db, username, user_update)
    return {"message": "User updated successfully"}


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


@app.get("/predict/transformer")
def predict_transformer(stock_symbols: str):
    data_to_send = {"stock_symbols": stock_symbols,
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-05",
                    "resolution": 'H'}
    api_url = "http://predict_transformer:8000/predict"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/cnn")
def predict_cnn():
    data_to_send = {"stock_symbol": "AAPL",
                    "start_date": "2021-02-01",
                    "end_date": "2021-03-03"}
    api_url = "http://predict_cnn:8000/predict"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


# @app.get("/predict/lstm")
# def predict_lstm(stock_symbol: str):
#     lstm = LstmInterface()
#     return lstm.predict('2021-01-04', '2021-01-06', 120)


@app.get("/predict/arima")
def predict_arima(stock_symbol: str):
    data_to_send = {"stock_symbol": stock_symbol,
                    "start_date": "2021-01-04",
                    "end_date": "2021-01-05"}
    api_url = "http://predict_ann:8000/predict/arima"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/ETS")
def predict_ets(stock_symbol: str):
    data_to_send = {"stock_symbol": stock_symbol,
                    "start_date": "2021-01-04",
                    "end_date": "2021-01-05"}
    api_url = "http://predict_ann:8000/predict/ETS"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/historicAverage")
def predict_historicAverage(stock_symbol: str):
    data_to_send = {"stock_symbol": stock_symbol,
                    "start_date": "2021-01-04",
                    "end_date": "2021-01-05"}
    api_url = "http://predict_ann:8000/predict/historicAverage"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/theta")
def predict_theta(stock_symbol: str):
    data_to_send = {"stock_symbol": stock_symbol,
                    "start_date": "2021-01-04",
                    "end_date": "2021-01-05"}
    api_url = "http://predict_ann:8000/predict/theta"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/naive")
def predict_naive(stock_symbol: str):
    data_to_send = {"stock_symbol": stock_symbol,
                    "start_date": "2021-01-04",
                    "end_date": "2021-01-05"}
    api_url = "http://predict_ann:8000/predict/naive"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/windowAverage")
def predict_windowAverage(stock_symbol: str):
    data_to_send = {"stock_symbol": stock_symbol,
                    "start_date": "2021-01-04",
                    "end_date": "2021-01-05"}
    api_url = "http://predict_ann:8000/predict/windowAverage"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/linearRegression")
def predict_linearRegression(stock_symbol: str):
    data_to_send = {"stock_symbol": stock_symbol,
                    "start_date": "2021-01-04",
                    "end_date": "2021-01-05"}
    api_url = "http://predict_ann:8000/predict/linearRegression"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/randomForest")
def predict_randomForest(stock_symbol: str):
    data_to_send = {"stock_symbol": stock_symbol,
                    "start_date": "2021-01-04",
                    "end_date": "2021-01-05"}
    api_url = "http://predict_ann:8000/predict/randomForest"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/gradientBoost")
def predict_gradientBoost(stock_symbol: str):
    data_to_send = {"stock_symbol": stock_symbol,
                    "start_date": "2021-01-04",
                    "end_date": "2021-01-05"}
    api_url = "http://predict_ann:8000/predict/gradientBoost"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/svm")
def predict_svm(stock_symbol: str):
    data_to_send = {"stock_symbol": stock_symbol,
                    "start_date": "2021-01-04",
                    "end_date": "2021-01-05"}
    api_url = "http://predict_ann:8000/predict/svm"
    response = requests.get(api_url, params=data_to_send)

    if response.status_code != 200:
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }

    return response.json()


@app.get("/predict/rl")
def predict_rl(stock_symbol: str  # , start_date: str, end_date: str
               ):
    """Predicts trading actions for a given stock symbol and time frame with every avialible model.

    Args:
        stock_symbol (str): The stock symbol to predict trading actions for.
        start_date (str): The start date of the time frame to predict trading actions for.
        end_date (str): The end date of the time frame to predict trading actions for.

    Returns:
        list[dict[Timestamp, dict[str, str]]]: A list containing the predictions for every model for every hour in the given time frame.
    """
    # data_to_send = {"stock_symbol": stock_symbol,
    #                 "start_date": "2021-01-04",
    #                 "end_date": "2021-01-05"}
    # api_url = "http://predict_rl:8000/predict"
    # response = requests.get(api_url, params=data_to_send)
    # if response.status_code != 200:
    #     return {
    #         "status_code": response.status_code,
    #         "response_text": response.text
    #     }
    # return response.json()
    from random import choice
    posible_actions = ['buy', 'sell', 'hold']
    model_names = ["q_learning_ma5", "q_learning_ma30", "q_learning_ma200",
                   "q_learning_transformer", "q_learning_cnn", "q_learning_arima"]

    ensemble = choice(["election", "ensemble"]) == 'ensemble'
    random_return = {}
    for current_date in pd.date_range("2021-01-04", "2021-01-05", freq='h'):
        random_return[current_date] = {model: choice(
            posible_actions) for model in model_names}
        if ensemble:
            random_return[current_date]['ensemble'] = choice(
                list(random_return[current_date].values()))
        else:
            lst = list(random_return[current_date].values())
            random_return[current_date]['election'] = max(
                set(lst), key=lst.count)

    return [random_return]


@app.get("/load/data")
def load_data():
    allColumns = ["DateTime", "Open", "Close", "High", "Low", "a"]
    relevantColumns = ["DateTime", "Open", "Close", "High", "Low"]
    start_date = pd.Timestamp("2021-01-04")
    end_date = pd.Timestamp("2021-01-06")

    return crud.loadDataFromFile(start_date=start_date, end_date=end_date, rsc_completePath="../data/Aktien/AAPL_1min.txt",
                                 ALL_DATA_COLUMNS=allColumns, COLUMNS_TO_KEEP=relevantColumns)
