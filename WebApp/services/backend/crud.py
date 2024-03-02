from datetime import datetime

import bcrypt
import database
import models
import numpy as np
import pandas as pd
import schemas

# import pandas_market_calendars as mcal
# import yfinance as yf
# from yahoo_fin.stock_info import get_data
from fastapi import HTTPException
from sqlalchemy import MetaData, Table
from sqlalchemy.orm import Session


# method to delete the table "users"
def delete_users(db: Session):
    metadata = MetaData()
    table = Table("users", metadata, autoload_with=database.engine)
    table.drop(database.engine)
    metadata.create_all(database.engine)

# method to delete a single user in table "users"
def delete_user(db: Session, username: str):
    user = get_user_by_username(db, username)
    if not user:
        return None

    db.delete(user)
    db.commit()
    return "User deleted successfully"

# method to get a user from table "users" by id
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

# method to get a user from table "users" by email
def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

# method to get all data from table "users" (max. 100 entries)
def get_users(db: Session, query: str = '', limit: int = 100):
    users = db.query(models.User).filter(
        models.User.username.like(f"%{query}%")).limit(limit).all()
    return [u.__dict__ for u in users]

# method to create a user into table "users"
def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = bcrypt.hashpw(
        user.password.encode('utf-8'), bcrypt.gensalt())
    db_user = models.User(
        email=user.email, password=hashed_password, username=user.username)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return schemas.User.from_orm(db_user).dict()


def update_user_by_username(db: Session, username: str, updated_data: schemas.UserUpdate):
    db_user = db.query(models.User).filter(
        models.User.username == username).first()

    if db_user:
        if updated_data.username:
            db_user.username = updated_data.username
        if updated_data.email:
            db_user.email = updated_data.email

        db.commit()
        db.refresh(db_user)
        return db_user
    else:
        raise HTTPException(status_code=404, detail="User not found")

# method to create a login entry into table "login"
def create_login(db: Session, owner_id: int):
    now = datetime.now()
    current_date = now
    db_login = models.Login(
        login_time=current_date,
        user_id=owner_id
    )
    db.add(db_login)
    db.commit()
    db.refresh(db_login)
    return db_login

# method to return all logins in table "login" (max. 100 entries)
def get_logins(db: Session, query: str = '', limit: int = 100):
    logins = db.query(models.Login).filter(
        models.Login.login_time.like(f"%{query}%")).limit(limit).all()
    return [u.__dict__ for u in logins]

# method to return all logins of a user in table "login"
def get_logins_by_user_id(db: Session, owner_id: int):
    return db.query(models.Login).filter(models.Login.user_id == owner_id).all()

# method to validate login
def check_login(db: Session, user: schemas.User, pw: str):
    pass

# method to return login by userid of a user in table "login"
def get_logins_by_user_id(db: Session, owner_id: int):
    return db.query(models.Login).filter(models.Login.user_id == owner_id).all()

#method to update budget of a user in table "users"
def update_budget_by_user(db: Session, username: str, new_budget: int):
    db_user = db.query(models.User).filter(
        models.User.username == username).first()

    if db_user:
        db_user.budget += new_budget

        db.commit()
        db.refresh(db_user)
        return db_user
    else:
        raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")


# method to get stock data of specific stock symbol for n days
# def get_stock_days(db: Session, stock_symbol: str, n: int):
#     # Download historical data from Yahoo Finance
#     stock_data = yf.download(stock_symbol, period=f"{n}d")
#     stock_data["Volume"] = stock_data["Volume"].astype(float)
#     return_data = []

#     for index, row in stock_data.iterrows():
#         return_data.append({"date": index.strftime(
#             '%m/%d/%y'), "open": row["Open"], "high": row["High"], "low": row["Low"], "close": row["Close"], "volume": row["Volume"]})

#     return return_data
    

# method to load data from csv file
def loadDataFromFile(start_date: pd.Timestamp, end_date: pd.Timestamp, rsc_completePath: str,
                     ALL_DATA_COLUMNS: list, COLUMNS_TO_KEEP: list) -> pd.DataFrame:

    df = pd.read_csv(rsc_completePath, sep=",",
                     names=ALL_DATA_COLUMNS, index_col=False)

    toRemove = []
    for col in df:
        if col not in COLUMNS_TO_KEEP:

            toRemove.append(col)

    data = df.drop(toRemove, axis=1)

    data['DateTime'] = pd.to_datetime(data['DateTime'])

    data = data[(data['DateTime'] >= start_date)
                & (data['DateTime'] <= end_date)]

    # Set 'DateTime' as the index (required for resampling)
    data.set_index('DateTime', inplace=True)

    # Resample to 30-minute intervals
    data = data.resample('30T').mean()

    # Replace non-compliant values with a compliant value (e.g., None)
    data.replace([np.inf, -np.inf, np.nan], None, inplace=True)

    # Filter out data from 20:00 to 04:00
    data = data[(data.index.hour < 20) & (data.index.hour >= 4)]

    # Reset the index
    data.reset_index(inplace=True)

    # Convert the DataFrame to a list of dictionaries
    return_data = data.to_dict(orient='records')

    return return_data
