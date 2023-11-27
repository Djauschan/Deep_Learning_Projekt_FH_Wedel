
from sqlalchemy.orm import Session
from sqlalchemy import MetaData, Table
from fastapi import HTTPException
import models, schemas, database
import bcrypt
from datetime import datetime
import socket
import datetime as DT
from yahoo_fin.stock_info import get_data

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
    users = db.query(models.User).filter(models.User.username.like(f"%{query}%")).limit(limit).all()
    return [u.__dict__ for u in users]

# method to create a user into table "users"
def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    db_user = models.User(email=user.email, password=hashed_password, username=user.username)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return schemas.User.from_orm(db_user).dict()

def update_user_by_username(db: Session, username: str, updated_data: schemas.UserUpdate):
    db_user = db.query(models.User).filter(models.User.username == username).first()
    
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
def create_login(db: Session, login: schemas.LoginCreate, owner_id: int, location: dict):
    now = datetime.now()
    current_date = now
    db_login = models.Login(
        login_time=current_date, 
        location=location,
        user_id=owner_id
    )
    db.add(db_login)
    db.commit()
    db.refresh(db_login)
    return db_login

# method to return all logins in table "login" (max. 100 entries)
def get_logins(db: Session, query: str = '', limit: int = 100):
    logins = db.query(models.Login).filter(models.Login.login_time.like(f"%{query}%")).limit(limit).all()
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

# method to add stock to table 'stocks'
def create_stock(db: Session, stock_name: str, ):
    today = DT.date.today()
    return_list = []
    if((today - DT.timedelta(days=301)).weekday() < 5):
        date = today - DT.timedelta(days=301)
        date2 = today - DT.timedelta(days=0) 
        stock = get_data(stock_name, start_date=date.strftime("%m/%d/%y"), end_date=date2.strftime("%m/%d/%y"), index_as_date = True, interval="1d")
        db_stock = models.Stock(name=stock_name, date=date.strftime("%m/%d/%y"), open=stock.open, high=stock.high, low=stock.low, close=stock.close)
        return_list.append(db_stock)
        print(stock)
        return return_list
    else:
        print("nope")
        return "it didnt worked"
    

# method to update stock data in table 'stocks'
        

# method to return all stocks in table 'stocks'
def get_stocks(db: Session):
    stocks = db.query(models.Stock).filter(models.Stock.name.like(f"%{''}%")).limit(100).all()
    return [u.__dict__ for u in stocks]