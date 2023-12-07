from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, LargeBinary, UniqueConstraint, JSON, Float
from sqlalchemy.orm import relationship

# import Base to create classes that inherit from it
from database import Base

# Class to create db table named "users"
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    is_active = Column(Boolean, default=True)

# Class for sqlalchemy to create db table named "login"
class Login(Base):
    __tablename__="login"

    id = Column(Integer, primary_key=True, index=True)
    ip = Column(String, index=True)
    location = Column(JSON, index=True)
    login_time = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))

# Class for sqlalchemy to create db table named "stocks"
class Stock(Base):
    __tablename__="stocks"

    name = Column(String, primary_key=True)
    start_date = Column(String, primary_key=True)
    end_date = Column(String, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)