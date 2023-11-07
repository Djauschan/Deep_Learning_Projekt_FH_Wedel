from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# create a database url for sqlalchemy
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# create sqlalchemy engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# SessionLocal class to allow db access via session 
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class to create the ORM (object-relational mapping) models
Base = declarative_base()