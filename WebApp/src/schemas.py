from pydantic import BaseModel
from typing import Optional

class StockBase(BaseModel):
    name: str
    date = str
    open = float
    high = float
    low = float
    close = float
    volume = str
    class Config:
        orm_mode = True
        from_attributes = True

class Stock(StockBase):
    id: int
    
class StockCreate(StockBase):
    pass 

class UserBase(BaseModel):
    email: str
    username: str
    password: str
    class Config:
        orm_mode = True
        from_attributes = True

class User(UserBase):
    id: int
    is_active: bool

class UserCreate(UserBase):
    pass

class UserUpdate(BaseModel):
    username: str
    email: str
    
class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str

class LoginBase(BaseModel):
    login_time: str
    ip: str
    location: str

class LoginCreate(LoginBase):
    pass

class Login(LoginBase):
    id: int
    user_id: int

    class Config:
        orm_mode = True
        
class LoginRequest(BaseModel):
    user: str
    password: str
    location: Optional[dict]