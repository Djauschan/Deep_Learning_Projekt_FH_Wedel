from pydantic import BaseModel
from typing import Optional



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