from pydantic import BaseModel

class UserBase(BaseModel):
    email: str
    username: str
    password: str
    budget: float
    class Config:
        from_attributes = True

class User(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True

class UserCreate(UserBase):
    pass

class UserUpdate(BaseModel):
    username: str
    email: str
    budget: float
    
class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str

class LoginBase(BaseModel):
    pass

class LoginCreate(LoginBase):
    pass

class Login(LoginBase):
    id: int
    user_id: int
    login_time: str

    class Config:
        orm_mode = True
        
class LoginRequest(BaseModel):
    user: str
    password: str

    class Config:
        orm_mode = True