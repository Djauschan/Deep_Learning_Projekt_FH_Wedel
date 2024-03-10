from pydantic import BaseModel

class UserBase(BaseModel):
    email: str
    username: str
    password: str

    class Config:
        from_attributes = True
        orm_mode = True

class User(UserBase):
    id: int
    is_active: bool
    budget: float

    class Config:
        from_attributes = True
        orm_mode = True

class UserCreate(UserBase):
    pass

class UserUpdate(BaseModel):
    username: str
    email: str
    budget: float

    class Config:
        from_attributes = True
        orm_mode = True
    
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
        from_attributes = True
        
class LoginRequest(BaseModel):
    user: str
    password: str

    class Config:
        orm_mode = True
        from_attributes = True