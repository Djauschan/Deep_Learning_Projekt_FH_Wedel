from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
import crud, models, schemas
from database import SessionLocal, engine
import bcrypt
from fastapi.middleware.cors import CORSMiddleware

# Create tables in the database based on the model definitions
models.Base.metadata.create_all(bind=engine)

# Create instance of FastAPI
app = FastAPI()


# Configure CORS settings
origins = ["http://localhost:8080"]  # Replace with the origins you want to allow

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

# post method to delete table "users"
@app.post("/deleteUsers/")
def delete_users(db: Session = Depends(get_db)):
    db_delete_users = crud.delete_users(db)
    if db_delete_users:
        raise HTTPException(status_code=400, detail="Not Found")
    #return crud.get_users(db=db, skip=0, limit=100)

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
    db_user_username = crud.get_user_by_username(db, username = user.username)
    if db_user_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    if db_user_username:
        raise HTTPException(status_code=400, detail="Username already registered")
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
        raise HTTPException(status_code=400, detail="Email does not exist")

    if not bcrypt.checkpw(login_request.password.encode('utf-8'), user.password):
        raise HTTPException(status_code=400, detail="Incorrect password")
    print(login_request.location)
    # Creating a login record after successful authentication
    login_record = crud.create_login(db, user.id, user.id, login_request.location)  # Removed schemas.LoginCreate()

    if not login_record:
        raise HTTPException(status_code=500, detail="Could not create login record")
    
    return {
        "username": user.username,
        "email": user.email,
        "id": user.id,
        "is_active": user.is_active
    }

@app.get("/getLogins/", response_model=list[schemas.Login])
def read_logins(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    logins = crud.get_logins(db, skip=skip, limit=limit)
    return logins 

@app.get("/getLoginByUser_ID/", response_model=list[schemas.Login])
def read_logins(user_id: int, db: Session = Depends(get_db)):
    logins = crud.get_logins_by_user_id(db, owner_id=user_id)
    return logins

@app.get("/login/validation")
def check_login(email: str, password: str, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, email)
    
    if user:
        raise HTTPException(status_code=400, detail="Email does not exist")
