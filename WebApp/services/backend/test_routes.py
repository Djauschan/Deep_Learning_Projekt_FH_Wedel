from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import app, get_db
import models

SQLALCHEMY_DATABASE_URL = "sqlite:///./test_sql_app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine
)

# Create tables in test database
models.Base.metadata.create_all(bind=engine)

client = TestClient(app)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        
app.dependency_overrides[get_db] = override_get_db 
# tries to create a testuser
def test_create_user():
    response = client.post(
        "/createUser/",
        json={
            "username": "testuser",
            "email": "test",
            "password": "testpassword",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test"

# tries to get the testuser from the db
def test_get_user():
    response = client.get("/get_user/testuser")
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test"
    
# tries to delete the testuser
def test_delete_user():
    response = client.post("/deleteUser/testuser")
    assert response.status_code == 200

