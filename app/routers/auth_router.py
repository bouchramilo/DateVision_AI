from fastapi import APIRouter
from app.schemas.user_schema import UserLogin



auth_router = APIRouter(prefix="/auth", tags=["/auth"])

@auth_router.post("/login", response_model=UserLogin)
def login():
    pass