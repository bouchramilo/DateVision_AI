from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from app.schemas.user_schema import UserCreate, UserLogin, UserLoginResponse, User
from app.core.database import get_db
from app.repositories.user_repository import get_all_users
from app.core.logger import logger
from app.core.deps import get_current_admin_user

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
user_router = APIRouter(prefix="/users", tags=["users"])

