from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.schemas.user_schema import UserCreate, UserLogin, UserLoginResponse
from app.core.database import get_db
from app.repositories.user_repository import check_user_existe, create_user, get_user_by_email, update_activation_user
from app.core.logger import logger
from app.security.hashing import verify_password
from app.security.manage_token import create_access_token


# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
auth_router = APIRouter(prefix="/auth", tags=["auth"])



# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# REGISTER USER
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@auth_router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)) -> dict:
    try:
        db_user = check_user_existe(
            db=db,
            email=user_data.email,
        )

        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User déjà existant avec cet email",
            )

        return create_user(db=db, user= user_data)
    except Exception as e:
        logger.error(f"❌❌ Erreur lors de l'inscription de l'utilisateur : {e} ❌")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'inscription de l'utilisateur : {e}")
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# LOGIN USER
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@auth_router.post("/login")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        user = get_user_by_email(db, email=form_data.username)
        if not user or not verify_password(form_data.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        logger.info(f"☑️☑️ User {form_data.username} logging in... ☑️☑️")
        user_acivate = update_activation_user(db, user, is_active=True)
        logger.info(f"☑️☑️ Login activation result for {user.username}: {getattr(user_acivate, 'is_active', 'None')} ☑️☑️")
        access_token = create_access_token(subject=user.email)
        return {"access_token": access_token, "token_type": "bearer"}

    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(status_code=501, detail=f"Error lors la login : {e}")
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
