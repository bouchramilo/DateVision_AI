from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.schemas.user_schema import UserCreate, UserLogin, UserLoginResponse, User as UserSchema
from app.core.database import get_db
from app.repositories.user_repository import check_user_existe, create_user, get_user_by_email, update_activation_user
from app.core.logger import logger
from app.security.hashing import verify_password
from app.security.manage_token import create_access_token
from app.core.deps import get_current_user

from app.core.metrics import (
    AUTH_LOGIN_TOTAL,
    AUTH_FAILED_TOTAL,
    AUTH_TOKEN_EXPIRED,
    SECURITY_SUSPICIOUS_REQUESTS,
    API_ERRORS_TOTAL,
)


# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
auth_router = APIRouter(prefix="/auth", tags=["auth"])



# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# REGISTER USER
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@auth_router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserSchema)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)) -> UserSchema:
    try:
        # Vérifier si l'utilisateur existe déjà
        db_user = check_user_existe(db=db, email=user_data.email)

        if db_user:
            logger.error(f"❌ Tentative d'inscription avec email existant: {user_data.email}")
            SECURITY_SUSPICIOUS_REQUESTS.inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User déjà existant avec cet email",
            )

        new_user = create_user(db=db, user=user_data)

        # Convertir l'objet SQLAlchemy en modèle Pydantic pour la réponse
        user_response = UserSchema.model_validate(new_user)

        logger.info(f"✅ Utilisateur créé avec succès: {new_user.username}")
        return user_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌❌ Erreur lors de l'inscription de l'utilisateur : {e} ❌")
        API_ERRORS_TOTAL.labels(
            endpoint="/auth/register",
            error_type=type(e).__name__,
        ).inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'inscription de l'utilisateur"
        )
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# LOGIN USER
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@auth_router.post("/login")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        user = get_user_by_email(db, email=form_data.username)

        if not user or not verify_password(form_data.password, user.password_hash):
            # ── Failed login ──────────────────────────────────────────────
            AUTH_FAILED_TOTAL.inc()
            # ─────────────────────────────────────────────────────────────
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        logger.info(f"☑️☑️ User {form_data.username} logging in... ☑️☑️")
        user_acivate = update_activation_user(db, user, is_active=True)
        logger.info(f"☑️☑️ Login activation result for {user.username}: {getattr(user_acivate, 'is_active', 'None')} ☑️☑️")

        access_token = create_access_token(subject=user.email)

        # ── Successful login ──────────────────────────────────────────────
        AUTH_LOGIN_TOTAL.inc()
        # ─────────────────────────────────────────────────────────────────

        return {"access_token": access_token, "token_type": "bearer"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        API_ERRORS_TOTAL.labels(
            endpoint="/auth/login",
            error_type=type(e).__name__,
        ).inc()
        raise HTTPException(status_code=500, detail=f"Error lors la login : {e}")
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# GET ME (user)
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@auth_router.get("/me")
def get_me(current_user = Depends(get_current_user)):
    return current_user