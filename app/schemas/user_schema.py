from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, ConfigDict, model_validator
from typing_extensions import Self


# =========================================================
# 🟢 BASE USER
# =========================================================
class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    is_active: Optional[bool] = True
    role: Optional[str] = "USER"
    password: Optional[str] = None
    password_repeat: Optional[str] = None

    @model_validator(mode='after')
    def check_passwords_match(self) -> Self:
        if self.password or self.password_repeat:
            if self.password != self.password_repeat:
                raise ValueError("Passwords do not match")
        return self


# =========================================================
# 🟢 CREATE USER
# =========================================================
class UserCreate(UserBase):
    email: EmailStr = Field(..., description="Email de l'utilisateur")
    username: str = Field(..., min_length=3, description="Nom d'utilisateur")
    password: str = Field(..., min_length=6, description="Mot de passe")
    password_repeat: str = Field(..., description="Confirmation du mot de passe")



# =========================================================
# 🟢 UPDATE USER
# =========================================================
class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[str] = None
    password: Optional[str] = Field(None, min_length=6)
    password_repeat: Optional[str] = None

    @model_validator(mode='after')
    def validate_password_update(self) -> Self:
        if self.password or self.password_repeat:
            if not self.password or not self.password_repeat:
                raise ValueError("Both password and password confirmation are required")
            if self.password != self.password_repeat:
                raise ValueError("Passwords do not match")
        return self


# =========================================================
# 🟢 DATABASE MODEL
# =========================================================
class UserInDBBase(BaseModel):
    id: int
    email: EmailStr
    username: str
    is_active: bool
    role: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class User(UserInDBBase):
    pass


# =========================================================
# 🟢 LOGIN
# =========================================================
class UserLogin(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    password: str = Field(..., min_length=6)

    @model_validator(mode='after')
    def validate_credentials(self) -> Self:
        if not self.email and not self.username:
            raise ValueError("Either email or username must be provided")
        return self


class UserLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User


# =========================================================
# 🟢 CHANGE PASSWORD
# =========================================================
class UserChangePassword(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)
    new_password_repeat: str

    @model_validator(mode='after')
    def validate_passwords(self) -> Self:
        if self.new_password != self.new_password_repeat:
            raise ValueError("New passwords do not match")

        if self.current_password == self.new_password:
            raise ValueError("New password must be different from current password")

        return self