from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.exceptions import add_exception_handlers
from app.routers.auth_router import auth_router
from app.routers.user_router import user_router
from app.core.config import settings
from app.core.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="DateVision AI API", lifespan=lifespan)
add_exception_handlers(app)

@app.get("/")
async def root():
    return {"message": "Welcome to DateVision AI API", "status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ROUTERS
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(user_router, prefix=settings.API_V1_STR)