from fastapi import FastAPI

from app.routers.auth_router import auth_router

app = FastAPI(title="DateVision AI API")

@app.get("/")
async def root():
    return {"message": "Welcome to DateVision AI API", "status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
