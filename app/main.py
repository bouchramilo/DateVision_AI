from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.exceptions import add_exception_handlers
from app.routers.auth_router import auth_router
from app.routers.user_router import user_router
from app.routers.admin_router import admin_router
from app.routers.pridect_router import pridect_router
from app.core.config import settings
from app.core.database import init_db
from fastapi.middleware.cors import CORSMiddleware
from app.services.model_loader import load_all_models
from fastapi import Request
import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from app.core.metrics import (
    API_REQUESTS_TOTAL,
    API_REQUEST_LATENCY,
    API_REQUESTS_IN_PROGRESS
)

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Initialisation de l'application...")

    init_db()

    # charger les modèles
    load_all_models()

    print("✅ Application prête !")

    yield

    print("🛑 Shutdown application")


# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

add_exception_handlers(app)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
EXCLUDED_PATHS = {
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/health",
    "/"
}

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    path = request.url.path

    # Ignorer certains endpoints
    if path in EXCLUDED_PATHS:
        return await call_next(request)

    API_REQUESTS_IN_PROGRESS.inc()

    start_time = time.time()

    try:
        response = await call_next(request)

        duration = time.time() - start_time

        API_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=path,
            status=str(response.status_code)
        ).inc()

        API_REQUEST_LATENCY.labels(
            endpoint=path
        ).observe(duration)

        return response

    except Exception as e:
        API_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=path,
            status="500"
        ).inc()
        raise e

    finally:
        API_REQUESTS_IN_PROGRESS.dec()


# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ROUTERS
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@app.get("/")
async def root():
    return {"message": "Welcome to DateVision AI API", "status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(user_router, prefix=settings.API_V1_STR)
app.include_router(admin_router, prefix=settings.API_V1_STR)
app.include_router(pridect_router, prefix=settings.API_V1_STR)