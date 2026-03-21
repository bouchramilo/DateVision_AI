from pydantic import model_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # General
    PROJECT_NAME: str = Field(default="DateVision API", validation_alias="PROJECT_NAME")
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "dev"
    
    # Security - ces champs sont requis
    SECRET_KEY: str = Field(...) 
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 11520
    
    # MODELS
    MODEL_CLASSIFICATION_BY_VARIETY: str = "model_classes_variete_googlenet_model.pth"
    MODEL_CLASSIFICATION_BY_MATURITY: str = "model_classes_maturity_googlenet_model.pth"
    MODEL_DETECTION: str = "date_detector_model.pt"
    
    # =========================
    # LLM CONFIG
    # =========================
    LLM_PROVIDER: str = "ollama"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"
    
    # Database Configuration - requis
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "DateVision_DB"
    DATABASE_URL: str | None = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra="ignore"
    )

    @model_validator(mode="after")
    def assemble_db_connection(self):
        """Construit DATABASE_URL si non fournie"""
        if not self.DATABASE_URL:
            self.DATABASE_URL = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        return self
    
    
settings = Settings()