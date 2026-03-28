from pydantic import model_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# =========================================================
# GLOBAL SETTINGS
# =========================================================

class Settings(BaseSettings):
    """
    Configuration centralisée de l'application via Pydantic Settings.
    Gère les variables d'environnement et les valeurs par défaut.
    """
    
    # --- General ---
    PROJECT_NAME: str = Field(default="DateVision API", validation_alias="PROJECT_NAME")
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "dev"
    
    # --- Security ---
    SECRET_KEY: str = Field(...) 
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 11520
    
    # --- AI Models ---
    MODEL_CLASSIFICATION_BY_VARIETY: str = "a_model_classes_variete_googlenet_model.pth"
    MODEL_CLASSIFICATION_BY_MATURITY: str = "a_model_classes_maturity_googlenet_model.pth"
    MODEL_DETECTION: str = "a_date_detector_model.pt"
    
    # --- LLM Config (Ollama) ---
    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"
    
    # --- Database (PostgreSQL) ---
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
        """Construit dynamiquement DATABASE_URL si non spécifiée."""
        if not self.DATABASE_URL:
            self.DATABASE_URL = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        return self

# Singleton instance
settings = Settings()