"""
Configuration management using pydantic-settings.
All settings loaded from environment variables (12-factor app).
"""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # QuickBooks OAuth2
    intuit_client_id: str = Field(default="", description="Intuit OAuth2 client ID")
    intuit_client_secret: str = Field(default="", description="Intuit OAuth2 client secret")
    intuit_redirect_uri: str = Field(
        default="http://localhost:8000/api/v1/auth/callback",
        description="OAuth2 redirect URI",
    )
    intuit_env: str = Field(default="sandbox", description="Intuit environment (sandbox|production)")
    intuit_realm_id: str = Field(default="", description="QuickBooks company ID")

    # Security
    jwt_secret: str = Field(
        default="change-this-to-a-secure-random-string-in-production",
        description="JWT signing secret",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(default=1440, description="JWT expiration (24h)")

    # Database
    db_type: str = Field(default="duckdb", description="Database type")
    db_path: str = Field(default="./data/bre.duckdb", description="DuckDB file path")
    db_memory_limit: str = Field(default="4GB", description="DuckDB memory limit")
    db_threads: int = Field(default=4, description="DuckDB thread count")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_max_connections: int = Field(default=50, description="Redis pool size")

    # API
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="Uvicorn workers")
    api_reload: bool = Field(default=True, description="Enable hot reload")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="CORS allowed origins (comma-separated)",
    )

    # Logging
    log_level: str = Field(default="info", description="Log level")
    log_format: str = Field(default="json", description="Log format (json|console)")
    structlog_renderer: str = Field(default="console", description="Structlog renderer")

    # Celery
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1", description="Celery broker URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2", description="Celery result backend"
    )
    celery_task_serializer: str = Field(default="json", description="Task serializer")
    celery_result_serializer: str = Field(default="json", description="Result serializer")
    celery_accept_content: str = Field(default="json", description="Accepted content types")

    # MLflow
    mlflow_tracking_uri: str = Field(default="./mlruns", description="MLflow tracking URI")
    mlflow_experiment_name: str = Field(
        default="ledgerguard-bre", description="MLflow experiment name"
    )

    # Engine Configuration
    anomaly_detection_sensitivity: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Anomaly detection sensitivity"
    )
    min_confidence_threshold: float = Field(
        default=0.70, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    blast_radius_max_depth: int = Field(
        default=5, ge=1, le=10, description="Max blast radius traversal depth"
    )
    monitor_evaluation_interval_seconds: int = Field(
        default=300, ge=60, description="Monitor evaluation interval"
    )

    # Feature Flags
    enable_ml_training: bool = Field(default=True, description="Enable ML model training")
    enable_auto_remediation: bool = Field(default=False, description="Enable auto-remediation")
    enable_realtime_analysis: bool = Field(default=True, description="Enable realtime analysis")
    enable_supplemental_data: bool = Field(
        default=True, description="Enable supplemental data adapters"
    )

    # Reporting
    report_output_dir: str = Field(default="./reports", description="Report output directory")
    report_logo_path: str = Field(default="./assets/logo.png", description="Report logo path")

    # Development
    dev_mode: bool = Field(default=True, description="Development mode")
    debug: bool = Field(default=False, description="Debug mode")
    testing: bool = Field(default=False, description="Testing mode")

    @field_validator("cors_origins")
    @classmethod
    def parse_cors_origins(cls, v: str) -> List[str]:
        """Parse comma-separated CORS origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def intuit_auth_url(self) -> str:
        """Construct Intuit authorization URL."""
        base = (
            "https://appcenter.intuit.com/connect/oauth2"
            if self.intuit_env == "production"
            else "https://appcenter.intuit.com/connect/oauth2"
        )
        return base

    @property
    def intuit_token_url(self) -> str:
        """Construct Intuit token URL."""
        return "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"

    @property
    def intuit_api_base_url(self) -> str:
        """Construct QuickBooks API base URL."""
        if self.intuit_env == "production":
            return "https://quickbooks.api.intuit.com"
        return "https://sandbox-quickbooks.api.intuit.com"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure singleton behavior.
    """
    return Settings()
