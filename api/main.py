"""
FastAPI application factory with middleware, CORS, and request tracing.
"""

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.routers import (
    analysis,
    auth,
    cascades,
    comparison,
    connection,
    incidents,
    ingestion,
    metrics,
    monitors,
    simulation,
    system,
)
from api.utils.logging import configure_logging, get_logger

# Configure logging at module level
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    settings = get_settings()

    # Startup
    logger.info(
        "application_startup",
        version=app.version,
        environment=settings.intuit_env,
        dev_mode=settings.dev_mode,
    )

    # Create necessary directories
    import os

    os.makedirs("./data", exist_ok=True)
    os.makedirs("./mlruns", exist_ok=True)
    os.makedirs("./reports", exist_ok=True)

    logger.info("directories_created", paths=["./data", "./mlruns", "./reports"])

    yield

    # Shutdown
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    """
    Application factory.
    Creates and configures FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="LedgerGuard API",
        description="Business Reliability Engine - Principal-grade anomaly detection and RCA",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # Request tracing middleware
    @app.middleware("http")
    async def request_tracing_middleware(request: Request, call_next):
        """Add request ID and timing to all requests."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Bind request ID to structlog context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Log request
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None,
        )

        # Process request
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id

            # Log response
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
            )

            return response
        except Exception as e:
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Internal server error",
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )

    # Health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint for load balancers and monitoring."""
        return {
            "status": "healthy",
            "version": app.version,
            "environment": settings.intuit_env,
        }

    # Include routers
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(connection.router, prefix="/api/v1/connection", tags=["Connection"])
    app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["Ingestion"])
    app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
    app.include_router(incidents.router, prefix="/api/v1/incidents", tags=["Incidents"])
    app.include_router(cascades.router, prefix="/api/v1/cascades", tags=["Cascades"])
    app.include_router(monitors.router, prefix="/api/v1/monitors", tags=["Monitors"])
    app.include_router(comparison.router, prefix="/api/v1/comparison", tags=["Comparison"])
    app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["Simulation"])
    app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["Metrics"])
    app.include_router(system.router, prefix="/api/v1/system", tags=["System"])

    logger.info("application_configured", routers_count=11)

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
