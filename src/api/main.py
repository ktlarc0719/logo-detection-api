import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from src.core.config import get_settings
from src.core.detection_engine import initialize_detection_engine
from src.core.training_engine import initialize_training_engine
from src.utils.logger import setup_logger, get_logger
from src.utils.metrics import performance_monitor
from src.models.schemas import ErrorResponse, ValidationErrorResponse

# Import routers
from src.api.endpoints import health, model_management, training, logo_management, download_images, system_management, inspection_v2, ml_system
from src.api.endpoints.dataset_management import router as dataset_management_router

# Setup logging
logger = setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    settings = get_settings()
    
    # Startup
    logger.info("Starting Logo Detection API...")
    
    try:
        # Initialize detection engine
        logger.info("Initializing detection engine...")
        initialize_detection_engine()
        logger.info("Detection engine initialized successfully")
        
        # Initialize training engine if enabled
        logger.info("Initializing training engine...")
        training_engine = initialize_training_engine()
        if training_engine:
            logger.info("Training engine initialized successfully")
        else:
            logger.info("Training pipeline disabled")
        
        # Start inspection queue
        from src.core.inspection_queue import get_inspection_queue
        inspection_queue = get_inspection_queue()
        await inspection_queue.start()
        logger.info("Inspection queue started")
        
        # Start background monitoring task
        if settings.enable_metrics:
            monitoring_task = asyncio.create_task(performance_monitor())
            logger.info("Performance monitoring started")
        else:
            monitoring_task = None
        
        logger.info("Logo Detection API started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
    
    # Shutdown
    logger.info("Shutting down Logo Detection API...")
    
    # Stop inspection queue
    await inspection_queue.stop()
    logger.info("Inspection queue stopped")
    
    # Cancel monitoring task
    if monitoring_task and not monitoring_task.done():
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
    
    logger.info("Logo Detection API shutdown completed")


# Create FastAPI application
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Logo Detection API",
        description="High-performance logo detection API using YOLOv8 with support for batch processing",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    setup_middleware(app, settings)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    setup_routers(app)
    
    # Mount static files
    import os
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
    if os.path.exists("templates"):
        app.mount("/assets", StaticFiles(directory="templates"), name="templates")
    
    return app


def setup_middleware(app: FastAPI, settings) -> None:
    """Setup middleware for the application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Gzip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} "
            f"({process_time:.4f}s) {request.method} {request.url}"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    # Request size limiting middleware
    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        
        if content_length:
            content_length = int(content_length)
            if content_length > settings.max_request_size:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request too large",
                        "max_size_bytes": settings.max_request_size,
                        "received_size_bytes": content_length
                    }
                )
        
        return await call_next(request)


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers."""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors."""
        logger.warning(f"Validation error for {request.url}: {exc.errors()}")
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation failed",
                "details": exc.errors(),
                "timestamp": time.time()  # datetimeではなくtimestamp使用
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        logger.warning(f"HTTP exception {exc.status_code} for {request.url}: {exc.detail}")
        
        # Return the original status code, not 500
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "error_type": "http_error",
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected errors."""
        logger.error(f"Unexpected error for {request.url}: {str(exc)}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                error_type="internal_error"
            ).dict()
        )


def setup_routers(app: FastAPI) -> None:
    """Setup API routers."""
    
    # Health and monitoring endpoints
    app.include_router(
        health.router,
        prefix="/api/v1",
        tags=["Health & Monitoring"]
    )
    
    
    
    # Model and brand management endpoints
    app.include_router(
        model_management.router,
        prefix="/api/v1",
        tags=["Model Management", "Brand Management"]
    )
    
    # Training pipeline endpoints
    app.include_router(
        training.router,
        prefix="/api/v1/training",
        tags=["Training Pipeline", "Dataset Management"]
    )
    
    # Logo management endpoints
    app.include_router(
        logo_management.router,
        prefix="/api/v1",
        tags=["Logo Management", "Custom Models"]
    )
    
    # Download images endpoint
    app.include_router(
        download_images.router,
        prefix="/api/v1",
        tags=["Image Download"]
    )
    
    # Dataset management endpoints
    app.include_router(
        dataset_management_router,
        prefix="/api/v1",
        tags=["Dataset Management"]
    )
    
    # System management endpoints
    app.include_router(
        system_management.router,
        prefix="/api/v1",
        tags=["System Management"]
    )
    
    
    # Inspection endpoints
    app.include_router(
        inspection_v2.router,
        prefix="/api/v1",
        tags=["Inspection"]
    )
    
    # ML System endpoints
    app.include_router(
        ml_system.router,
        prefix="/api/v1/ml",
        tags=["ML System"]
    )
    
    # Serve static files and UI
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # UI endpoints
    @app.get("/", tags=["UI"])
    async def serve_dashboard():
        """Serve the main dashboard."""
        return FileResponse('templates/dashboard.html')
    
    
    
    @app.get("/ui/inspection", tags=["UI"])
    async def serve_inspection_ui():
        """Serve the inspection management UI."""
        return FileResponse('templates/inspection_ui.html')
    
    @app.get("/ui/ml", tags=["UI"])
    async def serve_ml_system_ui():
        """Serve the ML system UI."""
        return FileResponse('templates/ml_system_ui.html')
    
    # API Info endpoint
    @app.get("/api", tags=["API Info"])
    async def api_info():
        """Root endpoint with API information."""
        return {
            "name": "Logo Detection API",
            "version": "3.0.0",
            "description": "High-performance logo detection API with training pipeline and brand classification",
            "features": [
                "Multi-model detection (general, trademark, custom)",
                "Brand name normalization (Japanese/English)",
                "Category classification",
                "Dynamic model switching",
                "High-performance batch processing",
                "Training pipeline for custom models",
                "Dataset management and generation",
                "Logo upload and management",
                "Real-time training monitoring"
            ],
            "endpoints": {
                "health": "/api/v1/health",
                "metrics": "/api/v1/metrics",
                "models": "/api/v1/models",
                "model_switch": "/api/v1/models/switch",
                "brands": "/api/v1/brands",
                "categories": "/api/v1/categories",
                "training_status": "/api/v1/training/status",
                "training_start": "/api/v1/training/start",
                "datasets": "/api/v1/training/datasets",
                "logo_classes": "/api/v1/logos/classes",
                "logo_upload": "/api/v1/logos/upload",
                "inspection_start": "/api/v1/inspection/start",
                "inspection_status": "/api/v1/inspection/status",
                "ml_system_status": "/api/v1/ml/status",
                "ml_dataset_validate": "/api/v1/ml/dataset/validate",
                "ml_training_start": "/api/v1/ml/training/start",
                "ml_model_validate": "/api/v1/ml/model/validate",
                "ml_model_visualize": "/api/v1/ml/model/visualize",
                "documentation": "/docs",
                "dashboard": "/",
                "ui": "/ui",
                "batch_ui": "/ui/batch",
                "inspection_ui": "/ui/inspection",
                "ml_ui": "/ui/ml"
            },
            "phase": "3.0 - Training Pipeline Implementation"
        }


# Create app instance
app = create_app()


# Main entry point for direct execution
if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )