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
from src.api.endpoints import batch_detection, single_detection, health, model_management, training, logo_management, download_images, system_management, url_batch_detection
from api.dataset_splitter import router as dataset_splitter_router

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
    
    # Processing endpoints
    app.include_router(
        batch_detection.router,
        prefix="/api/v1/process",
        tags=["Batch Processing"]
    )
    
    app.include_router(
        single_detection.router,
        prefix="/api/v1/process",
        tags=["Single Image Processing"]
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
    
    # Dataset splitter endpoint
    app.include_router(
        dataset_splitter_router,
        prefix="/api/v1",
        tags=["Dataset Management"]
    )
    
    # System management endpoints
    app.include_router(
        system_management.router,
        prefix="/api/v1",
        tags=["System Management"]
    )
    
    # URL batch processing endpoints
    app.include_router(
        url_batch_detection.router,
        prefix="/api/v1",
        tags=["URL Batch Processing"]
    )
    
    # Serve static files and UI
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # UI endpoints
    @app.get("/ui", tags=["UI"])
    async def serve_ui():
        """Serve the detection UI."""
        return FileResponse('static/index.html')
    
    @app.get("/ui/batch", tags=["UI"])
    async def serve_batch_ui():
        """Serve the batch processing UI."""
        return FileResponse('static/batch.html')
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
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
                "batch_processing": "/api/v1/process/batch",
                "single_processing": "/api/v1/process/single",
                "models": "/api/v1/models",
                "model_switch": "/api/v1/models/switch",
                "brands": "/api/v1/brands",
                "categories": "/api/v1/categories",
                "training_status": "/api/v1/training/status",
                "training_start": "/api/v1/training/start",
                "datasets": "/api/v1/training/datasets",
                "logo_classes": "/api/v1/logos/classes",
                "logo_upload": "/api/v1/logos/upload",
                "documentation": "/docs",
                "ui": "/ui",
                "batch_ui": "/ui/batch"
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