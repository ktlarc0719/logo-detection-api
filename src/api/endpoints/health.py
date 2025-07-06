import psutil
import time
from datetime import datetime
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from src.models.schemas import HealthResponse, MetricsResponse
from src.core.detection_engine import get_detection_engine
from src.core.batch_processor import get_batch_processor
from src.utils.metrics import get_metrics_collector
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Check the health status of the logo detection service.",
    tags=["Health & Monitoring"]
)
async def health_check() -> HealthResponse:
    """
    Check the health status of the service.
    
    Returns:
    - Service status (ok/degraded/error)
    - Timestamp of the check
    - Model loading status
    - System information
    """
    try:
        # Check detection engine status
        detection_engine = get_detection_engine()
        model_loaded = detection_engine.is_loaded()
        
        # Get system information
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            system_info = {
                "cpu_count": cpu_count,
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_used_percent": round(memory.percent, 1),
                "cpu_percent": round(psutil.cpu_percent(interval=0.1), 1)
            }
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
            system_info = {"error": "Failed to retrieve system information"}
        
        # Determine service status
        if model_loaded:
            status = "ok"
        else:
            status = "degraded"
            logger.warning("Service is degraded: Detection model not loaded")
        
        return HealthResponse(
            status=status,
            timestamp=datetime.utcnow(),
            model_loaded=model_loaded,
            system_info=system_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )


@router.get(
    "/health/detailed",
    response_model=Dict[str, Any],
    summary="Detailed health check",
    description="Get detailed health information including component status.",
    tags=["Health & Monitoring"]
)
async def detailed_health_check() -> Dict[str, Any]:
    """
    Get detailed health information about all system components.
    
    Returns comprehensive status information for troubleshooting.
    """
    try:
        # Get detection engine status
        detection_engine = get_detection_engine()
        model_info = detection_engine.get_model_info()
        
        # Get batch processor status
        batch_processor = get_batch_processor()
        processor_status = batch_processor.get_status()
        
        # Get metrics collector status
        metrics_collector = get_metrics_collector()
        current_metrics = metrics_collector.get_current_metrics()
        
        # Get detailed system information
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_details = {
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "used_percent": round(memory.percent, 1)
                },
                "cpu": {
                    "count": psutil.cpu_count(),
                    "percent": round(psutil.cpu_percent(interval=0.1), 1),
                    "load_avg": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": round(disk.percent, 1)
                }
            }
        except Exception as e:
            logger.warning(f"Failed to get detailed system info: {e}")
            system_details = {"error": "Failed to retrieve detailed system information"}
        
        # Determine overall health
        issues = []
        if not model_info.get("loaded", False):
            issues.append("Detection model not loaded")
        
        if current_metrics.get("error_rate", 0) > 0.1:  # More than 10% error rate
            issues.append("High error rate detected")
        
        if system_details.get("memory", {}).get("used_percent", 0) > 90:
            issues.append("High memory usage")
        
        overall_status = "ok" if not issues else "degraded" if len(issues) < 3 else "error"
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "issues": issues,
            "components": {
                "detection_engine": model_info,
                "batch_processor": processor_status,
                "metrics": current_metrics
            },
            "system": system_details
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Detailed health check failed"
        )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get service metrics",
    description="Get current performance metrics and statistics.",
    tags=["Health & Monitoring"]
)
async def get_metrics() -> MetricsResponse:
    """
    Get current service metrics and performance statistics.
    
    Returns:
    - Processing statistics
    - Performance metrics
    - Error rates and information
    - System resource usage
    """
    try:
        metrics_collector = get_metrics_collector()
        current_metrics = metrics_collector.get_current_metrics()
        
        # Convert to response model
        response = MetricsResponse(
            total_processed=current_metrics.get("total_processed", 0),
            total_successful=current_metrics.get("total_successful", 0),
            total_failed=current_metrics.get("total_failed", 0),
            avg_processing_time=current_metrics.get("avg_processing_time", 0.0),
            error_rate=current_metrics.get("error_rate", 0.0),
            uptime_seconds=current_metrics.get("uptime_seconds", 0.0),
            active_batches=current_metrics.get("active_batches", 0),
            recent_errors=current_metrics.get("recent_errors", []),
            errors_by_type=current_metrics.get("errors_by_type", {}),
            performance=current_metrics.get("performance")
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve metrics"
        )


@router.get(
    "/readiness",
    summary="Readiness probe",
    description="Kubernetes-style readiness probe endpoint.",
    tags=["Health & Monitoring"]
)
async def readiness_probe() -> Dict[str, str]:
    """
    Readiness probe for Kubernetes-style health checks.
    
    Returns 200 if service is ready to handle requests, 503 otherwise.
    """
    try:
        detection_engine = get_detection_engine()
        
        if detection_engine.is_loaded():
            return {"status": "ready"}
        else:
            raise HTTPException(
                status_code=503,
                detail="Service not ready: Detection model not loaded"
            )
            
    except Exception as e:
        logger.error(f"Readiness probe failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Service not ready"
        )


@router.get(
    "/liveness",
    summary="Liveness probe",
    description="Kubernetes-style liveness probe endpoint.",
    tags=["Health & Monitoring"]
)
async def liveness_probe() -> Dict[str, str]:
    """
    Liveness probe for Kubernetes-style health checks.
    
    Returns 200 if service is alive, 500 otherwise.
    """
    try:
        # Simple liveness check - just return success if we can respond
        return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Liveness probe failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Service not alive"
        )