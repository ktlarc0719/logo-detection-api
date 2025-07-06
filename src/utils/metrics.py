import time
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingMetrics:
    """Metrics for a single processing operation."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    processing_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Mark processing as finished."""
        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message


@dataclass
class BatchMetrics:
    """Metrics for batch processing."""
    batch_id: str
    total_images: int
    successful: int = 0
    failed: int = 0
    processing_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    errors: List[Dict[str, str]] = field(default_factory=list)
    
    def add_error(self, image_id: int, error: str):
        """Add error to batch metrics."""
        self.errors.append({"image_id": image_id, "error": error})
        self.failed += 1
    
    def add_success(self):
        """Increment successful count."""
        self.successful += 1
    
    def finish(self):
        """Finalize batch metrics."""
        self.processing_time = time.time() - self.start_time


class MetricsCollector:
    """Thread-safe metrics collector."""
    
    def __init__(self):
        self.settings = get_settings()
        self._lock = threading.Lock()
        self._start_time = time.time()
        
        # Processing metrics
        self._total_processed = 0
        self._total_successful = 0
        self._total_failed = 0
        self._processing_times = deque(maxlen=1000)  # Keep last 1000 processing times
        
        # Batch metrics
        self._batch_metrics: Dict[str, BatchMetrics] = {}
        self._completed_batches = deque(maxlen=100)  # Keep last 100 completed batches
        
        # Error tracking
        self._errors_by_type = defaultdict(int)
        self._recent_errors = deque(maxlen=100)
        
        # Performance tracking
        self._performance_samples = deque(maxlen=100)
        self._last_performance_check = time.time()
    
    def start_batch(self, batch_id: str, total_images: int) -> BatchMetrics:
        """Start tracking a new batch."""
        with self._lock:
            batch_metrics = BatchMetrics(batch_id=batch_id, total_images=total_images)
            self._batch_metrics[batch_id] = batch_metrics
            return batch_metrics
    
    def finish_batch(self, batch_id: str) -> Optional[BatchMetrics]:
        """Finish tracking a batch."""
        with self._lock:
            batch_metrics = self._batch_metrics.pop(batch_id, None)
            if batch_metrics:
                batch_metrics.finish()
                self._completed_batches.append(batch_metrics)
                self._total_processed += batch_metrics.total_images
                self._total_successful += batch_metrics.successful
                self._total_failed += batch_metrics.failed
            return batch_metrics
    
    def record_processing_time(self, processing_time: float):
        """Record a processing time."""
        with self._lock:
            self._processing_times.append(processing_time)
    
    def record_error(self, error_type: str, error_message: str):
        """Record an error."""
        with self._lock:
            self._errors_by_type[error_type] += 1
            self._recent_errors.append({
                "timestamp": datetime.utcnow().isoformat(),
                "type": error_type,
                "message": error_message
            })
    
    def record_performance_sample(self):
        """Record current system performance."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            sample = {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3)
            }
            
            with self._lock:
                self._performance_samples.append(sample)
                self._last_performance_check = time.time()
                
        except Exception as e:
            logger.warning(f"Failed to record performance sample: {e}")
    
    def get_current_metrics(self) -> Dict:
        """Get current metrics snapshot."""
        with self._lock:
            # Calculate average processing time
            avg_processing_time = 0.0
            if self._processing_times:
                avg_processing_time = sum(self._processing_times) / len(self._processing_times)
            
            # Calculate error rate
            error_rate = 0.0
            if self._total_processed > 0:
                error_rate = self._total_failed / self._total_processed
            
            # Get recent performance data
            latest_performance = None
            if self._performance_samples:
                latest_performance = self._performance_samples[-1]
            
            uptime = time.time() - self._start_time
            
            return {
                "total_processed": self._total_processed,
                "total_successful": self._total_successful,
                "total_failed": self._total_failed,
                "avg_processing_time": round(avg_processing_time, 4),
                "error_rate": round(error_rate, 4),
                "uptime_seconds": round(uptime, 2),
                "active_batches": len(self._batch_metrics),
                "recent_errors": list(self._recent_errors)[-10:],  # Last 10 errors
                "errors_by_type": dict(self._errors_by_type),
                "performance": latest_performance
            }
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict]:
        """Get status of a specific batch."""
        with self._lock:
            batch_metrics = self._batch_metrics.get(batch_id)
            if batch_metrics:
                return {
                    "batch_id": batch_metrics.batch_id,
                    "total_images": batch_metrics.total_images,
                    "successful": batch_metrics.successful,
                    "failed": batch_metrics.failed,
                    "processing_time": time.time() - batch_metrics.start_time,
                    "status": "processing",
                    "errors": batch_metrics.errors
                }
            return None
    
    def cleanup_old_data(self):
        """Clean up old metrics data based on retention settings."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.settings.metrics_retention_hours)
        cutoff_timestamp = cutoff_time.timestamp()
        
        with self._lock:
            # Clean up old performance samples
            while (self._performance_samples and 
                   self._performance_samples[0]["timestamp"] < cutoff_timestamp):
                self._performance_samples.popleft()
            
            # Clean up old error records
            while (self._recent_errors and 
                   datetime.fromisoformat(self._recent_errors[0]["timestamp"]).timestamp() < cutoff_timestamp):
                self._recent_errors.popleft()


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector


class MetricsTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        processing_time = self.end_time - self.start_time
        
        # Record the timing
        metrics_collector.record_processing_time(processing_time)
        
        # Log if enabled
        if exc_type is None:
            logger.debug(f"{self.operation_name} completed in {processing_time:.4f}s")
        else:
            logger.error(f"{self.operation_name} failed after {processing_time:.4f}s: {exc_val}")
            metrics_collector.record_error(
                error_type=exc_type.__name__,
                error_message=str(exc_val)
            )
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time (during or after execution)."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time


# Background task to collect performance metrics
import asyncio

async def performance_monitor():
    """Background task to monitor system performance."""
    while True:
        try:
            metrics_collector.record_performance_sample()
            metrics_collector.cleanup_old_data()
            await asyncio.sleep(60)  # Sample every minute
        except Exception as e:
            logger.error(f"Performance monitor error: {e}")
            await asyncio.sleep(60)