"""
Training Progress Monitoring System

This module provides real-time monitoring capabilities for training processes.
It includes WebSocket support for live progress updates, training metrics collection,
and progress visualization utilities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from pathlib import Path
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingMetrics:
    """Container for training metrics and statistics."""
    
    def __init__(self):
        self.epoch_metrics: List[Dict[str, Any]] = []
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.map_history: List[float] = []
        self.precision_history: List[float] = []
        self.recall_history: List[float] = []
        self.learning_rate_history: List[float] = []
        
        # Training session info
        self.session_start: Optional[datetime] = None
        self.session_end: Optional[datetime] = None
        self.total_epochs: int = 0
        self.current_epoch: int = 0
        self.best_metrics: Dict[str, float] = {}
        
        # Model info
        self.model_name: str = ""
        self.dataset_name: str = ""
        self.base_model: str = ""
        self.training_params: Dict[str, Any] = {}
    
    def add_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Add metrics for a completed epoch."""
        timestamp = datetime.now()
        
        epoch_data = {
            "epoch": epoch,
            "timestamp": timestamp.isoformat(),
            "metrics": metrics.copy()
        }
        
        self.epoch_metrics.append(epoch_data)
        self.current_epoch = epoch
        
        # Update history lists
        if "loss" in metrics:
            self.loss_history.append(metrics["loss"])
        if "val_loss" in metrics:
            self.val_loss_history.append(metrics["val_loss"])
        if "mAP" in metrics:
            self.map_history.append(metrics["mAP"])
        if "precision" in metrics:
            self.precision_history.append(metrics["precision"])
        if "recall" in metrics:
            self.recall_history.append(metrics["recall"])
        if "learning_rate" in metrics:
            self.learning_rate_history.append(metrics["learning_rate"])
        
        # Update best metrics
        for metric_name, value in metrics.items():
            if metric_name in ["mAP", "precision", "recall"]:
                # Higher is better
                if metric_name not in self.best_metrics or value > self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = value
            elif metric_name in ["loss", "val_loss"]:
                # Lower is better
                if metric_name not in self.best_metrics or value < self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = value
    
    def get_recent_metrics(self, num_epochs: int = 10) -> List[Dict[str, Any]]:
        """Get metrics for the most recent epochs."""
        return self.epoch_metrics[-num_epochs:]
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training session."""
        if not self.epoch_metrics:
            return {"status": "no_data"}
        
        duration = None
        if self.session_start and self.session_end:
            duration = (self.session_end - self.session_start).total_seconds()
        elif self.session_start:
            duration = (datetime.now() - self.session_start).total_seconds()
        
        latest_metrics = self.epoch_metrics[-1]["metrics"] if self.epoch_metrics else {}
        
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "base_model": self.base_model,
            "training_params": self.training_params,
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "session_end": self.session_end.isoformat() if self.session_end else None,
            "duration_seconds": duration,
            "total_epochs": self.total_epochs,
            "current_epoch": self.current_epoch,
            "completion_ratio": self.current_epoch / self.total_epochs if self.total_epochs > 0 else 0,
            "latest_metrics": latest_metrics,
            "best_metrics": self.best_metrics,
            "total_epoch_data": len(self.epoch_metrics)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary format."""
        return {
            "summary": self.get_training_summary(),
            "epoch_metrics": self.epoch_metrics,
            "history": {
                "loss": self.loss_history,
                "val_loss": self.val_loss_history,
                "mAP": self.map_history,
                "precision": self.precision_history,
                "recall": self.recall_history,
                "learning_rate": self.learning_rate_history
            }
        }


class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.settings = get_settings()
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSocket clients."""
        if not self.active_connections:
            return
        
        message_text = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.warning(f"Failed to broadcast to WebSocket: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


class TrainingMonitor:
    """Main training monitoring system."""
    
    def __init__(self):
        self.settings = get_settings()
        self.current_metrics = TrainingMetrics()
        self.websocket_manager = WebSocketManager()
        self.progress_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Progress tracking
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Metrics storage
        self.metrics_history: List[TrainingMetrics] = []
        self.max_history_sessions = 10
    
    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback function for progress updates."""
        self.progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove a progress callback function."""
        if callback in self.progress_callbacks:
            self.progress_callbacks.remove(callback)
    
    async def start_monitoring_session(self, 
                                     model_name: str,
                                     dataset_name: str,
                                     base_model: str,
                                     training_params: Dict[str, Any]):
        """Start a new training monitoring session."""
        # Save previous session if exists
        if self.current_metrics.epoch_metrics:
            self.current_metrics.session_end = datetime.now()
            self.metrics_history.append(self.current_metrics)
            
            # Limit history size
            if len(self.metrics_history) > self.max_history_sessions:
                self.metrics_history = self.metrics_history[-self.max_history_sessions:]
        
        # Initialize new session
        self.current_metrics = TrainingMetrics()
        self.current_metrics.model_name = model_name
        self.current_metrics.dataset_name = dataset_name
        self.current_metrics.base_model = base_model
        self.current_metrics.training_params = training_params
        self.current_metrics.session_start = datetime.now()
        self.current_metrics.total_epochs = training_params.get("epochs", 0)
        
        self.is_monitoring = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"Started monitoring session for model '{model_name}'")
        
        # Notify callbacks and WebSocket clients
        await self._broadcast_update({
            "type": "session_started",
            "data": self.current_metrics.get_training_summary()
        })
    
    async def stop_monitoring_session(self):
        """Stop the current monitoring session."""
        self.is_monitoring = False
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Finalize current session
        if self.current_metrics.epoch_metrics:
            self.current_metrics.session_end = datetime.now()
        
        logger.info("Stopped monitoring session")
        
        # Notify callbacks and WebSocket clients
        await self._broadcast_update({
            "type": "session_stopped",
            "data": self.current_metrics.get_training_summary()
        })
    
    async def update_progress(self, 
                            epoch: int,
                            metrics: Dict[str, Any],
                            status: str = "training"):
        """Update training progress with new metrics."""
        if not self.is_monitoring:
            return
        
        # Add metrics to current session
        self.current_metrics.add_epoch_metrics(epoch, metrics)
        
        # Create progress update
        progress_data = {
            "type": "progress_update",
            "data": {
                "epoch": epoch,
                "total_epochs": self.current_metrics.total_epochs,
                "metrics": metrics,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "summary": self.current_metrics.get_training_summary()
            }
        }
        
        # Broadcast update
        await self._broadcast_update(progress_data)
        
        logger.debug(f"Updated progress: epoch {epoch}, metrics: {metrics}")
    
    async def update_status(self, status: str, message: str = ""):
        """Update training status."""
        status_data = {
            "type": "status_update",
            "data": {
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "summary": self.current_metrics.get_training_summary()
            }
        }
        
        await self._broadcast_update(status_data)
    
    async def _broadcast_update(self, update_data: Dict[str, Any]):
        """Broadcast update to all clients and callbacks."""
        # Call progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback(update_data)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
        
        # Broadcast to WebSocket clients
        await self.websocket_manager.broadcast(update_data)
    
    async def _monitoring_loop(self):
        """Main monitoring loop for periodic updates."""
        try:
            while self.is_monitoring:
                await asyncio.sleep(1)  # Update every second
                
                # Send periodic heartbeat
                if self.current_metrics.epoch_metrics:
                    heartbeat_data = {
                        "type": "heartbeat",
                        "data": {
                            "timestamp": datetime.now().isoformat(),
                            "current_epoch": self.current_metrics.current_epoch,
                            "is_training": True
                        }
                    }
                    await self.websocket_manager.broadcast(heartbeat_data)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
    
    def get_current_session(self) -> Dict[str, Any]:
        """Get current training session data."""
        return self.current_metrics.to_dict()
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get historical training session data."""
        history = []
        
        # Add completed sessions
        for metrics in self.metrics_history:
            history.append(metrics.to_dict())
        
        # Add current session if active
        if self.current_metrics.epoch_metrics:
            history.append(self.current_metrics.to_dict())
        
        return history
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        all_sessions = self.get_session_history()
        
        if not all_sessions:
            return {"total_sessions": 0}
        
        # Calculate aggregate statistics
        total_sessions = len(all_sessions)
        total_epochs = sum(session["summary"]["current_epoch"] for session in all_sessions)
        
        # Find best metrics across all sessions
        best_overall = {}
        for session in all_sessions:
            best_metrics = session["summary"]["best_metrics"]
            for metric, value in best_metrics.items():
                if metric in ["mAP", "precision", "recall"]:
                    if metric not in best_overall or value > best_overall[metric]:
                        best_overall[metric] = value
                elif metric in ["loss", "val_loss"]:
                    if metric not in best_overall or value < best_overall[metric]:
                        best_overall[metric] = value
        
        # Calculate average session duration
        completed_sessions = [s for s in all_sessions if s["summary"]["duration_seconds"]]
        avg_duration = 0
        if completed_sessions:
            avg_duration = sum(s["summary"]["duration_seconds"] for s in completed_sessions) / len(completed_sessions)
        
        return {
            "total_sessions": total_sessions,
            "total_epochs_trained": total_epochs,
            "average_session_duration": avg_duration,
            "best_metrics_overall": best_overall,
            "current_session_active": self.is_monitoring,
            "current_epoch": self.current_metrics.current_epoch if self.is_monitoring else 0
        }
    
    async def save_session_log(self, session_name: Optional[str] = None) -> str:
        """Save current session metrics to file."""
        if not self.current_metrics.epoch_metrics:
            raise ValueError("No metrics to save")
        
        if session_name is None:
            session_name = f"training_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        log_dir = Path(self.settings.training_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{session_name}.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.current_metrics.to_dict(), f, indent=2)
        
        logger.info(f"Training session saved to {log_file}")
        return str(log_file)


# Global training monitor instance
_training_monitor = None


def get_training_monitor() -> TrainingMonitor:
    """Get the global training monitor instance."""
    global _training_monitor
    if _training_monitor is None:
        _training_monitor = TrainingMonitor()
    return _training_monitor


# WebSocket endpoint handler
async def handle_websocket_connection(websocket: WebSocket):
    """Handle WebSocket connections for real-time training updates."""
    monitor = get_training_monitor()
    await monitor.websocket_manager.connect(websocket)
    
    try:
        # Send current status on connection
        current_session = monitor.get_current_session()
        await monitor.websocket_manager.send_personal_message({
            "type": "connection_established",
            "data": current_session
        }, websocket)
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client
                message = await websocket.receive_text()
                data = json.loads(message)
                
                # Handle client requests
                if data.get("type") == "get_current_session":
                    current_session = monitor.get_current_session()
                    await monitor.websocket_manager.send_personal_message({
                        "type": "current_session_response",
                        "data": current_session
                    }, websocket)
                
                elif data.get("type") == "get_statistics":
                    stats = monitor.get_training_statistics()
                    await monitor.websocket_manager.send_personal_message({
                        "type": "statistics_response",
                        "data": stats
                    }, websocket)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                logger.warning("Received invalid JSON from WebSocket client")
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                break
    
    finally:
        monitor.websocket_manager.disconnect(websocket)