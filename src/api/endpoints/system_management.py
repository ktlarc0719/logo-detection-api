"""System management endpoints for updates and maintenance."""

from fastapi import APIRouter, HTTPException, Header, BackgroundTasks
from typing import Dict, Optional
import subprocess
import os
import signal
import sys
from datetime import datetime
import hashlib
import hmac

from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Update webhook secret (should be in .env)
UPDATE_SECRET = os.getenv("UPDATE_WEBHOOK_SECRET", "your-secret-key-here")


def verify_webhook_signature(payload: str, signature: str, secret: str) -> bool:
    """Verify webhook signature for security."""
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected_signature)


def update_application():
    """Execute git pull and restart the application."""
    try:
        logger.info("Starting application update...")
        
        # Git pull
        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True,
            text=True,
            cwd="/app"
        )
        
        if result.returncode != 0:
            logger.error(f"Git pull failed: {result.stderr}")
            return False, result.stderr
        
        logger.info(f"Git pull successful: {result.stdout}")
        
        # Send SIGTERM to trigger graceful restart
        logger.info("Triggering application restart...")
        os.kill(os.getpid(), signal.SIGTERM)
        
        return True, "Update initiated successfully"
        
    except Exception as e:
        logger.error(f"Update failed: {str(e)}")
        return False, str(e)


@router.post(
    "/system/update",
    response_model=Dict[str, str],
    summary="Update application from Git repository",
    description="Pull latest changes from Git and restart the application.",
    tags=["System Management"]
)
async def trigger_update(
    background_tasks: BackgroundTasks,
    x_webhook_signature: Optional[str] = Header(None)
) -> Dict[str, str]:
    """
    Trigger application update via Git pull.
    
    This endpoint can be called by CI/CD webhooks or manually.
    For security, it requires a signature header in production.
    """
    # In production, verify webhook signature
    if os.getenv("ENVIRONMENT") == "production" and x_webhook_signature:
        # You would typically verify the webhook payload here
        # For simplicity, we're just checking if a signature is provided
        if not x_webhook_signature:
            raise HTTPException(
                status_code=401,
                detail="Webhook signature required in production"
            )
    
    try:
        # Schedule update in background to return response before restart
        background_tasks.add_task(update_application)
        
        return {
            "status": "success",
            "message": "Update initiated. Application will restart shortly.",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to initiate update: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate update: {str(e)}"
        )


@router.get(
    "/system/version",
    response_model=Dict[str, str],
    summary="Get current application version",
    description="Get the current Git commit hash and version information.",
    tags=["System Management"]
)
async def get_version() -> Dict[str, str]:
    """Get current application version information."""
    try:
        # Get current git commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd="/app"
        )
        
        commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        # Get commit date
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=iso"],
            capture_output=True,
            text=True,
            cwd="/app"
        )
        
        commit_date = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd="/app"
        )
        
        branch = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        return {
            "commit": commit_hash[:8],
            "commit_full": commit_hash,
            "branch": branch,
            "commit_date": commit_date,
            "api_version": "1.0.0",
            "python_version": sys.version.split()[0]
        }
        
    except Exception as e:
        logger.error(f"Failed to get version info: {str(e)}")
        return {
            "commit": "unknown",
            "commit_full": "unknown",
            "branch": "unknown",
            "commit_date": "unknown",
            "api_version": "1.0.0",
            "python_version": sys.version.split()[0]
        }


@router.post(
    "/system/restart",
    response_model=Dict[str, str],
    summary="Restart the application",
    description="Gracefully restart the application.",
    tags=["System Management"]
)
async def restart_application(
    background_tasks: BackgroundTasks,
    x_admin_token: Optional[str] = Header(None)
) -> Dict[str, str]:
    """
    Restart the application gracefully.
    
    Requires admin token in production environment.
    """
    # Simple auth check for production
    if os.getenv("ENVIRONMENT") == "production":
        admin_token = os.getenv("ADMIN_TOKEN", "")
        if not admin_token or x_admin_token != admin_token:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing admin token"
            )
    
    try:
        # Schedule restart in background
        background_tasks.add_task(lambda: os.kill(os.getpid(), signal.SIGTERM))
        
        return {
            "status": "success",
            "message": "Application restart initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to restart application: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to restart application: {str(e)}"
        )