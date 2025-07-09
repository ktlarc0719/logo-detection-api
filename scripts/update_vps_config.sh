#!/bin/bash
# Update VPS configuration with recommended settings

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Updating VPS Configuration${NC}"
echo "================================="

# Configuration file path
ENV_FILE="/opt/logo-detection/.env"
MANAGER_FILE="/opt/logo-detection/manager.py"

# Create backup
echo -e "${YELLOW}Creating backup...${NC}"
sudo cp $ENV_FILE ${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)

# Update .env file with recommended settings
echo -e "${YELLOW}Updating .env file...${NC}"
cat <<'EOF' | sudo tee $ENV_FILE > /dev/null
# Logo Detection API Configuration
# Optimized for 2GB VPS based on performance testing
# Last updated: $(date)

# === RECOMMENDED SETTINGS ===
# Based on performance testing results:
# - Balanced configuration achieved best performance (10.23 URLs/sec)
# - Memory usage peaked at 593.8 MB
# - Optimal for 2-core 2GB VPS

# Performance Settings
MAX_CONCURRENT_DETECTIONS=2
MAX_CONCURRENT_DOWNLOADS=15
MAX_BATCH_SIZE=30

# Memory Settings (for Docker container)
MEMORY_LIMIT=1g
MEMORY_RESERVATION=512m

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO

# API Settings
PORT=8000

# Model Settings
MODEL_DEVICE=cpu
CONFIDENCE_THRESHOLD=0.5
MAX_DETECTIONS=10

# Additional Settings
DOWNLOAD_TIMEOUT=30
PROCESSING_TIMEOUT=300
EOF

# Update manager.py to use memory settings from .env
echo -e "${YELLOW}Updating manager.py...${NC}"
sudo sed -i 's/--memory", "1.5g"/--memory", os.getenv("MEMORY_LIMIT", "1g")/g' $MANAGER_FILE 2>/dev/null || true
sudo sed -i 's/--memory-reservation", "1g"/--memory-reservation", os.getenv("MEMORY_RESERVATION", "512m")/g' $MANAGER_FILE 2>/dev/null || true

# Create a deployment script that reads from .env
echo -e "${YELLOW}Creating deployment script...${NC}"
cat <<'DEPLOY_SCRIPT' | sudo tee /opt/logo-detection/deploy_with_env.sh > /dev/null
#!/bin/bash
# Deploy with settings from .env file

# Load environment variables
export $(grep -v '^#' /opt/logo-detection/.env | xargs)

# Stop and remove existing container
docker stop logo-detection-api 2>/dev/null || true
docker rm logo-detection-api 2>/dev/null || true

# Pull latest image
docker pull kentatsujikawadev/logo-detection-api:latest

# Run with environment variables
docker run -d \
  --name logo-detection-api \
  --restart=always \
  --memory="${MEMORY_LIMIT:-1g}" \
  --memory-reservation="${MEMORY_RESERVATION:-512m}" \
  -p "${PORT:-8000}:8000" \
  -v /opt/logo-detection/logs:/app/logs \
  -v /opt/logo-detection/data:/app/data \
  --env-file /opt/logo-detection/.env \
  kentatsujikawadev/logo-detection-api:latest

echo "Container started with optimized settings"
docker ps | grep logo-detection-api
DEPLOY_SCRIPT

sudo chmod +x /opt/logo-detection/deploy_with_env.sh

# Create status check script
echo -e "${YELLOW}Creating status check script...${NC}"
cat <<'STATUS_SCRIPT' | sudo tee /opt/logo-detection/check_status.sh > /dev/null
#!/bin/bash
# Check current configuration and status

echo "=== Current Configuration ==="
echo "From .env file:"
grep -E "^(MAX_|MEMORY_)" /opt/logo-detection/.env

echo ""
echo "=== Container Status ==="
docker stats logo-detection-api --no-stream

echo ""
echo "=== API Health ==="
curl -s http://localhost:8000/api/v1/health | jq . 2>/dev/null || echo "API not responding"

echo ""
echo "=== Recent Logs ==="
docker logs logo-detection-api --tail 10
STATUS_SCRIPT

sudo chmod +x /opt/logo-detection/check_status.sh

echo ""
echo -e "${GREEN}✅ Configuration updated successfully!${NC}"
echo ""
echo "Current settings in $ENV_FILE:"
grep -E "^(MAX_|MEMORY_)" $ENV_FILE
echo ""
echo -e "${YELLOW}To apply the new configuration:${NC}"
echo "  sudo /opt/logo-detection/deploy_with_env.sh"
echo ""
echo -e "${YELLOW}To check status:${NC}"
echo "  sudo /opt/logo-detection/check_status.sh"
echo ""
echo -e "${YELLOW}To update configuration via API:${NC}"
echo "  curl -X POST http://localhost:8080/deploy"
echo ""
echo -e "${GREEN}Note: All settings are now centralized in /opt/logo-detection/.env${NC}"