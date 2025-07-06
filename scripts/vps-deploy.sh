#!/bin/bash

# VPS deployment script
# This script should be run on your VPS to deploy the logo detection API

set -e

# Configuration
DOCKER_HUB_USERNAME="${DOCKER_HUB_USERNAME:-your-username}"
IMAGE_NAME="logo-detection-api"
TAG="${1:-latest}"
CONTAINER_NAME="logo-detection-api"
PORT="${PORT:-8000}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Deploying Logo Detection API on VPS${NC}"
echo ""

# Pull the latest image
echo -e "${GREEN}Pulling Docker image...${NC}"
docker pull ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${TAG}

# Stop existing container if running
if docker ps | grep -q ${CONTAINER_NAME}; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

# Create directories if they don't exist
echo -e "${GREEN}Creating necessary directories...${NC}"
mkdir -p ./models ./logs ./data

# Download model if not exists
if [ ! -f "./models/yolov8n.pt" ]; then
    echo -e "${YELLOW}Downloading YOLOv8 model...${NC}"
    wget -O ./models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
fi

# Run the container
echo -e "${GREEN}Starting container...${NC}"
docker run -d \
    --name ${CONTAINER_NAME} \
    --restart unless-stopped \
    -p ${PORT}:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    -e ENVIRONMENT=production \
    -e LOG_LEVEL=INFO \
    ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${TAG}

# Wait for container to be healthy
echo -e "${YELLOW}Waiting for API to be ready...${NC}"
for i in {1..30}; do
    if curl -f http://localhost:${PORT}/api/v1/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ API is ready!${NC}"
        break
    fi
    sleep 2
done

# Show status
echo ""
docker ps | grep ${CONTAINER_NAME}
echo ""
echo -e "${GREEN}Deployment complete!${NC}"
echo "API is available at: http://localhost:${PORT}"
echo "Batch UI is available at: http://localhost:${PORT}/ui/batch"