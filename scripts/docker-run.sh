#!/bin/bash

# Run the built Docker image locally
# Usage: ./scripts/docker-run.sh [tag]

set -e

# Configuration
DOCKER_HUB_USERNAME="kentatsujikawadev"
IMAGE_NAME="logo-detection-api"
TAG="${1:-latest}"
FULL_IMAGE_NAME="${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${TAG}"
CONTAINER_NAME="logo-detection-api"
PORT="${PORT:-8000}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Running Docker container${NC}"
echo "Image: ${FULL_IMAGE_NAME}"
echo "Port: ${PORT}"
echo ""

# Check if image exists
if ! docker images | grep -q "${DOCKER_HUB_USERNAME}/${IMAGE_NAME}.*${TAG}"; then
    echo -e "${RED}Error: Image not found locally${NC}"
    echo "Please run ./scripts/docker-build.sh first"
    exit 1
fi

# Stop existing container
if docker ps -a | grep -q ${CONTAINER_NAME}; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
fi

# Create directories
echo -e "${GREEN}Creating directories...${NC}"
mkdir -p ./models ./logs ./data

# Check model
if [ ! -f "./models/yolov8n.pt" ]; then
    echo -e "${YELLOW}Warning: Model not found at ./models/yolov8n.pt${NC}"
fi

# Run container
echo -e "${GREEN}Starting container...${NC}"
docker run -d \
    --name ${CONTAINER_NAME} \
    -p ${PORT}:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    -e ENVIRONMENT=development \
    -e LOG_LEVEL=INFO \
    ${FULL_IMAGE_NAME}

# Wait for startup
echo -e "${YELLOW}Waiting for API...${NC}"
for i in {1..30}; do
    if curl -f http://localhost:${PORT}/api/v1/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ API is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ API failed to start${NC}"
        echo "Check logs: docker logs ${CONTAINER_NAME}"
        exit 1
    fi
    sleep 2
done

# Show info
echo ""
echo -e "${GREEN}Container running!${NC}"
echo "  API: http://localhost:${PORT}"
echo "  Batch UI: http://localhost:${PORT}/ui/batch"
echo "  Docs: http://localhost:${PORT}/docs"
echo ""
echo "Commands:"
echo "  Logs: docker logs -f ${CONTAINER_NAME}"
echo "  Stop: docker stop ${CONTAINER_NAME}"