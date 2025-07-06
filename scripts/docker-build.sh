#!/bin/bash

# Safe Docker build script with error handling
# Usage: ./scripts/docker-build-safe.sh [tag]

# Configuration
DOCKER_HUB_USERNAME="kentatsujikawadev"
IMAGE_NAME="logo-detection-api"
TAG="${1:-latest}"
FULL_IMAGE_NAME="${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${TAG}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Error handling
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    error_exit "Docker is not installed. Please install Docker first."
fi

# Check if Docker daemon is running
if ! docker info > /dev/null 2>&1; then
    error_exit "Docker daemon is not running. Please start Docker."
fi

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    error_exit "Dockerfile not found in current directory. Please run from project root."
fi

echo -e "${YELLOW}Building Docker image${NC}"
echo "Image: ${FULL_IMAGE_NAME}"
echo "BuildKit: Enabled"
echo ""

# Show Docker version
echo -e "${YELLOW}Docker version:${NC}"
docker version --format 'Client: {{.Client.Version}}\nServer: {{.Server.Version}}'
echo ""

# Build the image with proper error handling
echo -e "${GREEN}Starting build...${NC}"

if DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    -t "${FULL_IMAGE_NAME}" \
    . 2>&1 | tee build.log; then
    
    echo ""
    echo -e "${GREEN}✓ Build completed successfully!${NC}"
    echo "Image: ${FULL_IMAGE_NAME}"
    echo ""
    
    # Show image info
    echo -e "${YELLOW}Image details:${NC}"
    docker images "${FULL_IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "  Test locally:  ./scripts/docker-run.sh"
    echo "  Push to hub:   ./scripts/docker-push.sh"
    
else
    echo ""
    echo -e "${RED}✗ Build failed!${NC}"
    echo "Check build.log for details"
    echo ""
    echo "Common issues:"
    echo "  - Out of disk space: df -h"
    echo "  - Docker daemon issues: sudo systemctl status docker"
    echo "  - Build context too large: check .dockerignore"
    exit 1
fi