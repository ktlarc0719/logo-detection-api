#!/bin/bash

# Push built Docker image to Docker Hub
# Usage: ./scripts/docker-push.sh [tag]

set -e

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

echo -e "${YELLOW}Pushing to Docker Hub${NC}"
echo "Image: ${FULL_IMAGE_NAME}"
echo ""

# Check if image exists
if ! docker images | grep -q "${DOCKER_HUB_USERNAME}/${IMAGE_NAME}.*${TAG}"; then
    echo -e "${RED}Error: Image not found locally${NC}"
    echo "Please run ./scripts/docker-build.sh first"
    exit 1
fi

# Check Docker Hub login
if ! docker info | grep -q "Username"; then
    echo -e "${YELLOW}Please log in to Docker Hub:${NC}"
    docker login
fi

# Push image
echo -e "${GREEN}Pushing image...${NC}"
docker push ${FULL_IMAGE_NAME}

# Also tag and push as latest if different
if [ "${TAG}" != "latest" ]; then
    echo -e "${GREEN}Tagging as latest...${NC}"
    docker tag ${FULL_IMAGE_NAME} ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest
    docker push ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest
fi

echo ""
echo -e "${GREEN}✓ Push complete!${NC}"
echo "Image available at: docker pull ${FULL_IMAGE_NAME}"