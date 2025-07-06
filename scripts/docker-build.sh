#!/bin/bash

# Build Docker image locally
# Usage: ./scripts/docker-build.sh [tag]

set -e

# Configuration
DOCKER_HUB_USERNAME="kentatsujikawadev"
IMAGE_NAME="logo-detection-api"
TAG="${1:-latest}"
FULL_IMAGE_NAME="${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${TAG}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Building Docker image${NC}"
echo "Image: ${FULL_IMAGE_NAME}"
echo ""

# Build the image
echo -e "${GREEN}Building...${NC}"
# BuildKitを無効にして標準ビルドを使用
docker build -t ${FULL_IMAGE_NAME} .

echo ""
echo -e "${GREEN}✓ Build complete!${NC}"
echo "Image: ${FULL_IMAGE_NAME}"