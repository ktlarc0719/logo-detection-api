# Docker Hub Deployment Guide

## Overview

This guide explains how to build, push to Docker Hub, and deploy the Logo Detection API to VPS servers.

## Prerequisites

1. Docker Hub account
2. Docker installed on local machine and VPS
3. VPS with Docker installed

## 1. Local Setup (Build & Push)

### Configure Docker Hub Username

```bash
export DOCKER_HUB_USERNAME="your-docker-hub-username"
```

### Login to Docker Hub

```bash
docker login
```

### Build and Push Image

```bash
# Build and push with latest tag
./scripts/docker-build-push.sh

# Build and push with specific tag
./scripts/docker-build-push.sh v1.0.0
```

## 2. VPS Deployment

### Option A: Using Deploy Script

1. Copy the deploy script to your VPS:
```bash
scp scripts/vps-deploy.sh user@your-vps:/home/user/
```

2. SSH into your VPS and run:
```bash
export DOCKER_HUB_USERNAME="your-docker-hub-username"
./vps-deploy.sh
```

### Option B: Using Docker Compose

1. Copy necessary files to VPS:
```bash
scp docker-compose.vps.yml nginx/nginx.conf user@your-vps:/home/user/logo-detection/
```

2. Create `.env` file on VPS:
```bash
DOCKER_HUB_USERNAME=your-docker-hub-username
PORT=8000
```

3. Deploy:
```bash
# API only
docker-compose -f docker-compose.vps.yml up -d

# With Nginx
docker-compose -f docker-compose.vps.yml --profile with-nginx up -d
```

## 3. Directory Structure on VPS

```
/home/user/logo-detection/
├── docker-compose.vps.yml
├── .env
├── nginx/
│   └── nginx.conf
├── models/
│   └── yolov8n.pt
├── logs/
└── data/
```

## 4. Updating Deployment

```bash
# Pull latest image
docker pull ${DOCKER_HUB_USERNAME}/logo-detection-api:latest

# Restart container
docker-compose -f docker-compose.vps.yml down
docker-compose -f docker-compose.vps.yml up -d
```

## 5. Monitoring

```bash
# Check logs
docker logs logo-detection-api

# Check health
curl http://localhost:8000/api/v1/health

# View real-time logs
docker logs -f logo-detection-api
```

## 6. Multiple VPS Deployment

For deploying to multiple VPS servers, you can use a simple bash script:

```bash
#!/bin/bash
VPS_SERVERS=("vps1.example.com" "vps2.example.com" "vps3.example.com")

for server in "${VPS_SERVERS[@]}"; do
    echo "Deploying to $server..."
    ssh user@$server "cd /home/user/logo-detection && docker-compose -f docker-compose.vps.yml pull && docker-compose -f docker-compose.vps.yml up -d"
done
```

## Environment Variables

- `DOCKER_HUB_USERNAME`: Your Docker Hub username
- `PORT`: API port (default: 8000)
- `TAG`: Docker image tag (default: latest)
- `ENVIRONMENT`: production/development
- `LOG_LEVEL`: INFO/DEBUG

## Troubleshooting

1. **Container won't start**: Check logs with `docker logs logo-detection-api`
2. **Model not found**: Ensure models directory contains yolov8n.pt
3. **Permission issues**: Check volume mount permissions
4. **Port already in use**: Change PORT in .env file