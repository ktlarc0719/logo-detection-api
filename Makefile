# Makefile for Logo Detection API

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_IMAGE = kentatsujikawadev/logo-detection-api:latest
CONTAINER_NAME = logo-detection-api

# Help target
help:
	@echo "Logo Detection API - Available commands:"
	@echo "  make git        - Add, commit and push all changes"
	@echo "  make build      - Build Docker image"
	@echo "  make run        - Run Docker container"
	@echo "  make stop       - Stop Docker container"
	@echo "  make logs       - Show container logs"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean up"

# Git commands
git:
	@echo "Syncing with git..."
	@git pull origin main --rebase || git pull origin main
	@git add .
	@git commit -m "Update $$(date +%Y-%m-%d_%H:%M:%S)" || true
	@git push origin main

# Quick git with custom message
gitmsg:
	@read -p "Commit message: " msg; \
	git add . && \
	git commit -m "$$msg" && \
	git push origin main

# Docker commands
build:
	docker build -t $(DOCKER_IMAGE) .

run:
	./scripts/docker-run.sh

stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

logs:
	docker logs -f $(CONTAINER_NAME)

# Development
test:
	python3 scripts/simple_performance_test.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -f *.log build.log

# VPS deployment
deploy-vps:
	@echo "Run on your VPS:"
	@echo "wget https://github.com/ktlarc0719/logo-detection-api/raw/master/scripts/vps_setup_final.sh"
	@echo "chmod +x vps_setup_final.sh"
	@echo "sudo ./vps_setup_final.sh"

.PHONY: help git gitmsg build run stop logs test clean deploy-vps