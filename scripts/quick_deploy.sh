#!/bin/bash
# Quick deployment script for Logo Detection API
# Usage: curl -sSL https://your-repo.com/scripts/quick_deploy.sh | bash

echo "🚀 Quick Deploy: Logo Detection API"
echo "=================================="

sudo apt update
sudo apt install -y curl
curl -sSL https://raw.githubusercontent.com/ktlarc0719/logo-detection-setup/refs/heads/main/vps-setup.sh | bash