#!/bin/bash

# Firewall setup script for Logo Detection API
# This script configures UFW (Uncomplicated Firewall) to allow necessary ports

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🔥 Setting up firewall rules...${NC}"

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run with sudo: sudo $0${NC}"
    exit 1
fi

# Install UFW if not installed
if ! command -v ufw &> /dev/null; then
    echo -e "${YELLOW}Installing UFW...${NC}"
    apt-get update
    apt-get install -y ufw
fi

# Configure UFW
echo -e "${YELLOW}Configuring firewall rules...${NC}"

# Allow SSH (important to not lock yourself out!)
ufw allow 22/tcp comment 'SSH'

# Allow API port
ufw allow 8000/tcp comment 'Logo Detection API'

# Allow Management API port (you might want to restrict this)
ufw allow 8080/tcp comment 'Logo Detection Manager'

# Allow HTTP and HTTPS (if you plan to use reverse proxy later)
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'

# Enable UFW (with auto-yes to avoid prompt)
echo "y" | ufw enable

# Show status
echo -e "${GREEN}✓ Firewall configured!${NC}"
echo ""
ufw status verbose

# Additional cloud provider specific instructions
echo ""
echo -e "${YELLOW}📝 Additional Steps:${NC}"
echo ""
echo "If you're using a cloud provider, you may also need to:"
echo ""
echo -e "${GREEN}AWS EC2:${NC}"
echo "  1. Go to EC2 Console > Security Groups"
echo "  2. Add inbound rules for ports 8000 and 8080"
echo ""
echo -e "${GREEN}Google Cloud:${NC}"
echo "  1. Go to VPC network > Firewall rules"
echo "  2. Create rules to allow tcp:8000,8080"
echo ""
echo -e "${GREEN}DigitalOcean:${NC}"
echo "  1. Go to Networking > Firewalls"
echo "  2. Add inbound rules for ports 8000 and 8080"
echo ""
echo -e "${GREEN}Azure:${NC}"
echo "  1. Go to Network Security Groups"
echo "  2. Add inbound security rules for ports 8000 and 8080"

# Check if iptables has any conflicting rules
echo ""
echo -e "${YELLOW}Checking iptables rules...${NC}"
if iptables -L INPUT -n | grep -E "DROP|REJECT" | grep -v "ufw"; then
    echo -e "${RED}⚠️  Warning: Found potentially conflicting iptables rules${NC}"
    echo "You may need to check: sudo iptables -L -n"
fi

# Test connectivity
echo ""
echo -e "${YELLOW}Testing local connectivity...${NC}"

# Check if services are listening
if netstat -tuln | grep -q ":8000"; then
    echo -e "${GREEN}✓ API is listening on port 8000${NC}"
else
    echo -e "${RED}✗ API is NOT listening on port 8000${NC}"
    echo "  Check: docker ps"
    echo "  Check: docker logs logo-detection-api"
fi

if netstat -tuln | grep -q ":8080"; then
    echo -e "${GREEN}✓ Manager is listening on port 8080${NC}"
else
    echo -e "${RED}✗ Manager is NOT listening on port 8080${NC}"
    echo "  Check: sudo systemctl status logo-detection-manager"
fi

echo ""
echo -e "${GREEN}🎉 Firewall setup complete!${NC}"
echo ""
echo "Test from your local machine:"
echo "  curl http://$(curl -s ifconfig.me):8000/api/v1/health"
echo "  curl http://$(curl -s ifconfig.me):8080/"