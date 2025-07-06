#!/bin/bash

# Port checking script for Docker containers

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🔍 Checking Docker port mappings...${NC}"
echo ""

# Check running containers and their port mappings
echo -e "${GREEN}Running containers:${NC}"
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"

echo ""
echo -e "${GREEN}Port mappings for logo-detection-api:${NC}"
docker port logo-detection-api 2>/dev/null || echo "Container not found or not running"

echo ""
echo -e "${GREEN}Listening ports on host:${NC}"
sudo netstat -tlnp | grep -E ":(8000|8080)" || echo "No services listening on ports 8000 or 8080"

echo ""
echo -e "${GREEN}Docker inspect (port bindings):${NC}"
docker inspect logo-detection-api --format='{{json .NetworkSettings.Ports}}' 2>/dev/null | python3 -m json.tool || echo "Container not found"

echo ""
echo -e "${YELLOW}📝 Current situation:${NC}"

# Check if container is running with correct port mapping
if docker ps | grep -q "0.0.0.0:8000->8000"; then
    echo -e "${GREEN}✓ Port 8000 is correctly mapped${NC}"
else
    echo -e "${RED}✗ Port 8000 mapping issue${NC}"
    echo ""
    echo "To fix, restart container with correct port mapping:"
    echo "docker stop logo-detection-api"
    echo "docker rm logo-detection-api"
    echo "docker run -d --name logo-detection-api --restart=always -p 8000:8000 kentatsujikawadev/logo-detection-api:latest"
fi

echo ""
echo -e "${YELLOW}🔧 Troubleshooting commands:${NC}"
echo ""
echo "1. Test from inside VPS:"
echo "   curl http://localhost:8000/api/v1/health"
echo ""
echo "2. Check firewall:"
echo "   sudo ufw status"
echo ""
echo "3. Check iptables directly:"
echo "   sudo iptables -L -n | grep 8000"
echo ""
echo "4. Test container networking:"
echo "   docker exec logo-detection-api curl http://localhost:8000/api/v1/health"