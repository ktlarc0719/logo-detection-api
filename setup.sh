#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/YOUR_USERNAME/logo-detection-api.git"
PROJECT_NAME="logo-detection-api"
PROJECT_DIR="/opt/${PROJECT_NAME}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Logo Detection API VPS Setup Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}" 
   exit 1
fi

# Update system
echo -e "${YELLOW}Updating system packages...${NC}"
apt-get update
apt-get upgrade -y

# Install basic dependencies
echo -e "${YELLOW}Installing basic dependencies...${NC}"
apt-get install -y \
    curl \
    git \
    vim \
    htop \
    ufw \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Docker
echo -e "${YELLOW}Installing Docker...${NC}"
if ! command -v docker &> /dev/null; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io
    systemctl enable docker
    systemctl start docker
else
    echo -e "${GREEN}Docker is already installed${NC}"
fi

# Install Docker Compose
echo -e "${YELLOW}Installing Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_VERSION="2.23.0"
    curl -L "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
else
    echo -e "${GREEN}Docker Compose is already installed${NC}"
fi

# Configure firewall
echo -e "${YELLOW}Configuring firewall...${NC}"
ufw --force enable
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8000/tcp
ufw reload

# Clone repository
echo -e "${YELLOW}Cloning repository...${NC}"
if [ -d "$PROJECT_DIR" ]; then
    echo -e "${YELLOW}Project directory already exists. Pulling latest changes...${NC}"
    cd "$PROJECT_DIR"
    git pull origin main
else
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# Create .env file if it doesn't exist
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo -e "${RED}Please edit .env file with your configuration${NC}"
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/data/temp"
mkdir -p "$PROJECT_DIR/nginx/ssl"

# Set permissions
echo -e "${YELLOW}Setting permissions...${NC}"
chown -R 1000:1000 "$PROJECT_DIR/models"
chown -R 1000:1000 "$PROJECT_DIR/logs"
chown -R 1000:1000 "$PROJECT_DIR/data"

# Create nginx configuration if it doesn't exist
if [ ! -f "$PROJECT_DIR/nginx/nginx.conf" ]; then
    echo -e "${YELLOW}Creating nginx configuration...${NC}"
    cat > "$PROJECT_DIR/nginx/nginx.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        server_name _;

        client_max_body_size 100M;

        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }

        location /health {
            proxy_pass http://api/health;
            access_log off;
        }
    }
}
EOF
fi

# Create update script
echo -e "${YELLOW}Creating update script...${NC}"
cat > "$PROJECT_DIR/update.sh" << 'EOF'
#!/bin/bash
cd /opt/logo-detection-api
git pull origin main
docker-compose -f docker-compose.production.yml build
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d
docker-compose -f docker-compose.production.yml logs -f --tail=100
EOF
chmod +x "$PROJECT_DIR/update.sh"

# Create systemd service for auto-update endpoint
echo -e "${YELLOW}Creating auto-update service...${NC}"
cat > /etc/systemd/system/logo-detection-update.service << EOF
[Unit]
Description=Logo Detection API Update Service
After=network.target

[Service]
Type=oneshot
ExecStart=$PROJECT_DIR/update.sh
User=root
Group=root

[Install]
WantedBy=multi-user.target
EOF

# Build and start services
echo -e "${YELLOW}Building and starting services...${NC}"
cd "$PROJECT_DIR"
docker-compose -f docker-compose.production.yml build
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Check if services are running
echo -e "${YELLOW}Checking service status...${NC}"
docker-compose -f docker-compose.production.yml ps

# Test API endpoint
echo -e "${YELLOW}Testing API endpoint...${NC}"
if curl -f http://localhost:8000/health; then
    echo -e "${GREEN}API is running successfully!${NC}"
else
    echo -e "${RED}API health check failed. Check logs with: docker-compose -f docker-compose.production.yml logs${NC}"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env file: vim $PROJECT_DIR/.env"
echo "2. Place your model files in: $PROJECT_DIR/models/"
echo "3. Monitor logs: docker-compose -f docker-compose.production.yml logs -f"
echo "4. Update application: $PROJECT_DIR/update.sh"
echo ""
echo "API endpoints:"
echo "- Health check: http://your-server-ip/health"
echo "- API docs: http://your-server-ip/docs"
echo ""