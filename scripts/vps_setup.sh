#!/bin/bash
set -e

# 色付き出力
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Logo Detection API VPS Setup (Final Version)${NC}"
echo "================================================"

# 1. 環境変数を設定して再起動プロンプトを無効化
echo -e "${YELLOW}🔧 Configuring system to avoid restart prompts...${NC}"
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

# needrestartの設定を更新（再起動プロンプトを完全に無効化）
if [ -f /etc/needrestart/needrestart.conf ]; then
    sudo sed -i "s/#\$nrconf{restart} = 'i';/\$nrconf{restart} = 'a';/" /etc/needrestart/needrestart.conf
    sudo sed -i "s/\$nrconf{restart} = 'i';/\$nrconf{restart} = 'a';/" /etc/needrestart/needrestart.conf
fi

# 2. パッケージの更新（再起動プロンプトなし）
echo -e "${YELLOW}📦 Updating system packages...${NC}"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"

# 3. 必要なパッケージをインストール
echo -e "${YELLOW}📦 Installing required packages...${NC}"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    python3 \
    python3-venv \
    python3-pip \
    git \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold"

# 4. Dockerのインストール（既にインストールされている場合はスキップ）
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}🐳 Installing Docker...${NC}"
    
    # Dockerのリポジトリを追加
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Dockerをインストール
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin \
        -o Dpkg::Options::="--force-confdef" \
        -o Dpkg::Options::="--force-confold"
    
    # 現在のユーザーをdockerグループに追加
    sudo usermod -aG docker $USER
    
    # Dockerサービスを開始
    sudo systemctl start docker
    sudo systemctl enable docker
else
    echo -e "${GREEN}✓ Docker already installed${NC}"
fi

# 5. アプリケーション用ディレクトリを作成
echo -e "${YELLOW}📁 Creating application directories...${NC}"
sudo mkdir -p /opt/logo-detection/{logs,data}
cd /opt/logo-detection

# 6. 環境設定ファイルを作成
echo -e "${YELLOW}⚙️ Creating environment configuration...${NC}"
cat <<'EOF' | sudo tee /opt/logo-detection/.env > /dev/null
# Logo Detection API Configuration
# Optimized for 2-core 2GB VPS

# Performance Settings (Optimized for 2GB VPS)
MAX_CONCURRENT_DETECTIONS=2
MAX_CONCURRENT_DOWNLOADS=30
MAX_BATCH_SIZE=30

# Memory Settings
MEMORY_LIMIT=1g
MEMORY_RESERVATION=512m

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO

# API Settings
PORT=8000
EOF

# 7. Python仮想環境のセットアップ
echo -e "${YELLOW}🐍 Setting up Python virtual environment...${NC}"
python3 -m venv /opt/logo-detection/venv

# 仮想環境内でパッケージをインストール
/opt/logo-detection/venv/bin/pip install --upgrade pip
/opt/logo-detection/venv/bin/pip install flask

# 8. 管理APIサーバーを作成
echo -e "${YELLOW}📝 Creating management API server...${NC}"
cat <<'EOF' | sudo tee /opt/logo-detection/manager.py > /dev/null
from flask import Flask, jsonify, request
import subprocess
import os
import json
import time
from datetime import datetime

app = Flask(__name__)

# Configuration
DOCKER_IMAGE = "kentatsujikawadev/logo-detection-api:latest"
CONTAINER_NAME = "logo-detection-api"
API_PORT = 8000
GIT_REPO_URL = "https://github.com/ktlarc0719/logo-detection-api.git"
GIT_REPO_DIR = "/opt/logo-detection/repo"

def load_env():
    """Load environment variables from .env file"""
    env_vars = {}
    env_file = "/opt/logo-detection/.env"
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

def run_command(cmd):
    """Execute command and return result"""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=isinstance(cmd, str))
    return {
        "command": cmd if isinstance(cmd, str) else " ".join(cmd),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "success": result.returncode == 0
    }

@app.route("/", methods=["GET"])
def index():
    """Get current status"""
    # Container status
    ps_result = run_command(["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}\\t{{.Status}}\\t{{.Ports}}"])
    
    # Current configuration
    env_vars = load_env()
    
    # API health check
    health = run_command(f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{API_PORT}/api/v1/health || echo '000'")
    api_healthy = health["stdout"].strip() == "200"
    
    # Git status
    git_status = {}
    if os.path.exists(GIT_REPO_DIR):
        git_result = run_command(f"cd {GIT_REPO_DIR} && git log -1 --oneline")
        git_status["last_commit"] = git_result["stdout"].strip()
        git_branch = run_command(f"cd {GIT_REPO_DIR} && git branch --show-current")
        git_status["branch"] = git_branch["stdout"].strip()
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "container": {
            "name": CONTAINER_NAME,
            "status": ps_result["stdout"].strip(),
            "api_healthy": api_healthy
        },
        "configuration": env_vars,
        "git": git_status
    }), 200

@app.route("/deploy", methods=["POST"])
def deploy():
    """Pull latest image and restart container"""
    results = {}
    
    # Pull latest image
    print("Pulling latest image...")
    results["pull"] = run_command(["docker", "pull", DOCKER_IMAGE])
    
    # Stop and remove existing container
    results["stop"] = run_command(["docker", "stop", CONTAINER_NAME])
    results["rm"] = run_command(["docker", "rm", CONTAINER_NAME])
    
    # Load environment variables
    env_vars = load_env()
    
    # Build docker run command
    docker_cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--restart=always",
        "--memory", env_vars.get("MEMORY_LIMIT", "1g"),
        "--memory-reservation", env_vars.get("MEMORY_RESERVATION", "512m"),
        "-p", f"{API_PORT}:8000",
        "-v", "/opt/logo-detection/logs:/app/logs",
        "-v", "/opt/logo-detection/data:/app/data"
    ]
    
    # Add environment variables
    for key, value in env_vars.items():
        docker_cmd.extend(["-e", f"{key}={value}"])
    
    docker_cmd.append(DOCKER_IMAGE)
    
    # Start new container
    print("Starting new container...")
    results["run"] = run_command(docker_cmd)
    
    # Wait for health check
    time.sleep(5)
    health = run_command(f"curl -s http://localhost:{API_PORT}/api/v1/health || echo '{{}}'")
    results["health"] = {
        "success": bool(health["stdout"].strip()),
        "response": health["stdout"]
    }
    
    return jsonify(results), 200 if results["run"]["success"] else 500

@app.route("/logs", methods=["GET"])
def logs():
    """Get container logs"""
    lines = request.args.get("lines", "100")
    result = run_command(["docker", "logs", "--tail", lines, CONTAINER_NAME])
    return jsonify({
        "logs": result["stdout"] + result["stderr"]
    }), 200

@app.route("/config", methods=["GET", "POST"])
def config():
    """Get or update configuration"""
    if request.method == "GET":
        return jsonify(load_env()), 200
    
    # Update configuration
    new_config = request.json
    if not new_config:
        return jsonify({"error": "No configuration provided"}), 400
    
    env_vars = load_env()
    env_vars.update(new_config)
    
    # Write to file
    with open("/opt/logo-detection/.env", "w") as f:
        f.write(f"# Updated: {datetime.now().isoformat()}\n")
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    return jsonify({
        "message": "Configuration updated. Run /deploy to apply.",
        "config": env_vars
    }), 200

@app.route("/git/pull", methods=["POST", "GET"])
def git_pull():
    """Pull latest code from GitHub and optionally rebuild/restart"""
    results = {}
    
    # For GET requests, just pull without rebuild
    if request.method == "GET":
        rebuild = False
    else:
        # For POST requests, try to get JSON data
        try:
            data = request.get_json(silent=True) or {}
        except:
            data = {}
        rebuild = data.get("rebuild", True)
    
    # Check if repo exists, clone if not
    if not os.path.exists(GIT_REPO_DIR):
        print(f"Cloning repository to {GIT_REPO_DIR}...")
        os.makedirs(os.path.dirname(GIT_REPO_DIR), exist_ok=True)
        results["clone"] = run_command(f"git clone {GIT_REPO_URL} {GIT_REPO_DIR}")
    else:
        # Pull latest changes
        print("Pulling latest changes from GitHub...")
        results["pull"] = run_command(f"cd {GIT_REPO_DIR} && git pull origin main")
    
    # Get current commit info
    commit_info = run_command(f"cd {GIT_REPO_DIR} && git log -1 --pretty=format:'%h - %s (%cr)'")
    results["current_commit"] = commit_info["stdout"]
    if rebuild:
        print("Rebuilding Docker image...")
        results["build"] = run_command(f"cd {GIT_REPO_DIR} && docker build -t {DOCKER_IMAGE} .")
        
        # If build successful, restart container
        if results["build"]["success"]:
            deploy_result = deploy()
            results["deploy"] = deploy_result[0].json
    
    return jsonify({
        "success": True,
        "message": "Git pull completed",
        "results": results
    }), 200

@app.route("/git/status", methods=["GET"])
def git_status():
    """Get current git status"""
    if not os.path.exists(GIT_REPO_DIR):
        return jsonify({
            "error": "Repository not found",
            "message": f"No repository at {GIT_REPO_DIR}. Use /git/pull to clone."
        }), 404
    
    results = {}
    results["status"] = run_command(f"cd {GIT_REPO_DIR} && git status --short")
    results["branch"] = run_command(f"cd {GIT_REPO_DIR} && git branch --show-current")
    results["last_commit"] = run_command(f"cd {GIT_REPO_DIR} && git log -1 --pretty=format:'%h - %s (%cr) <%an>'")
    results["remote_url"] = run_command(f"cd {GIT_REPO_DIR} && git config --get remote.origin.url")
    
    return jsonify({
        "repository": GIT_REPO_DIR,
        "branch": results["branch"]["stdout"].strip(),
        "last_commit": results["last_commit"]["stdout"].strip(),
        "remote_url": results["remote_url"]["stdout"].strip(),
        "changes": results["status"]["stdout"].strip()
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
EOF

# 9. systemdサービスファイルを作成
echo -e "${YELLOW}🔧 Creating systemd service...${NC}"
cat <<EOF | sudo tee /etc/systemd/system/logo-detection-manager.service > /dev/null
[Unit]
Description=Logo Detection API Manager
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/opt/logo-detection/venv/bin/python /opt/logo-detection/manager.py
WorkingDirectory=/opt/logo-detection
Restart=always
RestartSec=5
User=root
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=multi-user.target
EOF

# 10. パーミッション設定
echo -e "${YELLOW}🔐 Setting permissions...${NC}"
sudo chown -R root:root /opt/logo-detection
sudo chmod -R 755 /opt/logo-detection

# 11. ファイアウォールの設定
echo -e "${YELLOW}🔥 Configuring firewall...${NC}"

# UFWがインストールされていない場合はインストール
if ! command -v ufw &> /dev/null; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ufw
fi

# UFWの設定
sudo ufw --force disable  # 一旦無効化
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 必要なポートを開放
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 8000/tcp comment 'Logo Detection API'
sudo ufw allow 8080/tcp comment 'Management API'
sudo ufw allow 80/tcp comment 'HTTP'
sudo ufw allow 443/tcp comment 'HTTPS'

# UFWを有効化（--forceで確認プロンプトをスキップ）
echo "y" | sudo ufw --force enable

echo -e "${GREEN}✓ Firewall configured${NC}"
sudo ufw status numbered

# 12. サービスを開始
echo -e "${YELLOW}🚀 Starting management service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable logo-detection-manager
sudo systemctl restart logo-detection-manager

# 12. 管理サーバーの起動を待つ
echo -e "${YELLOW}⏳ Waiting for management server to start...${NC}"
MAX_ATTEMPTS=20
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8080/ > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Management server is running${NC}"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "Waiting... ($ATTEMPT/$MAX_ATTEMPTS)"
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "${RED}✗ Management server failed to start${NC}"
    echo "Checking logs..."
    sudo journalctl -u logo-detection-manager --no-pager -n 50
    exit 1
fi

# 13. 初回デプロイ
echo -e "${YELLOW}🐳 Deploying Logo Detection API...${NC}"
DEPLOY_RESULT=$(curl -s -X POST http://localhost:8080/deploy)
echo "$DEPLOY_RESULT" | python3 -m json.tool || echo "$DEPLOY_RESULT"

# 14. APIの起動を待つ
echo -e "${YELLOW}⏳ Waiting for API to start...${NC}"
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is running${NC}"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "Waiting... ($ATTEMPT/$MAX_ATTEMPTS)"
    sleep 2
done

# 15. 最終ステータス確認
echo -e "${YELLOW}📊 Final status check...${NC}"
curl -s http://localhost:8080/ | python3 -m json.tool || curl -s http://localhost:8080/

# 16. IPアドレスを取得（複数の方法を試す）
PUBLIC_IP=$(curl -s -4 ifconfig.me || curl -s -4 icanhazip.com || curl -s -4 ident.me || echo "YOUR_SERVER_IP")

# 17. 完了メッセージ
echo ""
echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo ""
echo -e "${GREEN}📍 Access Points:${NC}"
echo "  API: http://${PUBLIC_IP}:8000"
echo "  API Docs: http://${PUBLIC_IP}:8000/docs"
echo "  Batch UI: http://${PUBLIC_IP}:8000/ui/batch"
echo "  Manager: http://${PUBLIC_IP}:8080"
echo ""
echo -e "${GREEN}🔧 Quick Commands:${NC}"
echo "  # Check status"
echo "  curl http://localhost:8080/"
echo ""
echo "  # View logs"
echo "  curl http://localhost:8080/logs?lines=50"
echo ""
echo "  # Update configuration"
echo "  curl -X POST http://localhost:8080/config \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"MAX_BATCH_SIZE\":\"50\"}'"
echo ""
echo "  # Deploy latest version (from Docker Hub)"
echo "  curl -X POST http://localhost:8080/deploy"
echo ""
echo "  # Git status"
echo "  curl http://localhost:8080/git/status"
echo ""
echo "  # Pull code + rebuild + deploy (default)"
echo "  curl -X POST http://localhost:8080/git/pull"
echo ""
echo "  # Pull code only (no rebuild)"
echo "  curl -X POST http://localhost:8080/git/pull \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"rebuild\": false}'"
echo ""

# 18. 再起動が必要かチェック
if [ -f /var/run/reboot-required ]; then
    echo -e "${YELLOW}⚠️  System restart required. Please reboot when convenient.${NC}"
    echo "  sudo reboot"
else
    echo -e "${GREEN}✓ No system restart required${NC}"
fi

# 19. クラウドプロバイダー別の追加設定案内
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: Cloud Provider Firewall Settings${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "The UFW firewall has been configured, but you may also need to configure"
echo "your cloud provider's firewall/security group settings:"
echo ""
echo -e "${GREEN}AWS EC2:${NC}"
echo "  Security Groups → Edit inbound rules → Add:"
echo "  - Type: Custom TCP, Port: 8000, Source: 0.0.0.0/0"
echo "  - Type: Custom TCP, Port: 8080, Source: 0.0.0.0/0"
echo ""
echo -e "${GREEN}Google Cloud Platform:${NC}"
echo "  VPC network → Firewall → Create firewall rule:"
echo "  - Direction: Ingress, Targets: Specified tags"
echo "  - Source IP ranges: 0.0.0.0/0"
echo "  - Protocols and ports: tcp:8000,8080"
echo ""
echo -e "${GREEN}Azure:${NC}"
echo "  Network security group → Inbound security rules → Add:"
echo "  - Port: 8000, Protocol: TCP, Action: Allow"
echo "  - Port: 8080, Protocol: TCP, Action: Allow"
echo ""
echo -e "${GREEN}DigitalOcean:${NC}"
echo "  Networking → Firewalls → Create/Edit Firewall:"
echo "  - Inbound Rules: TCP 8000, TCP 8080"
echo ""
echo -e "${GREEN}Vultr/Linode:${NC}"
echo "  Usually only UFW configuration is needed (already done!)"
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"