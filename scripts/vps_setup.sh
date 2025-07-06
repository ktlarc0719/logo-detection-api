#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

# 色付き出力
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Logo Detection API VPS Setup Starting...${NC}"

# 1. 必要なパッケージをインストール
echo -e "${YELLOW}📦 Installing required packages...${NC}"
sudo apt update && sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg lsb-release python3 python3-venv python3-pip

# 2. Dockerのインストール
echo -e "${YELLOW}🐳 Installing Docker...${NC}"
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc > /dev/null
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 現在のユーザーをdockerグループに追加
sudo usermod -aG docker $USER

# 3. ローカル用ディレクトリを作成
echo -e "${YELLOW}📁 Creating directories...${NC}"
sudo mkdir -p /opt/logo-detection-api/{models,logs,data,scripts}
cd /opt/logo-detection-api

# 4. Python 仮想環境作成 & Flask インストール
echo -e "${YELLOW}🐍 Setting up Python environment...${NC}"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install flask requests
deactivate

# 5. 環境設定ファイルを作成
echo -e "${YELLOW}⚙️ Creating environment configuration...${NC}"
cat <<'EOF' | sudo tee /opt/logo-detection-api/.env > /dev/null
# 2コア2GB VPS向け最適化設定
MAX_CONCURRENT_DETECTIONS=2
MAX_CONCURRENT_DOWNLOADS=15
MAX_BATCH_SIZE=30
ENVIRONMENT=production
LOG_LEVEL=INFO
PORT=8000
EOF

# 6. pull_restart_server.py を配置
echo -e "${YELLOW}🔧 Creating management API server...${NC}"
cat <<'EOF' | sudo tee /opt/logo-detection-api/pull_restart_server.py > /dev/null
from flask import Flask, jsonify, request
import subprocess
import os
import time

app = Flask(__name__)

# Docker設定
DOCKER_IMAGE = "kentatsujikawadev/logo-detection-api:latest"
CONTAINER_NAME = "logo-detection-api"
API_PORT = 8000
MANAGEMENT_PORT = 8080

def load_env_vars():
    """環境変数を読み込む"""
    env_vars = {}
    env_file = "/opt/logo-detection-api/.env"
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

@app.route("/", methods=["GET"])
def status():
    """コンテナのステータス確認"""
    ps = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Status}}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    logs_tail = subprocess.run(
        ["docker", "logs", "--tail", "20", CONTAINER_NAME],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    return jsonify({
        "container_name": CONTAINER_NAME,
        "status": ps.stdout.strip() or "Not found",
        "recent_logs": logs_tail.stdout + logs_tail.stderr
    }), 200

@app.route("/pull-restart", methods=["POST"])
def pull_and_restart():
    """最新イメージをpullして再起動"""
    results = {}
    
    # 1. 最新イメージをpull
    pull = subprocess.run(
        ["docker", "pull", DOCKER_IMAGE],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    results["pull"] = {"stdout": pull.stdout, "stderr": pull.stderr, "code": pull.returncode}
    
    # 2. 既存コンテナを停止・削除
    stop = subprocess.run(
        ["docker", "stop", CONTAINER_NAME],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    results["stop"] = {"stdout": stop.stdout, "stderr": stop.stderr, "code": stop.returncode}
    
    rm = subprocess.run(
        ["docker", "rm", CONTAINER_NAME],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    results["rm"] = {"stdout": rm.stdout, "stderr": rm.stderr, "code": rm.returncode}
    
    # 3. 環境変数を読み込み
    env_vars = load_env_vars()
    
    # 4. 新しいコンテナを起動
    docker_cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--restart=always",
        "-p", f"{API_PORT}:8000",
        "-v", "/opt/logo-detection-api/models:/app/models",
        "-v", "/opt/logo-detection-api/logs:/app/logs",
        "-v", "/opt/logo-detection-api/data:/app/data"
    ]
    
    # 環境変数を追加
    for key, value in env_vars.items():
        docker_cmd.extend(["-e", f"{key}={value}"])
    
    docker_cmd.append(DOCKER_IMAGE)
    
    run = subprocess.run(
        docker_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    results["run"] = {"stdout": run.stdout, "stderr": run.stderr, "code": run.returncode}
    
    # 5. ヘルスチェック
    time.sleep(5)
    health_check = subprocess.run(
        ["curl", "-s", f"http://localhost:{API_PORT}/api/v1/health"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    results["health_check"] = {
        "stdout": health_check.stdout,
        "stderr": health_check.stderr,
        "code": health_check.returncode
    }
    
    return jsonify(results), 200 if run.returncode == 0 else 500

@app.route("/logs", methods=["GET"])
def get_logs():
    """コンテナのログを取得"""
    lines = request.args.get("lines", "100")
    logs = subprocess.run(
        ["docker", "logs", "--tail", lines, CONTAINER_NAME],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return jsonify({
        "logs": logs.stdout + logs.stderr
    }), 200

@app.route("/config", methods=["GET", "POST"])
def config():
    """環境設定の確認・更新"""
    if request.method == "GET":
        return jsonify(load_env_vars()), 200
    
    elif request.method == "POST":
        config_data = request.json
        if not config_data:
            return jsonify({"error": "No configuration provided"}), 400
        
        # 既存の設定を読み込み
        env_vars = load_env_vars()
        env_vars.update(config_data)
        
        # .envファイルに書き込み
        with open("/opt/logo-detection-api/.env", "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        return jsonify({
            "message": "Configuration updated. Restart container to apply changes.",
            "config": env_vars
        }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=MANAGEMENT_PORT)
EOF

# 7. systemd サービスファイル作成
echo -e "${YELLOW}🔧 Creating systemd service...${NC}"
cat <<EOF | sudo tee /etc/systemd/system/logo-detection-manager.service > /dev/null
[Unit]
Description=Logo Detection API Management Server
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/opt/logo-detection-api/venv/bin/python /opt/logo-detection-api/pull_restart_server.py
WorkingDirectory=/opt/logo-detection-api
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
User=root
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=multi-user.target
EOF

# 8. YOLOモデルのダウンロードスクリプト
echo -e "${YELLOW}📥 Creating model download script...${NC}"
cat <<'EOF' | sudo tee /opt/logo-detection-api/scripts/download_model.sh > /dev/null
#!/bin/bash
MODEL_DIR="/opt/logo-detection-api/models"
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
MODEL_FILE="$MODEL_DIR/yolov8n.pt"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading YOLOv8n model..."
    wget -q --show-progress "$MODEL_URL" -O "$MODEL_FILE"
    echo "Model downloaded successfully!"
else
    echo "Model already exists at $MODEL_FILE"
fi
EOF

sudo chmod +x /opt/logo-detection-api/scripts/download_model.sh

# 9. モデルをダウンロード
echo -e "${YELLOW}📥 Downloading YOLOv8 model...${NC}"
sudo /opt/logo-detection-api/scripts/download_model.sh

# 10. ファイルの権限設定
echo -e "${YELLOW}🔐 Setting permissions...${NC}"
sudo chown -R $USER:$USER /opt/logo-detection-api
sudo chmod -R 755 /opt/logo-detection-api

# 11. systemd サービス起動
echo -e "${YELLOW}🚀 Starting services...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable logo-detection-manager
sudo systemctl restart logo-detection-manager

# 12. 管理サーバーの起動を待つ
echo -e "${YELLOW}⏳ Waiting for management server to start...${NC}"
for i in {1..10}; do
    if curl -s http://127.0.0.1:8080/ > /dev/null; then
        echo -e "${GREEN}✅ Management server is running${NC}"
        break
    else
        echo "Waiting... ($i/10)"
        sleep 2
    fi
done

# 13. 初回のDocker起動
echo -e "${YELLOW}🐳 Starting Logo Detection API container...${NC}"
curl -X POST http://127.0.0.1:8080/pull-restart

# 14. APIの起動を待つ
echo -e "${YELLOW}⏳ Waiting for API to start...${NC}"
sleep 10
for i in {1..20}; do
    if curl -s http://127.0.0.1:8000/api/v1/health > /dev/null; then
        echo -e "${GREEN}✅ Logo Detection API is running${NC}"
        break
    else
        echo "Waiting... ($i/20)"
        sleep 3
    fi
done

# 15. ステータス確認
echo -e "${YELLOW}📊 Checking status...${NC}"
curl -s http://127.0.0.1:8080/ | python3 -m json.tool

echo -e "${GREEN}🎉 Setup completed!${NC}"
echo ""
echo -e "${GREEN}API Endpoints:${NC}"
echo "  - Logo Detection API: http://$(curl -s ifconfig.me):8000"
echo "  - API Documentation: http://$(curl -s ifconfig.me):8000/docs"
echo "  - Batch UI: http://$(curl -s ifconfig.me):8000/ui/batch"
echo ""
echo -e "${GREEN}Management Endpoints:${NC}"
echo "  - Status: http://$(curl -s ifconfig.me):8080/"
echo "  - Pull & Restart: curl -X POST http://$(curl -s ifconfig.me):8080/pull-restart"
echo "  - View Logs: http://$(curl -s ifconfig.me):8080/logs"
echo "  - View/Update Config: http://$(curl -s ifconfig.me):8080/config"
echo ""
echo -e "${GREEN}Useful Commands:${NC}"
echo "  - View container logs: docker logs -f logo-detection-api"
echo "  - View management logs: sudo journalctl -u logo-detection-manager -f"
echo "  - Update config: curl -X POST http://localhost:8080/config -H 'Content-Type: application/json' -d '{\"MAX_CONCURRENT_DETECTIONS\":\"3\"}'"