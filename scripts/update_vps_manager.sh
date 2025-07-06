#!/bin/bash

# Script to update manager.py on VPS with git functionality

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🔄 Updating VPS Manager with Git functionality${NC}"
echo "=============================================="

# Backup existing manager.py
echo -e "${YELLOW}Backing up existing manager.py...${NC}"
sudo cp /opt/logo-detection/manager.py /opt/logo-detection/manager.py.backup

# Create new manager.py
echo -e "${YELLOW}Creating updated manager.py...${NC}"
sudo tee /opt/logo-detection/manager.py > /dev/null << 'EOF'
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

@app.route("/git/pull", methods=["POST"])
def git_pull():
    """Pull latest code from GitHub and optionally rebuild/restart"""
    results = {}
    
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
    
    # Check if rebuild is requested
    rebuild = request.json.get("rebuild", False) if request.json else False
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
EOF

# Install git if not already installed
echo -e "${YELLOW}Installing git...${NC}"
sudo apt-get update
sudo apt-get install -y git

# Restart the manager service
echo -e "${YELLOW}Restarting manager service...${NC}"
sudo systemctl restart logo-detection-manager

echo -e "${GREEN}✅ Manager updated successfully!${NC}"
echo ""
echo "New endpoints available:"
echo "  - GET  /git/status     - Check git repository status"
echo "  - POST /git/pull       - Pull latest code from GitHub"
echo "    Options: {\"rebuild\": true} to rebuild and redeploy"
echo ""
echo "Usage examples:"
echo "  # Pull latest code only"
echo "  curl -X POST http://localhost:8080/git/pull"
echo ""
echo "  # Pull code, rebuild image, and restart container"
echo "  curl -X POST http://localhost:8080/git/pull \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"rebuild\": true}'"