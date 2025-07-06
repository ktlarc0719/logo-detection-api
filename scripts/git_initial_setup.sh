#!/bin/bash

# Initial Git setup script for Logo Detection API
# Run this once to initialize the repository

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🔧 Git Initial Setup for Logo Detection API${NC}"
echo "============================================"

# Initialize git if not already initialized
if [ ! -d .git ]; then
    echo -e "${YELLOW}Initializing git repository...${NC}"
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
else
    echo -e "${GREEN}✓ Git repository already initialized${NC}"
fi

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo -e "${YELLOW}Creating .gitignore...${NC}"
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
logs/
*.pt
*.onnx
*.engine
data/
models/downloaded/
.env
.env.local
.env.*.local

# Docker
.dockerignore

# Test
.coverage
htmlcov/
.pytest_cache/
.tox/

# Jupyter Notebook
.ipynb_checkpoints

# Build artifacts
build.log
performance_test_results_*.json
simple_performance_test_*.json
EOF
    echo -e "${GREEN}✓ .gitignore created${NC}"
else
    echo -e "${GREEN}✓ .gitignore already exists${NC}"
fi

# Set git user if not set
if [ -z "$(git config user.name)" ]; then
    echo -e "${YELLOW}Setting up git user...${NC}"
    echo "Enter your name:"
    read -r user_name
    git config user.name "$user_name"
    
    echo "Enter your email:"
    read -r user_email
    git config user.email "$user_email"
    
    echo -e "${GREEN}✓ Git user configured${NC}"
else
    echo -e "${GREEN}✓ Git user already configured:${NC}"
    echo "  Name: $(git config user.name)"
    echo "  Email: $(git config user.email)"
fi

# Add all files
echo -e "${YELLOW}Adding all files...${NC}"
git add -A

# Initial commit
if [ -z "$(git log --oneline -1 2>/dev/null)" ]; then
    echo -e "${YELLOW}Creating initial commit...${NC}"
    git commit -m "Initial commit: Logo Detection API"
    echo -e "${GREEN}✓ Initial commit created${NC}"
else
    echo -e "${GREEN}✓ Repository already has commits${NC}"
fi

# Check for remote
if ! git remote -v | grep -q origin; then
    echo ""
    echo -e "${YELLOW}No remote repository configured${NC}"
    echo "Would you like to add a GitHub repository? (y/n)"
    read -r add_remote
    
    if [[ $add_remote == "y" || $add_remote == "Y" ]]; then
        echo "Enter your GitHub username:"
        read -r github_username
        
        echo "Enter repository name (default: logo-detection-api):"
        read -r repo_name
        repo_name=${repo_name:-logo-detection-api}
        
        echo "Use SSH (recommended) or HTTPS? (ssh/https):"
        read -r protocol
        
        if [[ $protocol == "ssh" || $protocol == "SSH" ]]; then
            remote_url="git@github.com:${github_username}/${repo_name}.git"
        else
            remote_url="https://github.com/${github_username}/${repo_name}.git"
        fi
        
        git remote add origin "$remote_url"
        echo -e "${GREEN}✓ Remote repository added: $remote_url${NC}"
        
        echo ""
        echo -e "${YELLOW}Note: You need to create the repository on GitHub first:${NC}"
        echo "  1. Go to https://github.com/new"
        echo "  2. Repository name: $repo_name"
        echo "  3. Don't initialize with README (we already have files)"
        echo "  4. Create repository"
        echo "  5. Then run: ./scripts/git_push_all.sh"
    fi
else
    echo -e "${GREEN}✓ Remote repository already configured:${NC}"
    git remote -v
fi

echo ""
echo -e "${GREEN}🎉 Git setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Make sure your GitHub repository exists"
echo "  2. Run: ./scripts/git_push_all.sh"
echo ""
echo "For regular updates:"
echo "  ./scripts/git_push_all.sh"