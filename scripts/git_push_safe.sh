#!/bin/bash

# Safe Git push script with better error handling
# This script commits and pushes all changes to the repository

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🚀 Git Push Script for Logo Detection API${NC}"
echo "=========================================="

# Error handler
handle_error() {
    echo -e "${RED}Error occurred at line $1${NC}"
    exit 1
}

# Set error trap
trap 'handle_error $LINENO' ERR

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Show current status
echo -e "${YELLOW}📊 Current git status:${NC}"
git status --short

# Check if there are changes
if [ -z "$(git status --porcelain)" ]; then
    echo -e "${GREEN}✓ No changes to commit${NC}"
    echo "Repository is up to date!"
    exit 0
fi

# Add all files
echo -e "${YELLOW}📁 Adding all files...${NC}"
git add -A

# Show what will be committed
echo -e "${YELLOW}📝 Changes to be committed:${NC}"
git diff --cached --stat

# Get commit message
echo ""
echo -e "${YELLOW}Enter commit message (or press Enter for auto-generated):${NC}"
read -r commit_message

# If no message provided, generate one
if [ -z "$commit_message" ]; then
    # Count changes safely
    file_count=$(git diff --cached --name-only | wc -l)
    commit_message="Update ${file_count} file(s) - $(date '+%Y-%m-%d %H:%M:%S')"
fi

# Commit changes
echo -e "${YELLOW}💾 Committing changes...${NC}"
if ! git commit -m "$commit_message"; then
    echo -e "${RED}Commit failed${NC}"
    exit 1
fi

# Get current branch
current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
echo -e "${YELLOW}🌿 Current branch: ${current_branch}${NC}"

# Check if we have a remote
if ! git remote | grep -q "origin"; then
    echo -e "${RED}Error: No remote repository configured${NC}"
    echo "Please add a remote repository:"
    echo "  git remote add origin https://github.com/YOUR_USERNAME/logo-detection-api.git"
    exit 1
fi

# Check if branch exists on remote
echo -e "${YELLOW}🔍 Checking remote branch...${NC}"
if git ls-remote --heads origin | grep -q "refs/heads/${current_branch}"; then
    # Branch exists, do normal push
    echo -e "${YELLOW}📤 Pushing to existing branch...${NC}"
    push_command="git push origin ${current_branch}"
else
    # New branch, need to set upstream
    echo -e "${YELLOW}📤 Pushing new branch...${NC}"
    push_command="git push --set-upstream origin ${current_branch}"
fi

# Execute push
if eval "$push_command"; then
    echo -e "${GREEN}✓ Successfully pushed to origin/${current_branch}${NC}"
else
    echo -e "${RED}✗ Push failed${NC}"
    echo ""
    echo "Possible solutions:"
    echo "1. Check your internet connection"
    echo "2. Verify your Git credentials:"
    echo "   git config user.name"
    echo "   git config user.email"
    echo "3. For authentication issues, try:"
    echo "   - Using personal access token instead of password"
    echo "   - Setting up SSH keys"
    exit 1
fi

# Show summary
echo ""
echo -e "${GREEN}🎉 Git push completed successfully!${NC}"
echo ""
echo "Summary:"
echo "  Branch: $current_branch"
echo "  Commit: $(git rev-parse --short HEAD)"
echo "  Message: $commit_message"
echo ""

# Show remote URL
remote_url=$(git config --get remote.origin.url || echo "No remote URL")
echo "Remote repository: $remote_url"

# If it's a GitHub repository, show the URL
if [[ $remote_url == *"github.com"* ]]; then
    # Convert SSH URL to HTTPS if needed
    if [[ $remote_url == git@github.com:* ]]; then
        repo_path=${remote_url#git@github.com:}
        repo_path=${repo_path%.git}
        web_url="https://github.com/$repo_path"
    else
        web_url=${remote_url%.git}
    fi
    echo "View on GitHub: $web_url"
fi