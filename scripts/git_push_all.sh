#!/bin/bash

# Git push script for Logo Detection API
# This script commits and pushes all changes to the repository

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🚀 Git Push Script for Logo Detection API${NC}"
echo "=========================================="

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
git status --short

# Get commit message
echo ""
echo -e "${YELLOW}Enter commit message (or press Enter for auto-generated):${NC}"
read -r commit_message

# If no message provided, generate one
if [ -z "$commit_message" ]; then
    # Count changes
    added=$(git status --porcelain | grep -c "^A" || true)
    modified=$(git status --porcelain | grep -c "^M" || true)
    deleted=$(git status --porcelain | grep -c "^D" || true)
    
    commit_message="Update: "
    [ $added -gt 0 ] && commit_message="${commit_message}${added} added, "
    [ $modified -gt 0 ] && commit_message="${commit_message}${modified} modified, "
    [ $deleted -gt 0 ] && commit_message="${commit_message}${deleted} deleted, "
    commit_message="${commit_message%??}" # Remove trailing comma and space
    
    # Add timestamp
    commit_message="${commit_message} ($(date '+%Y-%m-%d %H:%M:%S'))"
fi

# Commit changes
echo -e "${YELLOW}💾 Committing changes...${NC}"
git commit -m "$commit_message"

# Get current branch
current_branch=$(git rev-parse --abbrev-ref HEAD)
echo -e "${YELLOW}🌿 Current branch: ${current_branch}${NC}"

# Check if we have a remote
if ! git remote -v | grep -q origin; then
    echo -e "${RED}Error: No remote repository configured${NC}"
    echo "Please add a remote repository:"
    echo "  git remote add origin https://github.com/YOUR_USERNAME/logo-detection-api.git"
    exit 1
fi

# Push to remote
echo -e "${YELLOW}📤 Pushing to remote...${NC}"
if git push origin "$current_branch"; then
    echo -e "${GREEN}✓ Successfully pushed to origin/${current_branch}${NC}"
else
    echo -e "${RED}✗ Push failed${NC}"
    echo ""
    echo "If this is a new branch, try:"
    echo "  git push --set-upstream origin $current_branch"
    echo ""
    echo "If authentication failed, check your credentials:"
    echo "  git config --global user.name 'Your Name'"
    echo "  git config --global user.email 'your.email@example.com'"
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
remote_url=$(git config --get remote.origin.url)
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