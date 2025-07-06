#!/bin/bash

# Debug version of git push script
# Shows each step in detail

# Enable debug mode
set -x

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🚀 Git Push Script (Debug Mode)${NC}"
echo "===================================="

# Basic checks
echo "Checking git installation..."
which git || { echo "Git not found"; exit 1; }

echo "Checking current directory..."
pwd

echo "Checking .git directory..."
ls -la .git || { echo ".git directory not found"; exit 1; }

echo "Git status..."
git status

echo "Git status --porcelain..."
git status --porcelain

# Check for changes
if [ -z "$(git status --porcelain)" ]; then
    echo "No changes to commit"
    exit 0
fi

echo "Adding files..."
git add -A

echo "Status after add..."
git status

# Simple commit
echo "Committing with simple message..."
git commit -m "Update files $(date '+%Y-%m-%d %H:%M:%S')" || { echo "Commit failed"; exit 1; }

echo "Getting current branch..."
branch=$(git branch --show-current)
echo "Current branch: $branch"

echo "Checking remotes..."
git remote -v

echo "Pushing..."
git push origin "$branch" || {
    echo "Push failed, trying with --set-upstream..."
    git push --set-upstream origin "$branch"
}

echo "Done!"