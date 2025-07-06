#!/bin/bash

# Force push script - use with caution!

echo "⚠️  FORCE GIT PUSH ⚠️"
echo "===================="
echo "This will overwrite the remote repository!"
echo ""

# Add all files
echo "Adding files..."
git add .

# Commit
echo "Committing..."
git commit -m "Force update $(date +%Y%m%d_%H%M%S)" || echo "No changes to commit"

# Force push
echo "Force pushing to origin/main..."
git push origin main --force

echo "✓ Force push completed!"