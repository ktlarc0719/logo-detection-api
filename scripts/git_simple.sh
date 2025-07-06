#!/bin/bash

# Ultra simple git push script
# No fancy features, just basic git commands

echo "Simple Git Push"
echo "==============="

# Add all files
echo "Adding files..."
git add .

# Commit with timestamp
echo "Committing..."
git commit -m "Update $(date +%Y%m%d_%H%M%S)" || echo "No changes to commit"

# Push
echo "Pushing to origin/main..."
git push origin main

echo "Done!"