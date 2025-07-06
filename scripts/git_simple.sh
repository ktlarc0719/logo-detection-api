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

# Push (force if needed)
echo "Pushing to origin/main..."
git push origin main || {
    echo ""
    echo "Normal push failed. Choose an option:"
    echo "1) Force push (overwrites remote)"
    echo "2) Pull and merge"
    echo "3) Cancel"
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            echo "Force pushing..."
            git push origin main --force
            ;;
        2)
            echo "Pulling and merging..."
            git pull origin main
            git push origin main
            ;;
        *)
            echo "Cancelled"
            exit 1
            ;;
    esac
}

echo "Done!"