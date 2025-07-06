#!/bin/bash

echo "=== Git Test Script ==="
echo "Testing basic git commands..."

echo ""
echo "1. Testing pwd:"
pwd

echo ""
echo "2. Testing git version:"
git --version

echo ""
echo "3. Testing git status (simple):"
git status 2>&1 | head -5

echo ""
echo "4. Testing git remote:"
git remote -v

echo ""
echo "5. Testing git branch:"
git branch

echo ""
echo "Script completed successfully!"