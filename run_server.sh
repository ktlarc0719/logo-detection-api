#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run the server
echo "Starting server..."
cd /root/projects/logo-detection-api
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000