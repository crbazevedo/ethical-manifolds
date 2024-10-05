#!/bin/bash

# Create a virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the project in editable mode with all dependencies
pip install -e .

echo "Virtual environment created and dependencies installed."
echo "To activate the virtual environment, run: source env/bin/activate"