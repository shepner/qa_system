#!/bin/bash

# Exit on error
set -e

# Function to print messages
print_step() {
    echo "===> $1"
}

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    print_step "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_step "Upgrading pip..."
python3 -m pip install -q --upgrade pip

# Install/upgrade dependencies
print_step "Installing/upgrading dependencies..."
pip install -q -r requirements.txt

# Run the program
print_step "Starting the program..."
cd "$SCRIPT_DIR"
PYTHONPATH="$SCRIPT_DIR" python3 -m qa_system "$@"

# Deactivate virtual environment
deactivate 