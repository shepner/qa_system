#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up QA System...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install dependencies if requirements.txt has been modified
if [ ! -f ".venv/.requirements_installed" ] || [ requirements.txt -nt ".venv/.requirements_installed" ]; then
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install -r requirements.txt
    touch .venv/.requirements_installed
fi

# Check if .env exists, if not, create from example
if [ ! -f ".env" ]; then
    echo -e "${BLUE}Creating .env file from example...${NC}"
    cp .env.example .env
    echo -e "${GREEN}Please edit .env file with your configuration before running again${NC}"
    exit 1
fi

# Load environment variables from .env
echo -e "${BLUE}Loading environment variables from .env...${NC}"
set -a  # automatically export all variables
source .env
set +a  # stop automatically exporting

# Run the application with all arguments passed through
echo -e "${GREEN}Starting QA System...${NC}"
python -m qa_system "$@" 