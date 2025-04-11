#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up QA System...${NC}"

# Create required directories
echo -e "${BLUE}Creating required directories...${NC}"
mkdir -p config
mkdir -p secrets

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

# Check if config/config.yaml exists, if not, create from example
if [ ! -f "config/config.yaml" ]; then
    if [ -f "config/config.yaml.example" ]; then
        echo -e "${BLUE}Creating config.yaml from example...${NC}"
        cp config/config.yaml.example config/config.yaml
        echo -e "${GREEN}Created config/config.yaml from example${NC}"
    else
        echo -e "${RED}Warning: No config/config.yaml or config/config.yaml.example found${NC}"
        echo -e "${RED}Please create config/config.yaml before running the application${NC}"
    fi
fi

# Check if secrets/.env exists, if not, create from example
if [ ! -f "secrets/.env" ]; then
    if [ -f "secrets/.env.example" ]; then
        echo -e "${BLUE}Creating .env file from example...${NC}"
        cp secrets/.env.example secrets/.env
        echo -e "${GREEN}Created secrets/.env from example${NC}"
        echo -e "${RED}Please edit secrets/.env with your configuration before running again${NC}"
        exit 1
    else
        echo -e "${RED}Warning: No secrets/.env or secrets/.env.example found${NC}"
        echo -e "${RED}Please create secrets/.env with required environment variables${NC}"
        exit 1
    fi
fi

# Load environment variables from secrets/.env
echo -e "${BLUE}Loading environment variables from secrets/.env...${NC}"
set -a  # automatically export all variables
source secrets/.env
set +a  # stop automatically exporting

# Run the application with all arguments passed through
echo -e "${GREEN}Starting QA System...${NC}"
if [ $# -eq 0 ]; then
    # No arguments provided, run in interactive mode
    python -m qa_system config/config.yaml
else
    # Pass through all provided arguments while ensuring config is first
    python -m qa_system config/config.yaml "$@"
fi 