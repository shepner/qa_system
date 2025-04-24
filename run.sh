#!/bin/bash

# Exit on error
set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration
VENV_DIR=".venv"
CONFIG_DIR="config"
LOGS_DIR="logs"
DATA_DIR="data/vector_store"
SECRETS_DIR="secrets"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create required directories
create_directories() {
    log_info "Creating required directories..."
    mkdir -p "$LOGS_DIR" "$DATA_DIR" "$CONFIG_DIR" "$SECRETS_DIR"
}

# Setup virtual environment
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    log_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
}

# Install/upgrade dependencies
install_dependencies() {
    log_info "Installing/upgrading dependencies..."
    pip install --upgrade pip
    pip install -q -r requirements.txt
}

# Check configuration
check_config() {
    if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
        if [ -f "$CONFIG_DIR/config.yaml.example" ]; then
            log_warn "config.yaml not found. Copying from example..."
            cp "$CONFIG_DIR/config.yaml.example" "$CONFIG_DIR/config.yaml"
            log_warn "Please update $CONFIG_DIR/config.yaml with your settings"
        else
            log_error "No configuration file found!"
            exit 1
        fi
    fi
}

# Load environment variables
load_env() {
    if [ -f "$SECRETS_DIR/.env" ]; then
        log_info "Loading environment variables..."
        set -a
        source "$SECRETS_DIR/.env"
        set +a
    else
        log_warn "No .env file found in $SECRETS_DIR"
    fi
}

# Main execution
main() {
    cd "$SCRIPT_DIR"
    
    create_directories
    setup_venv
    install_dependencies
    check_config
    load_env

    if [ $# -eq 0 ]; then
        log_info "No arguments provided. Starting interactive mode..."
        python -m qa_system --query
    else
        log_info "Running with arguments: $@"
        python -m qa_system "$@"
    fi
}

# Run main function with all script arguments
main "$@" 