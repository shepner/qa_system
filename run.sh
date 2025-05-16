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

# Set the log level for the application (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL="WARNING"  # Change this to control the default log level for the app

# Log level priorities
LOG_LEVELS=("DEBUG" "INFO" "WARNING" "ERROR" "CRITICAL")

# Function to get log level index
get_log_level_index() {
    local level="$1"
    for i in "${!LOG_LEVELS[@]}"; do
        if [[ "${LOG_LEVELS[$i]}" == "$level" ]]; then
            echo "$i"
            return
        fi
    done
    # Default to highest if not found
    echo "4"
}

# Function to check if a message should be logged
should_log() {
    local msg_level="$1"
    local msg_index
    local log_index
    msg_index=$(get_log_level_index "$msg_level")
    log_index=$(get_log_level_index "$LOG_LEVEL")
    if (( msg_index >= log_index )); then
        return 0
    else
        return 1
    fi
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    if should_log "INFO"; then
        echo -e "${GREEN}[INFO]${NC} $1"
    fi
}

log_warn() {
    if should_log "WARNING"; then
        echo -e "${YELLOW}[WARN]${NC} $1"
    fi
}

log_error() {
    if should_log "ERROR"; then
        echo -e "${RED}[ERROR]${NC} $1"
    fi
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
    pip install --upgrade pip > /dev/null 2>&1
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
    load_env  # Load any updated environment variables

    # Check if --log-level is supported
    if python -m qa_system --help 2>&1 | grep -q -- '--log-level'; then
        LOG_LEVEL_ARG=(--log-level "$LOG_LEVEL")
    else
        LOG_LEVEL_ARG=()
    fi

    if [ $# -eq 0 ]; then
        log_info "No arguments provided. Starting interactive mode..."
        python -m qa_system --query "${LOG_LEVEL_ARG[@]}"
    else
        log_info "Running with arguments: $@"
        python -m qa_system "$@" "${LOG_LEVEL_ARG[@]}"
    fi
}

# Run main function with all script arguments
main "$@" 