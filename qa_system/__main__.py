"""
Main entry point for the QA system
"""
from .config import load_config
from .cli import run

if __name__ == "__main__":
    run() 