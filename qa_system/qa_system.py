from typing import Union, Dict, Any
from .config import load_config
from .qa_engine import QAEngine

class QASystem:
    """Main QA System class that coordinates all components."""

    def __init__(self, config_path: Union[str, Dict[str, Any]] = None):
        """Initialize the QA System with configuration."""
        self.config = load_config(config_path)
        self.qa_engine = QAEngine(self.config)

    async def initialize(self):
        """Initialize the QA System components."""
        await self.qa_engine.initialize()

    async def cleanup(self):
        """Clean up system resources before shutdown."""
        if hasattr(self, 'qa_engine') and self.qa_engine is not None:
            await self.qa_engine.cleanup()
            self.qa_engine = None 