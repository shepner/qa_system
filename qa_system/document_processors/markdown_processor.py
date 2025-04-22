from typing import Dict, Any
from .base_processor import BaseDocumentProcessor

class MarkdownDocumentProcessor(BaseDocumentProcessor):
    """Document processor for markdown files."""
    
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a markdown document.
        
        Args:
            file_path: Path to the markdown file to process
            metadata: Initial metadata dictionary with basic file info
            
        Returns:
            Dict containing the processed document metadata
        """
        self.logger.debug(f"Processing markdown file: {file_path}")
        return metadata 