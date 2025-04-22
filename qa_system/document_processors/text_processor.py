from typing import Dict, Any
from .base_processor import BaseDocumentProcessor

class TextDocumentProcessor(BaseDocumentProcessor):
    """Document processor for plain text files."""
    
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a text document.
        
        Args:
            file_path: Path to the text file to process
            metadata: Initial metadata dictionary with basic file info
            
        Returns:
            Dict containing the processed document metadata
        """
        self.logger.debug(f"Processing text file: {file_path}")
        return metadata 