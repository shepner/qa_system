from typing import Dict, Any
from .base_processor import BaseDocumentProcessor

class ImageDocumentProcessor(BaseDocumentProcessor):
    """Document processor for image files."""
    
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process an image document.
        
        Args:
            file_path: Path to the image file to process
            metadata: Initial metadata dictionary with basic file info
            
        Returns:
            Dict containing the processed document metadata
        """
        self.logger.debug(f"Processing image file: {file_path}")
        return metadata 