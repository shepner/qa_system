from typing import Dict, Any
from .base_processor import BaseDocumentProcessor

class PDFDocumentProcessor(BaseDocumentProcessor):
    """Document processor for PDF files."""
    
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a PDF document.
        
        Args:
            file_path: Path to the PDF file to process
            metadata: Initial metadata dictionary with basic file info
            
        Returns:
            Dict containing the processed document metadata
        """
        self.logger.debug(f"Processing PDF file: {file_path}")
        return metadata 