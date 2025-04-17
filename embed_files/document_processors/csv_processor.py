from typing import Dict, Any
from .base_processor import BaseDocumentProcessor

class CSVDocumentProcessor(BaseDocumentProcessor):
    """Document processor for CSV files."""
    
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a CSV document.
        
        Args:
            file_path: Path to the CSV file to process
            metadata: Initial metadata dictionary with basic file info
            
        Returns:
            Dict containing the processed document metadata
        """
        self.logger.debug(f"Processing CSV file: {file_path}")
        return metadata 