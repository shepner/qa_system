import logging
from typing import Dict, Any

class BaseDocumentProcessor:
    """Base class for all document processors."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document and return its metadata.
        
        Args:
            file_path: Path to the document to process
            metadata: Initial metadata dictionary with basic file info
            
        Returns:
            Dict containing the processed document metadata
        """
        raise NotImplementedError("Document processors must implement process()") 