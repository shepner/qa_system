from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from datetime import datetime
import os
from PIL import Image
from google.cloud import vision
from .base_processor import BaseDocumentProcessor

class ImageDocumentProcessor(BaseDocumentProcessor):
    """Document processor for image files.
    
    Handles processing of image files according to the architecture specification.
    Features:
    - Image analysis using Google Cloud Vision API
    - Text extraction through OCR
    - Object detection and labeling
    - Safe search annotations
    - Image properties extraction
    - Metadata extraction
    """
    
    SUPPORTED_FORMATS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}
    
    def __init__(self, config=None):
        """Initialize the image document processor.
        
        Args:
            config: Configuration dictionary or object (optional)
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Vision client
        try:
            self.vision_client = vision.ImageAnnotatorClient()
        except Exception as e:
            self.logger.error(f"Failed to initialize Vision client: {e}")
            self.vision_client = None
        
        # Set default configuration
        self.enable_text_detection = True
        self.enable_label_detection = True
        self.enable_object_detection = True
        self.enable_safe_search = True
        self.enable_image_properties = True
        self.max_labels = 20
        self.max_objects = 20
        
        # Update configuration if provided
        if config:
            if hasattr(config, 'get_nested'):
                doc_processing = config.get_nested('DOCUMENT_PROCESSING', {})
                vision_config = config.get_nested('DOCUMENT_PROCESSING.VISION_PROCESSOR', {})
            else:
                doc_processing = config.get('DOCUMENT_PROCESSING', {})
                vision_config = doc_processing.get('VISION_PROCESSOR', {})
            
            self.enable_text_detection = vision_config.get('ENABLE_TEXT_DETECTION', self.enable_text_detection)
            self.enable_label_detection = vision_config.get('ENABLE_LABEL_DETECTION', self.enable_label_detection)
            self.enable_object_detection = vision_config.get('ENABLE_OBJECT_DETECTION', self.enable_object_detection)
            self.enable_safe_search = vision_config.get('ENABLE_SAFE_SEARCH', self.enable_safe_search)
            self.enable_image_properties = vision_config.get('ENABLE_IMAGE_PROPERTIES', self.enable_image_properties)
            self.max_labels = vision_config.get('MAX_LABELS', self.max_labels)
            self.max_objects = vision_config.get('MAX_OBJECTS', self.max_objects)
    
    def _get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract basic image metadata using PIL.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary of image metadata
        """
        with Image.open(image_path) as img:
            return {
                'format': img.format.lower(),
                'mode': img.mode,
                'size': {'width': img.width, 'height': img.height},
                'format_description': img.format_description,
                'info': img.info
            }
    
    def _analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using Google Cloud Vision API.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            RuntimeError: If Vision API call fails
        """
        if not self.vision_client:
            raise RuntimeError("Vision client not initialized")
            
        try:
            # Load image
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Prepare feature requests based on configuration
            features = []
            if self.enable_text_detection:
                features.append(vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION))
            if self.enable_label_detection:
                features.append(vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION))
            if self.enable_object_detection:
                features.append(vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION))
            if self.enable_safe_search:
                features.append(vision.Feature(type_=vision.Feature.Type.SAFE_SEARCH_DETECTION))
            if self.enable_image_properties:
                features.append(vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES))
            
            # Make API request
            response = self.vision_client.annotate_image({
                'image': image,
                'features': features
            })
            
            # Process response
            results = {}
            
            # Text detection
            if self.enable_text_detection and response.text_annotations:
                results['text'] = {
                    'full_text': response.text_annotations[0].description,
                    'text_blocks': [
                        {
                            'text': text.description,
                            'confidence': text.confidence,
                            'bounding_box': [
                                {'x': vertex.x, 'y': vertex.y}
                                for vertex in text.bounding_poly.vertices
                            ]
                        }
                        for text in response.text_annotations[1:]  # Skip first (full text)
                    ]
                }
            
            # Label detection
            if self.enable_label_detection and response.label_annotations:
                results['labels'] = [
                    {
                        'description': label.description,
                        'score': label.score,
                        'topicality': label.topicality
                    }
                    for label in response.label_annotations[:self.max_labels]
                ]
            
            # Object detection
            if self.enable_object_detection and response.localized_object_annotations:
                results['objects'] = [
                    {
                        'name': obj.name,
                        'confidence': obj.score,
                        'bounding_box': {
                            'left': obj.bounding_poly.normalized_vertices[0].x,
                            'top': obj.bounding_poly.normalized_vertices[0].y,
                            'right': obj.bounding_poly.normalized_vertices[2].x,
                            'bottom': obj.bounding_poly.normalized_vertices[2].y
                        }
                    }
                    for obj in response.localized_object_annotations[:self.max_objects]
                ]
            
            # Safe search
            if self.enable_safe_search and response.safe_search_annotation:
                results['safe_search'] = {
                    'adult': response.safe_search_annotation.adult.name,
                    'medical': response.safe_search_annotation.medical.name,
                    'spoof': response.safe_search_annotation.spoof.name,
                    'violence': response.safe_search_annotation.violence.name,
                    'racy': response.safe_search_annotation.racy.name
                }
            
            # Image properties
            if self.enable_image_properties and response.image_properties_annotation:
                colors = response.image_properties_annotation.dominant_colors.colors
                results['colors'] = [
                    {
                        'color': {
                            'red': color.color.red,
                            'green': color.color.green,
                            'blue': color.color.blue
                        },
                        'score': color.score,
                        'pixel_fraction': color.pixel_fraction
                    }
                    for color in colors
                ]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vision API error: {e}")
            raise RuntimeError(f"Vision API analysis failed: {e}")
    
    def process(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process an image file and extract its content and metadata.
        
        Args:
            file_path: Path to the image file
            metadata: Initial metadata dictionary
            
        Returns:
            Dictionary containing:
            - Document metadata
            - Extracted text chunks
            - Token counts
            
        Raises:
            ValueError: If file format is invalid
            IOError: If file cannot be read
        """
        try:
            # Enhance metadata with standard fields
            metadata = self._enhance_metadata(file_path, metadata)
            
            # Validate metadata
            self._validate_metadata(metadata)
            
            # Validate file format
            file_ext = Path(file_path).suffix.lower().lstrip('.')
            if file_ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported image format: {file_ext}")
            
            # Get image metadata
            image_metadata = self._get_image_metadata(file_path)
            
            # Analyze image with Vision API
            vision_results = self._analyze_image(file_path)
            
            # Extract text content for chunking if available
            text_content = vision_results.get('text', {}).get('full_text', '')
            if text_content:
                chunks = self._chunk_text_with_sentences(text_content)
                total_tokens = sum(chunk['token_count'] for chunk in chunks)
            else:
                chunks = []
                total_tokens = 0
            
            # Update metadata with processing results
            metadata.update({
                'image_metadata': image_metadata,
                'vision_results': vision_results,
                'chunks': chunks,
                'chunk_count': len(chunks),
                'total_tokens': total_tokens,
                'processing_status': 'success',
                'processor_type': self.__class__.__name__,
                'processing_completed': datetime.utcnow().isoformat(),
                'vision_settings': {
                    'text_detection': self.enable_text_detection,
                    'label_detection': self.enable_label_detection,
                    'object_detection': self.enable_object_detection,
                    'safe_search': self.enable_safe_search,
                    'image_properties': self.enable_image_properties,
                    'max_labels': self.max_labels,
                    'max_objects': self.max_objects
                }
            })
            
            self.logger.info(
                f"Successfully processed {file_path}: "
                f"format={image_metadata['format']}, "
                f"size={image_metadata['size']}, "
                f"text_chunks={len(chunks)}, tokens={total_tokens}"
            )
            
            return {
                'metadata': metadata,
                'chunks': chunks
            }
            
        except Exception as e:
            self.logger.error(
                f"Error processing image file {file_path}: {str(e)}",
                extra={
                    'component': 'image_processor',
                    'operation': 'process',
                    'file_path': file_path,
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise 