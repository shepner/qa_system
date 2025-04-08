"""
Document processing and embedding generation
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import PyPDF2
import docx
from tika import parser
import numpy as np
import google.generativeai as genai
from google.cloud import vision
from google.cloud.vision_v1 import ImageAnnotatorClient
from google.auth import credentials
import fnmatch
from vertexai import generative_models
import vertexai

# Get logger for this module
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and embedding generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the document processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._setup_logging()
        self._initialize_apis()
        
        # Get document processing settings from new config structure
        doc_config = config.get("DOCUMENT_PROCESSING", {})
        self.chunk_size = doc_config.get("MAX_CHUNK_SIZE", 1000)
        self.chunk_overlap = doc_config.get("CHUNK_OVERLAP", 200)
        self.concurrent_tasks = doc_config.get("CONCURRENT_TASKS", 4)
        self.batch_size = doc_config.get("BATCH_SIZE", 10)
        
        # Get exclude patterns
        self.exclude_patterns = doc_config.get("EXCLUDE_PATTERNS", [])
        if isinstance(self.exclude_patterns, str):
            self.exclude_patterns = [self.exclude_patterns]
        
        # Get application settings
        app_config = config.get("APP", {})
        self.base_dir = Path(app_config.get("BASE_DIR", "."))
        
        logger.info("Starting API configuration and project ID resolution")
        
        # First check environment variables directly
        api_key = os.getenv("API_KEY")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        logger.debug(f"Initial environment check - Project ID from env: {'Found' if project_id else 'Not found'}")
        logger.debug(f"Initial environment check - Credentials path from env: {'Found' if credentials_path else 'Not found'}")
        
        # If not in environment, try config
        if not api_key:
            security_config = config.get("SECURITY", {})
            api_key = (
                security_config.get("API_KEY") or 
                config.get("SECURITY_API_KEY")
            )
            logger.debug("API key not found in environment, attempting to load from config")
        
        if not project_id:
            security_config = config.get("SECURITY", {})
            project_id = (
                security_config.get("GOOGLE_CLOUD_PROJECT") or 
                config.get("SECURITY_GOOGLE_CLOUD_PROJECT")
            )
            logger.debug("Project ID not found in environment, attempting to load from config")
        
        if not credentials_path:
            security_config = config.get("SECURITY", {})
            credentials_path = (
                security_config.get("GOOGLE_APPLICATION_CREDENTIALS") or 
                config.get("SECURITY_GOOGLE_APPLICATION_CREDENTIALS")
            )
            logger.debug("Credentials path not found in environment, attempting to load from config")
        
        # Log the final resolution status of required configurations
        if not api_key:
            logger.error("Google API key not found in environment or config")
            raise ValueError("Google API key not found in environment or config")
        if not project_id:
            logger.error("Google Cloud project ID not found in environment or config")
            raise ValueError("Google Cloud project ID not found in environment or config")
        if not credentials_path:
            logger.error("Google Cloud credentials path not found in environment or config")
            raise ValueError("Google Cloud credentials path not found in environment or config")
        
        try:
            # Resolve credentials path to absolute path
            credentials_path = str(Path(credentials_path).resolve())
            logger.info(f"Resolved credentials path to: {credentials_path}")
            
            # Set environment variables for Google Cloud client libraries
            previous_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
            
            if previous_project_id and previous_project_id != project_id:
                logger.warning(f"Project ID mismatch detected - Previous: {previous_project_id}, New: {project_id}")
            
            # Initialize Vision client
            logger.info("Initializing Vision API client...")
            self.vision_client = ImageAnnotatorClient()
            
            # Verify the project ID matches what we expect
            client_project_id = self.vision_client._client_options.quota_project_id or project_id
            if client_project_id != project_id:
                logger.warning(f"Vision API client project ID mismatch - Expected: {project_id}, Got: {client_project_id}")
            else:
                logger.info(f"Successfully initialized Vision API client with project ID: {project_id}")
            
            logger.info(f"Using credentials from: {credentials_path}")
            
            # Configure Gemini API
            try:
                logger.info("Configuring Generative AI API...")
                genai.configure(api_key=api_key)
                self.embedding_model = "models/embedding-001"
                logger.info("Successfully configured Generative AI API")
            except Exception as e:
                logger.error(f"Failed to initialize Generative AI API: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Failed to initialize Google AI clients: {str(e)}")
            raise

    def _setup_logging(self):
        """Configure detailed logging for API and project operations."""
        logger.info("Initializing DocumentProcessor with configuration")
        logger.debug("Configuration details: %s", {
            k: v for k, v in self.config.items() 
            if not any(sensitive in k.lower() for sensitive in ['key', 'secret', 'password'])
        })

    def _initialize_apis(self):
        """Initialize Google Cloud APIs with enhanced logging."""
        try:
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            if not project_id:
                logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
                raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
            
            logger.info("Initializing APIs with project ID: %s", project_id)
            
            # Initialize Vertex AI
            try:
                logger.debug("Initializing Vertex AI...")
                vertexai.init(project=project_id)
                self.generation_model = generative_models.GenerativeModel("gemini-pro")
                logger.info("Successfully initialized Vertex AI generation model")
            except Exception as e:
                logger.error("Failed to initialize Vertex AI: %s", str(e))
                raise

            # Initialize Vision AI
            try:
                logger.debug("Initializing Vision AI client...")
                self.vision_client = vision.ImageAnnotatorClient()
                logger.info("Successfully initialized Vision AI client")
            except Exception as e:
                logger.error("Failed to initialize Vision AI client: %s", str(e))
                raise

            logger.info("All APIs successfully initialized")
            
        except Exception as e:
            logger.critical("Critical error during API initialization: %s", str(e))
            raise

    def _needs_reprocessing(self, file_path: Path, existing_doc_id: Optional[str] = None, was_successful: Optional[bool] = None) -> bool:
        """Check if a document needs to be reprocessed based on modification time and previous processing status.
        
        Args:
            file_path: Path to the document
            existing_doc_id: Optional ID of existing document
            was_successful: Optional flag indicating if previous processing was successful
            
        Returns:
            True if document needs reprocessing
        """
        # Always process if no existing doc
        if not existing_doc_id:
            logger.info(f"Document {file_path} not found in vector store, will process")
            return True
            
        # Always reprocess if previous attempt failed
        if was_successful is False:
            logger.info(f"Document {file_path} had failed processing previously, will reprocess")
            return True
            
        # Check if file has been modified since last processing
        new_id = self._generate_doc_id(file_path)
        needs_reprocessing = new_id != existing_doc_id
        
        if needs_reprocessing:
            logger.info(f"Document {file_path} has been modified since last processing, will reprocess")
        else:
            logger.info(f"Document {file_path} is unchanged and was successfully processed, skipping")
            
        return needs_reprocessing

    async def process_document(self, file_path: str) -> Dict:
        """Process a document and prepare it for embedding generation.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document metadata including chunks, directory structure, and extracted metadata
        """
        file_path = Path(file_path)
        
        # Only check for supported file types
        if file_path.suffix.lower() not in self.supported_types:
            logger.info(f"Skipping unsupported file type: {file_path}")
            raise ValueError(f"Unsupported file type: {file_path}")
            
        logger.info(f"Processing document: {file_path}")
        
        # Extract directory structure information
        try:
            relative_path = file_path.resolve().relative_to(self.base_dir.resolve())
            dir_parts = list(relative_path.parent.parts)
            
            # Create directory hierarchy metadata
            dir_metadata = {
                "full_path": str(file_path.parent),
                "relative_path": str(relative_path.parent),
                "hierarchy": dir_parts,
                "depth": len(dir_parts),
                "parent_dir": str(file_path.parent.name),
                "root_dir": dir_parts[0] if dir_parts else None
            }
            
            logger.info("Extracting text content...")
            if file_path.suffix.lower() == '.md':
                extracted = self._extract_markdown(file_path)
                content = extracted["content"]
                metadata = {
                    "yaml_metadata": extracted["metadata"],
                    "hashtags": extracted["hashtags"],
                    "links": extracted["links"],
                    "directory": dir_metadata  # Add directory structure metadata
                }
            else:
                content = self._extract_text(file_path)
                metadata = {
                    "directory": dir_metadata  # Add directory structure metadata
                }

            if not content.strip():
                raise ValueError(f"No text content extracted from {file_path}")
                
            logger.info("Chunking text...")
            chunks = self._chunk_text(content)
            logger.info(f"Created {len(chunks)} chunks")
            
            doc_id = self._generate_doc_id(file_path)
            logger.info(f"Generated document ID: {doc_id}")
            
            return {
                "id": doc_id,
                "path": str(file_path),
                "filename": file_path.name,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "file_type": file_path.suffix.lower()[1:],
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            raise
    
    async def generate_embeddings(self, doc_metadata: Dict) -> List[Dict]:
        """Generate embeddings for document chunks.
        
        Args:
            doc_metadata: Document metadata including chunks
            
        Returns:
            List of chunk embeddings with metadata
        """
        embeddings = []
        total_chunks = len(doc_metadata["chunks"])
        logger.info(f"Generating embeddings for {total_chunks} chunks from {doc_metadata['filename']}")
        
        for i, chunk in enumerate(doc_metadata["chunks"]):
            # Generate embedding using Google's Generative AI embedding model
            logger.debug(f"Generating embedding for chunk {i+1}/{total_chunks}")
            embedding = await self._get_embedding(chunk)
            
            embeddings.append({
                "id": f"{doc_metadata['id']}_chunk_{i}",
                "doc_id": doc_metadata["id"],
                "chunk_index": i,
                "content": chunk,
                "embedding": embedding,
            })
            
            if (i + 1) % 10 == 0 or i + 1 == total_chunks:
                logger.info(f"Progress: {i+1}/{total_chunks} chunks processed ({((i+1)/total_chunks)*100:.1f}%)")
            
        return embeddings
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text content from a file."""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == ".pdf":
                return self._extract_pdf(file_path)
            elif suffix in [".docx", ".doc"]:
                return self._extract_docx(file_path)
            elif suffix == ".txt":
                return file_path.read_text()
            elif suffix in [".rtf", ".odt"]:
                # Use Apache Tika for other supported formats
                logger.info(f"Using Tika to extract text from {suffix} file")
                parsed = parser.from_file(str(file_path))
                if not parsed.get("content"):
                    raise ValueError(f"Tika failed to extract content from {file_path}")
                return parsed["content"]
            elif suffix == ".py":
                return file_path.read_text()
            elif suffix in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
                return self._extract_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
                
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {str(e)}")
            raise
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return " ".join(page.extract_text() for page in reader.pages)
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        doc = docx.Document(file_path)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    
    def _extract_markdown(self, file_path: Path) -> Dict[str, Any]:
        """Extract text and metadata from Markdown file.
        
        This method reads the markdown file and processes:
        - YAML frontmatter for metadata
        - Content body with preserved formatting
        - Hashtags (#tag) for keywords
        - Links (both [text](url) and bare URLs)
        
        Returns:
            Dict containing:
            - content: The main markdown content
            - metadata: Dict of YAML frontmatter fields
            - hashtags: List of extracted hashtags
            - links: List of extracted links
        """
        try:
            content = file_path.read_text()
            
            # Parse YAML frontmatter if present
            metadata = {}
            if content.startswith('---'):
                try:
                    import yaml
                    # Find the second '---' that closes the frontmatter
                    end_idx = content.find('---', 3)
                    if end_idx != -1:
                        frontmatter = content[3:end_idx].strip()
                        metadata = yaml.safe_load(frontmatter)
                        # Remove frontmatter from content
                        content = content[end_idx + 3:].strip()
                except ImportError:
                    logger.warning("yaml package not installed. Skipping frontmatter parsing.")
                except Exception as e:
                    logger.warning(f"Failed to parse YAML frontmatter: {str(e)}")

            # Extract hashtags (excluding those in code blocks)
            import re
            hashtags = []
            # Match hashtags that aren't inside code blocks
            code_block = False
            for line in content.split('\n'):
                if line.startswith('```'):
                    code_block = not code_block
                elif not code_block:
                    # Find hashtags that aren't part of URLs
                    tags = re.findall(r'(?<![\w/])\#([\w-]+)', line)
                    hashtags.extend(tags)

            # Extract links
            links = []
            # Match markdown links [text](url) and bare URLs
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)|(?<![\[\(])(https?://\S+)'
            for match in re.finditer(link_pattern, content):
                if match.group(2):  # Markdown link
                    links.append(match.group(2))
                elif match.group(3):  # Bare URL
                    links.append(match.group(3))

            return {
                "content": content,
                "metadata": metadata,
                "hashtags": list(set(hashtags)),  # Remove duplicates
                "links": list(set(links))  # Remove duplicates
            }
            
        except Exception as e:
            logger.error(f"Failed to process markdown file {file_path}: {str(e)}")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of approximately equal size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # Add 1 for space
            if current_size + word_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate a unique document ID based on file content.
        
        Args:
            file_path: Path to the document
            
        Returns:
            A 16-character hexadecimal hash of the file content
        """
        # Read file in binary mode to handle all file types
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files efficiently
            sha256_hash = hashlib.sha256()
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Google's Generative AI embedding model."""
        try:
            # Call the Google embedding model
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            
            if not result or 'embedding' not in result:
                raise ValueError("No embeddings returned from the model")
            
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def _extract_image(self, file_path: Path) -> str:
        """Extract text and generate description from image using Google Vision AI.
        
        This method:
        1. Performs OCR to extract any text in the image
        2. Detects objects and labels in the image
        3. Generates a natural language description
        4. Combines all information into a structured text representation
        """
        try:
            # Read the image file
            with open(file_path, "rb") as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Perform OCR
            text_detection = self.vision_client.text_detection(image=image)
            extracted_text = text_detection.text_annotations[0].description if text_detection.text_annotations else ""
            
            # Detect labels
            label_detection = self.vision_client.label_detection(image=image)
            labels = [label.description for label in label_detection.label_annotations]
            
            # Detect objects
            object_detection = self.vision_client.object_localization(image=image)
            objects = [obj.name for obj in object_detection.localized_object_annotations]
            
            # Generate a structured description
            description_parts = []
            
            if extracted_text:
                description_parts.append(f"Text found in image: {extracted_text}")
            
            if labels:
                description_parts.append(f"Image contains: {', '.join(labels)}")
            
            if objects:
                description_parts.append(f"Objects detected: {', '.join(objects)}")
            
            # Combine all information
            full_description = "\n\n".join(description_parts)
            
            if not full_description:
                return "No text or recognizable content found in image."
            
            return full_description
            
        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {str(e)}")
            raise
    
    def should_exclude(self, file_path: Path) -> bool:
        """Check if a file should be excluded based on exclude patterns.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file should be excluded
        """
        # Get the relative path for pattern matching
        try:
            relative_path = str(file_path.resolve().relative_to(self.base_dir.resolve()))
        except ValueError:
            # If file is not under base_dir, use absolute path
            relative_path = str(file_path)
            
        # Check each exclude pattern
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(relative_path, pattern):
                logger.info(f"Excluding file {file_path} - matches pattern '{pattern}'")
                return True
                
        return False 