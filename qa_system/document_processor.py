"""
Document processing and embedding generation
"""
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
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
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        logger.debug(f"Initial environment check - Project ID from env: {'Found' if project_id else 'Not found'}")
        logger.debug(f"Initial environment check - Credentials path from env: {'Found' if credentials_path else 'Not found'}")
        
        # If not in environment, try config
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
            
            # Initialize Vertex AI with project and location
            location = "us-central1"  # default location
            vertexai.init(project=project_id, location=location)
            
            # Initialize Gemini
            genai.configure(transport="rest")
            logger.info("Initialized document processor with Gemini")
            
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
        """Initialize API clients."""
        try:
            # Get embedding model configuration
            embedding_config = self.config.get("EMBEDDING_MODEL", {})
            model_name = embedding_config.get("MODEL_NAME", "models/gemini-embedding-exp-03-07")
            
            # Initialize embedding model
            self.embedding_model = model_name
            
            logger.info(f"APIs initialized successfully with embedding model {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize APIs: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to initialize required APIs") from e

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

    async def process_document(self, file_path: str) -> Optional[Dict]:
        """Process a document and prepare it for embedding generation.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document metadata including chunks, directory structure, and extracted metadata
            Returns None if file should be skipped
        """
        file_path = Path(file_path)
        
        # Only check for supported file types
        if file_path.suffix.lower() not in self.supported_types:
            logger.info(f"Skipping unsupported file type: {file_path}")
            return None
            
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
        """Split text into chunks of approximately equal size.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks with specified overlap
            
        Raises:
            ValueError: If chunk_size or chunk_overlap parameters are invalid
        """
        # Validate chunk parameters
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative") 
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
            
        # Clean and normalize input text
        text = text.strip()
        if not text:
            return []
            
        # Split text into sentences
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If this single sentence is larger than chunk_size, split it further
            if sentence_size > self.chunk_size:
                if current_chunk:  # Save current chunk if it exists
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split long sentence into smaller pieces
                words = sentence.split()
                temp_chunk = []
                temp_size = 0
                
                for word in words:
                    if temp_size + len(word) + 1 > self.chunk_size:
                        if temp_chunk:  # Save accumulated words if any
                            chunks.append(" ".join(temp_chunk))
                        temp_chunk = [word]
                        temp_size = len(word)
                    else:
                        temp_chunk.append(word)
                        temp_size += len(word) + 1
                
                if temp_chunk:  # Save any remaining words
                    chunks.append(" ".join(temp_chunk))
                continue
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Keep overlap sentences for next chunk
                overlap_size = 0
                overlap_chunk = []
                
                for prev_sentence in reversed(current_chunk):
                    if overlap_size + len(prev_sentence) > self.chunk_overlap:
                        break
                    overlap_chunk.insert(0, prev_sentence)
                    overlap_size += len(prev_sentence) + 1
                
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1
        
        # Add final chunk if any sentences remain
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        # Validate final chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        # Log chunking results
        if chunks:
            logger.info(f"Split text into {len(chunks)} chunks")
            logger.debug(f"Average chunk size: {sum(len(c) for c in chunks)/len(chunks):.1f} characters")
        else:
            logger.warning("No valid chunks were generated from the input text")
        
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
        """Get embedding from Google's Generative AI embedding model.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValueError: If embedding generation fails or dimensions don't match
        """
        try:
            # Get embedding model configuration
            embedding_config = self.config.get("EMBEDDING_MODEL", {})
            max_length = embedding_config.get("MAX_LENGTH", 8192)
            expected_dim = embedding_config.get("DIMENSIONS", 3072)
            
            # Get rate limit settings
            rate_limits = embedding_config.get("RATE_LIMITS", {})
            max_retries = rate_limits.get("MAX_RETRIES", 5)
            initial_delay = rate_limits.get("INITIAL_RETRY_DELAY", 1)
            max_delay = rate_limits.get("MAX_RETRY_DELAY", 60)
            backoff_factor = rate_limits.get("BACKOFF_FACTOR", 2)
            
            # Truncate text if needed
            if len(text) > max_length:
                logger.warning(f"Text length {len(text)} exceeds max length {max_length}. Truncating.")
                text = text[:max_length]
            
            # Implement retry logic with exponential backoff
            for retry in range(max_retries):
                try:
                    # Call the Google embedding model
                    result = genai.embed_content(
                        model=self.embedding_model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    
                    if not result or 'embedding' not in result:
                        raise ValueError("No embeddings returned from the model")
                    
                    embedding = result['embedding']
                    
                    # Validate embedding dimensions
                    if len(embedding) != expected_dim:
                        raise ValueError(f"Embedding dimension mismatch. Expected {expected_dim}, got {len(embedding)}")
                    
                    # Validate embedding values
                    if not all(isinstance(x, float) for x in embedding):
                        raise ValueError("Embedding contains non-float values")
                    
                    if not all(-1 <= x <= 1 for x in embedding):
                        raise ValueError("Embedding values outside expected range [-1, 1]")
                    
                    return embedding
                    
                except Exception as e:
                    if retry < max_retries - 1:
                        # Calculate exponential backoff delay
                        delay = min(initial_delay * (backoff_factor ** retry), max_delay)
                        logger.warning(f"Retry {retry + 1}/{max_retries} after error: {str(e)}. Waiting {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        raise
            
            raise ValueError(f"Failed to generate embedding after {max_retries} retries")
            
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

    @property
    def supported_types(self) -> List[str]:
        """Get list of supported file extensions."""
        return [
            '.txt', '.pdf', '.docx', '.doc', '.rtf', '.odt', '.md',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.py'
        ]

    async def _should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed based on exclude patterns.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file should be processed, False if it should be excluded
        """
        # Convert path to string for pattern matching
        file_str = str(file_path)
        
        # Check against exclude patterns
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(file_str, pattern):
                logger.debug(f"File {file_path} matches exclude pattern {pattern}")
                return False
                
        return True

    async def process(self, path: Union[str, Path]) -> Dict[str, int]:
        """Process documents from a file or directory path.
        
        Args:
            path: Path to a file or directory to process
            
        Returns:
            Dict containing processing statistics:
                - total_files: Total number of files found
                - processed: Number of files successfully processed
                - failed: Number of files that failed processing
                - skipped: Number of files skipped (excluded or already processed)
        """
        path = Path(path)
        logger.info(f"Starting document processing for path: {path}")
        
        stats = {
            "total_files": 0,
            "processed": 0,
            "failed": 0,
            "skipped": 0
        }
        
        # Get rate limit settings
        rate_limits = self.config.get("EMBEDDING_MODEL", {}).get("RATE_LIMITS", {})
        max_concurrent = rate_limits.get("MAX_CONCURRENT_REQUESTS", 5)
        
        # Create semaphore for concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path: Path) -> bool:
            async with semaphore:
                try:
                    await self.process_document(file_path)
                    return True
                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {str(e)}")
                    return False
        
        if path.is_file():
            # Process single file
            logger.info(f"Processing single file: {path}")
            stats["total_files"] = 1
            try:
                should_process = await self._should_process_file(path)
                if should_process:
                    success = await process_with_semaphore(path)
                    if success:
                        stats["processed"] += 1
                    else:
                        stats["failed"] += 1
                else:
                    logger.info(f"Skipping excluded file: {path}")
                    stats["skipped"] += 1
            except Exception as e:
                logger.error(f"Failed to process file {path}: {str(e)}")
                stats["failed"] += 1
            
        elif path.is_dir():
            # Process directory recursively
            logger.info(f"Processing directory recursively: {path}")
            tasks = []
            
            # First, collect all files and check which ones should be processed
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    stats["total_files"] += 1
                    try:
                        should_process = await self._should_process_file(file_path)
                        if should_process:
                            # Create task for processing the document
                            task = asyncio.create_task(process_with_semaphore(file_path))
                            tasks.append(task)
                        else:
                            logger.info(f"Skipping excluded file: {file_path}")
                            stats["skipped"] += 1
                    except Exception as e:
                        logger.error(f"Failed to check file {file_path}: {str(e)}")
                        stats["failed"] += 1
                        continue
            
            # Process all tasks with progress logging
            total_tasks = len(tasks)
            for i, task in enumerate(asyncio.as_completed(tasks)):
                success = await task
                if success:
                    stats["processed"] += 1
                else:
                    stats["failed"] += 1
                
                if (i + 1) % 10 == 0 or i + 1 == total_tasks:
                    logger.info(f"Processed {i + 1}/{total_tasks} files")
        
        logger.info(f"Document processing complete. Stats: {stats}")
        return stats 