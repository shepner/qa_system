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
import time
from datetime import datetime

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
        
        # Add progress tracking
        self.total_files = 0
        self.processed_files = 0
        self.total_chunks = 0
        self.processed_chunks = 0
        self.start_time = None
        
        # Get exclude patterns
        self.exclude_patterns = doc_config.get("EXCLUDE_PATTERNS", [])
        if isinstance(self.exclude_patterns, str):
            self.exclude_patterns = [self.exclude_patterns]
        
        # Get application settings
        app_config = config.get("APP", {})
        self.base_dir = Path(app_config.get("BASE_DIR", "."))
        
        # Get allowed extensions
        self.allowed_extensions = doc_config.get("ALLOWED_EXTENSIONS", [])
        
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

    def convert_lists_to_strings(self, value: Any) -> Any:
        """Convert any list values in metadata to comma-separated strings.
        
        Args:
            value: The value to convert
            
        Returns:
            Converted value with lists transformed to strings
        """
        try:
            if isinstance(value, list):
                logger.debug(f"Converting list to string: {value}")
                # Handle empty lists
                if not value:
                    return ""
                # Convert all elements to strings and filter out None values
                str_values = [str(v) for v in value if v is not None]
                result = ", ".join(str_values)
                logger.debug(f"Converted list to string: {result}")
                return result
            elif isinstance(value, dict):
                logger.debug(f"Converting dictionary values for keys: {list(value.keys())}")
                converted = {}
                for k, v in value.items():
                    try:
                        converted[k] = self.convert_lists_to_strings(v)
                    except Exception as e:
                        logger.warning(f"Failed to convert value for key '{k}': {str(e)}")
                        converted[k] = str(v)  # Fallback to string conversion
                return converted
            elif value is None:
                return ""
            return value
        except Exception as e:
            logger.error(f"Error converting value {type(value)}: {str(e)}")
            # Fallback to string conversion in case of errors
            try:
                return str(value)
            except Exception as str_e:
                logger.error(f"Failed even string conversion: {str_e}")
                return ""

    async def process_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a document and prepare it for embedding generation.
        
        Args:
            file_path: Path to the document to process (can be string or Path)
            
        Returns:
            Dict containing processed document data including:
            - document_id: Unique identifier for the document
            - content: Processed text content
            - metadata: Document metadata including file info and extracted data
            - chunks: List of text chunks for embedding
            
            Returns None if the file is empty or contains no processable content.
        """
        try:
            # Ensure file_path is a Path object
            if isinstance(file_path, str):
                file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"Document not found: {file_path}")

            if self.should_exclude(file_path):
                logger.info(f"Skipping excluded file: {file_path}")
                return None

            # Generate unique document ID
            document_id = self._generate_doc_id(file_path)
            logger.info(f"Processing document: {file_path} (ID: {document_id})")

            # Extract directory structure metadata first
            metadata = self._extract_directory_metadata(file_path)
            logger.debug("Directory metadata: %s", metadata)

            # Process based on file type
            if file_path.suffix.lower() == '.md':
                # Process markdown file
                extracted = self._extract_markdown(file_path)
                content = extracted["content"]
                # Merge markdown metadata with directory metadata (directory metadata takes precedence)
                metadata.update(extracted.get("metadata", {}))
            else:
                # Process non-markdown file
                content = await self._extract_text(file_path)
                
            # Check if content is empty after processing
            if not content.strip():
                logger.warning(f"File contains no processable content after extraction: {file_path}")
                return None

            # Ensure required metadata fields are present
            metadata.update({
                "document_id": document_id,
                "id": document_id,  # Add id for compatibility
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower()[1:] if file_path.suffix else "unknown",
                "path": str(file_path.resolve())  # Ensure absolute path is always included
            })

            # Log metadata for debugging
            logger.debug(f"Final metadata for {file_path}: {metadata}")

            # Generate chunks
            chunks = self._chunk_text(content)
            if not chunks:
                logger.warning(f"No valid chunks could be generated from file: {file_path}")
                return None
                
            logger.info(f"Generated {len(chunks)} chunks from document")

            return {
                "document_id": document_id,
                "id": document_id,  # Add id for compatibility
                "content": content,
                "metadata": metadata,
                "chunks": chunks
            }

        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {str(e)}")
            raise

    async def generate_embeddings(self, doc_metadata: Dict) -> List[Dict]:
        """Generate embeddings for document chunks.
        
        Args:
            doc_metadata: Document metadata including chunks
            
        Returns:
            List of chunk embeddings with metadata
        """
        embeddings = []
        chunk_start_time = time.time()
        total_chunks = len(doc_metadata["chunks"])
        self.total_chunks += total_chunks
        
        logger.info(f"Starting embedding generation for {total_chunks} chunks from {doc_metadata.get('metadata', {}).get('file_name', 'unknown')}")
        logger.info(f"Total chunks processed so far: {self.processed_chunks}/{self.total_chunks}")
        
        for i, chunk in enumerate(doc_metadata["chunks"], 1):
            chunk_process_start = time.time()
            
            # Generate embedding using Google's Generative AI embedding model
            logger.debug(f"Generating embedding for chunk {i}/{total_chunks}")
            embedding = await self._get_embedding(chunk)
            
            chunk_process_time = time.time() - chunk_process_start
            logger.debug(f"Chunk {i} embedding generated in {chunk_process_time:.2f}s")
            
            # Create embedding dictionary with full metadata
            embedding_dict = {
                "id": f"{doc_metadata['id']}_chunk_{i}",
                "doc_id": doc_metadata["id"],
                "chunk_index": i-1,
                "content": chunk,
                "embedding": embedding,
            }
            
            # Add all document metadata to each chunk
            if "metadata" in doc_metadata:
                embedding_dict.update(doc_metadata["metadata"])
            
            embeddings.append(embedding_dict)
            
            self.processed_chunks += 1
            
            # Log progress every 5 chunks or on completion
            if i % 5 == 0 or i == total_chunks:
                self._log_progress(
                    f"Chunks processed for {doc_metadata.get('metadata', {}).get('file_name', 'unknown')}", 
                    i, 
                    total_chunks,
                    chunk_start_time
                )
                self._log_progress(
                    "Total chunks processed",
                    self.processed_chunks,
                    self.total_chunks,
                    self.start_time
                )
        
        # Log completion for this document
        total_time = time.time() - chunk_start_time
        rate = total_chunks / total_time if total_time > 0 else 0
        logger.info(
            f"Completed embedding generation for {doc_metadata.get('metadata', {}).get('file_name', 'unknown')}: "
            f"{total_chunks} chunks in {total_time:.1f}s ({rate:.1f} chunks/s)"
        )
        
        return embeddings
    
    async def _extract_text(self, file_path: Path) -> str:
        """Extract text content from a file."""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == ".pdf":
                return await self._extract_pdf(file_path)
            elif suffix in [".docx", ".doc"]:
                return await self._extract_docx(file_path)
            elif suffix == ".txt":
                return file_path.read_text()
            elif suffix == ".md":
                # Extract markdown content and return the main text
                markdown_data = self._extract_markdown(file_path)
                return markdown_data["content"]
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
                return await self._extract_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
                
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {str(e)}")
            raise
    
    async def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return " ".join(page.extract_text() for page in reader.pages)
    
    async def _extract_docx(self, file_path: Path) -> str:
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
            logger.info(f"Starting markdown extraction for file: {file_path}")
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
                        logger.debug(f"Extracted YAML frontmatter metadata: {metadata}")
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
            
            logger.debug(f"Extracted hashtags: {hashtags}")

            # Extract links
            links = []
            # Match markdown links [text](url) and bare URLs
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)|(?<![\[\(])(https?://\S+)'
            for match in re.finditer(link_pattern, content):
                if match.group(2):  # Markdown link
                    links.append(match.group(2))
                elif match.group(3):  # Bare URL
                    links.append(match.group(3))
            
            logger.debug(f"Extracted links: {links}")

            # Extract linked document names
            linked_docs = []
            for link in links:
                if isinstance(link, str) and link.endswith('.md'):
                    linked_docs.append(Path(link).stem)
            
            logger.debug(f"Extracted linked documents: {linked_docs}")

            # Build complete metadata dictionary
            result_metadata = metadata.copy()  # Start with frontmatter metadata
            logger.debug(f"Initial metadata from frontmatter: {result_metadata}")
            
            if hashtags:
                result_metadata["hashtags"] = ", ".join(hashtags)  # Convert to string immediately
            if links:
                result_metadata["links"] = ", ".join(links)  # Convert to string immediately
            if linked_docs:
                result_metadata["linked_documents"] = ", ".join(linked_docs)  # Convert to string immediately
            
            logger.debug(f"Metadata after adding extracted fields: {result_metadata}")

            # Convert any remaining list values in frontmatter metadata
            result_metadata = self.convert_lists_to_strings(result_metadata)
            
            # Add required path metadata
            result_metadata["path"] = str(file_path.resolve())
            
            # Log the final metadata state
            logger.info("Final metadata after processing: %s", 
                {k: v for k, v in result_metadata.items() if not any(s in k.lower() for s in ['secret', 'key', 'token'])})

            return {
                "content": content,
                "metadata": result_metadata,
                # Return original lists for potential future processing
                "hashtags": hashtags,
                "links": links
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
        # Pattern to match sentence boundaries and list items
        sentence_pattern = r'([.!?]|\n[-•]|\n\d+\.)\s+'
        sentences = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Handle bullet points and numbered lists
            if line.startswith(('-', '•')) or re.match(r'^\d+\.', line):
                sentences.append(line)
            else:
                # Split normal sentences and keep the delimiter
                parts = re.split(sentence_pattern, line)
                # Reconstruct sentences with their delimiters
                line_sentences = []
                for i in range(0, len(parts)-1, 2):
                    if i+1 < len(parts):
                        sentence = parts[i] + parts[i+1]
                        line_sentences.append(sentence.strip())
                if parts and len(parts) % 2 == 1:  # Handle last part if it exists
                    line_sentences.append(parts[-1].strip())
                sentences.extend([s for s in line_sentences if s])
        
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
    
    async def _extract_image(self, file_path: Path) -> str:
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
        """Check if a file should be excluded based on exclude patterns and allowed extensions.
        
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
            
        # Check if file extension is allowed (config has extensions without the dot)
        file_ext = file_path.suffix.lstrip('.')  # Remove the dot from the extension
        if not file_ext or file_ext.lower() not in (ext.lower() for ext in self.allowed_extensions):
            logger.info(f"Excluding file with unsupported extension: {file_path}")
            return True
            
        # Check each exclude pattern
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(relative_path, pattern):
                logger.info(f"Excluding file matching pattern {pattern}: {file_path}")
                return True
                
        return False

    @property
    def supported_types(self) -> List[str]:
        """Get list of supported file extensions."""
        return [
            '.txt', '.pdf', '.docx', '.doc', '.rtf', '.odt', '.md',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.py'
        ]

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
                if not self.should_exclude(path):
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
                        if not self.should_exclude(file_path):
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

    def _log_progress(self, message: str, current: int, total: int, start_time: float = None):
        """Log progress with percentage and rate information."""
        percentage = (current / total * 100) if total > 0 else 0
        progress_msg = f"{message}: {current}/{total} ({percentage:.1f}%)"
        
        if start_time is not None:
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            eta = (total - current) / rate if rate > 0 else 0
            progress_msg += f" [Rate: {rate:.1f}/s, ETA: {eta:.1f}s]"
        
        logger.info(progress_msg)

    def _extract_directory_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from the directory structure.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing directory-based metadata:
            - path: Full path to the file (required)
            - file_type: File extension without dot (required)
            - relative_path: Path relative to workspace root
            - directory: Full directory hierarchy
            - parent_directory: Immediate parent directory
        """
        try:
            # Get absolute paths
            abs_file_path = file_path.resolve()
            abs_base_dir = self.base_dir.resolve()
            
            # Get relative path from base directory
            try:
                relative_path = str(abs_file_path.relative_to(abs_base_dir))
            except ValueError:
                # If file is not under base_dir, use absolute path
                relative_path = str(abs_file_path)
            
            # Get directory hierarchy
            directory = str(abs_file_path.parent)
            parent_directory = abs_file_path.parent.name
            
            metadata = {
                "path": str(abs_file_path),  # Required field
                "file_type": file_path.suffix.lower()[1:] if file_path.suffix else "unknown",  # Required field
                "relative_path": relative_path,
                "directory": directory,
                "parent_directory": parent_directory,
                "filename_stem": file_path.stem
            }
            
            logger.debug(f"Extracted directory metadata: {metadata}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract directory metadata for {file_path}: {str(e)}")
            # Return minimal metadata on error, but ensure required fields are present
            return {
                "path": str(file_path),  # Required field
                "file_type": file_path.suffix.lower()[1:] if file_path.suffix else "unknown",  # Required field
                "relative_path": str(file_path),
                "directory": str(file_path.parent),
                "parent_directory": file_path.parent.name,
            } 