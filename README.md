# Local File Question-Answering System

A system that uses Google Gemini to provide intelligent responses to questions about local files.

## Features

- Local file processing and indexing
- Vector-based document search using ChromaDB
- Integration with Google Gemini for question answering
- Support for multiple file formats:
  - Documents: PDF, DOCX, TXT, RTF, ODT, MD
  - Images: JPG, JPEG, PNG, GIF, BMP, WEBP (with OCR and object detection)
- Thread-based conversation context
- Image analysis capabilities:
  - Optical Character Recognition (OCR)
  - Object and label detection
  - Image content description

## Requirements

### System Requirements
- Python 3.9+
- 16GB RAM minimum
- SSD storage recommended
- Google Cloud account with Gemini API access
- libmagic (for file type detection)

### System Dependencies Installation

#### libmagic Installation
- macOS: `brew install libmagic`
- Ubuntu/Debian: `sudo apt-get install libmagic1`
- CentOS/RHEL: `sudo yum install file-devel`

### Python Dependencies
The following Python packages are required:
- python-dotenv (1.1.0+) - Environment variable management
- PyYAML (6.0.2+) - YAML configuration parsing
- numpy (2.2.0+) - Numerical computations
- chromadb (1.0.4+) - Vector database
- python-magic (0.4.27+) - File type detection
- fastapi (0.110.0+) - API endpoints and web interface
- pydantic (2.6.4+) - Data validation
- pytest (8.1.1+) - Testing framework
- langchain (0.1.12+) - LLM integration
- google-cloud-vision (3.7.1+) - Image processing
- google-cloud-core (2.4.1+) - Google Cloud integration
- httpx (0.27.0+) - HTTP client
- duckdb (0.10.0+) - Database operations
- typing-extensions (4.10.0+) - Type hinting support

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install system dependencies (see above)

3. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The system can be configured through:
1. YAML configuration file (`config/config.yaml`)
2. Environment variables (with QA_ prefix)
3. Command-line arguments

### Configuration File Structure

```yaml
# Logging Configuration
LOGGING:
  LEVEL: INFO
  LOG_FILE: "logs/qa_system.log"

# Security & API Configuration
SECURITY:
  GOOGLE_APPLICATION_CREDENTIALS: ${GOOGLE_APPLICATION_CREDENTIALS}
  GOOGLE_CLOUD_PROJECT: ${GOOGLE_CLOUD_PROJECT}
  GOOGLE_VISION_API_KEY: ${GOOGLE_VISION_API_KEY}

# Document Processing
FILE_SCANNER:
  DOCUMENT_PATH: "./docs"
  ALLOWED_EXTENSIONS:
    - "txt"
    - "md"
    - "pdf"
    - "doc"
    - "docx"
    - "rtf"
    - "html"
    - "png"
    - "jpg"
    - "jpeg"
    - "gif"
    - "webp"
    - "bmp"
    - "csv"
  EXCLUDE_PATTERNS:
    - ".*"
    - "!README.md"
    - "!ARCHITECTURE.md"
  HASH_ALGORITHM: "sha256"
  SKIP_EXISTING: true

# Vector Database Configuration
VECTOR_STORE:
  TYPE: "chroma"
  PERSIST_DIRECTORY: "./data/vector_store"
  COLLECTION_NAME: "qa_documents"
  DISTANCE_METRIC: "cosine"
  TOP_K: 40
  DIMENSIONS: 768

# Embedding Model Configuration
EMBEDDING_MODEL:
  MODEL_NAME: "models/embedding-001"
  BATCH_SIZE: 15
  MAX_LENGTH: 3072
  DIMENSIONS: 768
```

### Environment Variables

Create a `.env` file in the `secrets` directory:

```ini
# Google Cloud & AI Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=./secrets/qa-system-key.json
GOOGLE_VISION_API_KEY=your-vision-api-key

# Vector Database Configuration
QA_VECTOR_STORE_TYPE=chromadb
QA_VECTOR_STORE_PATH=./data/vectordb

# Security Configuration
QA_SECURITY_AUTH_REQUIRED=true
```

## Usage

### Command-Line Interface

The system provides several command-line operations:

1. Process files or directories:
   ```bash
   python -m qa_system --add /path/to/docs                     # Process entire directory
   python -m qa_system --add /path/to/docs/specific_file.md    # Process single file
   python -m qa_system --add file1.md --add file2.pdf         # Process multiple files
   ```

2. List vector store contents:
   ```bash
   python -m qa_system --list                                  # List all contents
   python -m qa_system --list --filter "*.md"                  # List only markdown files
   ```

3. Remove data from vector store:
   ```bash
   python -m qa_system --remove /path/to/docs/old_file.md      # Remove specific file
   python -m qa_system --remove /path/to/old_docs/             # Remove entire directory
   python -m qa_system --remove --filter "*.pdf"               # Remove all PDF files
   ```

4. Interactive chat mode:
   ```bash
   python -m qa_system --query                                 # Start interactive chat
   python -m qa_system --query "What is the project about?"    # Single query mode
   ```

5. Configuration and debugging:
   ```bash
   python -m qa_system --add /path/to/docs --config custom_config.yaml    # Use custom config
   python -m qa_system --add /path/to/docs --debug                        # Enable debug logging
   ```

### Using the Run Script

For convenience, you can use the provided `run.sh` script:

```bash
./run.sh                                    # Start with default settings
./run.sh --add /path/to/docs               # Add documents
./run.sh --query "What is the project?"    # Run a single query
```

The script will:
1. Create/activate virtual environment
2. Install/upgrade dependencies
3. Create necessary directories
4. Load environment variables
5. Run the program with provided arguments

## Directory Structure

```
.
├── .env                    # Environment configuration (in secrets/)
├── .gitignore             # Git ignore file
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── run.sh                # Setup and run script
├── config/               # Configuration files
│   └── config.yaml      # Main configuration
├── secrets/              # Sensitive configuration and keys
│   ├── .env            # Environment variables
│   └── qa-system-key.json  # Google Cloud service account key
├── data/                # Data storage
│   └── vector_store/   # Vector database files
├── logs/               # Application logs
│   └── qa_system.log  # Main log file
└── qa_system/         # Application source code
```

## Troubleshooting

1. Environment Issues:
   - Verify `.env` file exists in `secrets/` directory
   - Check service account key path is correct
   - Ensure virtual environment is activated

2. API Issues:
   - Verify API keys are valid
   - Check Google Cloud APIs are enabled
   - Review service account permissions

3. Vector Database Issues:
   - Check `VECTOR_DB_PATH` exists and is writable
   - Verify ChromaDB is properly initialized

4. File Processing Issues:
   - Check file permissions
   - Verify file formats are supported
   - Check exclude patterns in configuration

## License

MIT License 