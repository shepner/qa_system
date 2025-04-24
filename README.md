# QA System

A powerful document processing and question-answering system that uses Google's Gemini model for embeddings and semantic search capabilities.

## Features

- File management operations (add, remove, list, query)
- Support for multiple file formats (PDF, Text, Markdown, CSV, Images)
- Local file processing and embedding generation
- Vector database storage using ChromaDB
- Semantic search capabilities
- Efficient batch processing
- Comprehensive metadata tracking

## Requirements

- Python 3.13 or higher
- Google Cloud account with Gemini API access
- Sufficient storage for embeddings and metadata
- Optional: GPU support for improved performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/qa_system.git
cd qa_system
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
   - Copy `config/config.yaml.example` to `config/config.yaml`
   - Update configuration values as needed
   - Set required environment variables in `.env` file

## Configuration

The system is configured through:
- `config/config.yaml`: Main configuration file
- Environment variables (prefixed with `QA_`)
- Command-line arguments

Required environment variables:
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud credentials file
- `GOOGLE_CLOUD_PROJECT`: Google Cloud project ID
- `GOOGLE_VISION_API_KEY`: (Optional) Google Vision API key

## Usage

### Adding Documents

```bash
python -m qa_system --add path/to/document
python -m qa_system --add path/to/directory  # Process entire directory
```

### Listing Documents

```bash
python -m qa_system --list
python -m qa_system --list --filter "*.pdf"  # Filter by pattern
```

### Removing Documents

```bash
python -m qa_system --remove path/to/document
python -m qa_system --remove "*.pdf"  # Remove by pattern
```

### Querying

```bash
python -m qa_system --query "Your question here"
python -m qa_system --query  # Enter interactive chat mode
```

## Architecture

The system follows a modular architecture with clear separation of concerns:

1. **Common Components**
   - Main Module: Command-line interface
   - Configuration Module: System settings
   - Logging Setup: Centralized logging
   - Vector Database: ChromaDB integration

2. **Document Processing**
   - File Scanner: File discovery and validation
   - Document Processors: Format-specific processing
   - Embedding Generator: Gemini model integration
   - Metadata Management: Document tracking

3. **Vector Operations**
   - Storage: ChromaDB operations
   - Querying: Semantic search
   - Collection Management: Database lifecycle

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Development

### Running Tests

```bash
pytest
pytest --cov=qa_system  # With coverage
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run all checks:
```bash
black .
isort .
flake8
mypy .
```

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and style checks
5. Submit a pull request

## Support

[Your Support Information] 