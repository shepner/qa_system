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

The recommended way to install and set up the QA System is to use the provided `run.sh` script, which automates environment preparation and dependency installation.

1. Clone the repository:
```bash
git clone https://github.com/yourusername/qa_system.git
cd qa_system
```

2. Run the setup script from the project folder:
```bash
./run.sh
```

This script will:
- Create required directories (`logs`, `data/vector_store`, `config`, `secrets`)
- Set up and activate a Python virtual environment (`.venv`)
- Install all dependencies from `requirements.txt`
- Check for and copy the example config if needed
- Load environment variables from `secrets/.env` if present

> **Note:** If you need to pass arguments to the main program, you can do so with `./run.sh [arguments]` (e.g., `./run.sh --add path/to/document`).

## Initial Configuration Recommendations

Before using the QA System, it is recommended to analyze your document corpus and generate configuration recommendations. This helps optimize chunking, embedding, and query settings for your specific data.

1. **Run the configuration recommendation tool:**

```bash
python tools/recommend_config.py
```

- This script scans your document directory (as set in `config/config.yaml` under `FILE_SCANNER.DOCUMENT_PATH`) and generates a report at `config/config_recommendations.md`.
- The report provides recommended values for `DOCUMENT_PROCESSING`, `EMBEDDING_MODEL`, and `QUERY` settings based on your corpus statistics and current config strategies.
- Review the generated `config/config_recommendations.md` and update your `config/config.yaml` accordingly for best results.

> **Note:** The script does not modify your config automatically. You must manually review and apply the recommended settings.

## Configuration

The system is configured through:
- `config/config.yaml`: Main configuration file
- Environment variables (prefixed with `QA_`)
- Command-line arguments

Required environment variables:
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud credentials file
- `GOOGLE_CLOUD_PROJECT`: Google Cloud project ID
- `GOOGLE_VISION_API_KEY`: (Optional) Google Vision API key

## Setting Up Secrets (.env)

To use the QA System, you must provide required secrets and API keys in a `secrets/.env` file. This file is used to securely store sensitive environment variables needed for authentication and API access.

1. **Create the secrets directory if it does not exist:**
   ```bash
   mkdir -p secrets
   ```

2. **Create a file named `.env` inside the `secrets/` directory:**
   ```bash
   touch secrets/.env
   ```

3. **Copy and fill in the following template with your actual credentials:**

   ```env
   # Google Cloud service account credentials file path (absolute or relative path to your JSON key)
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json

   # Google Cloud region (e.g., us-central1)
   GOOGLE_CLOUD_REGION=us-central1

   # Google Cloud project ID
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id

   # Google Vision API key
   GOOGLE_VISION_API_KEY=your-google-vision-api-key

   # Gemini API key (for generative AI/embedding)
   # Create the Google gemini API key here: https://aistudio.google.com/app/apikey
   GEMINI_API_KEY=your-gemini-api-key
   ```

> **Note:** Never commit your real `secrets/.env` file to version control. You may provide a `secrets/.env.example` with placeholder values for collaborators.

## Usage

After configuring your system (see above), you can use the following commands:

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

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).
