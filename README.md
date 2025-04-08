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

- Python 3.9+
- 16GB RAM minimum
- SSD storage
- Google Cloud account with Gemini API access

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Run the setup script:
   ```bash
   ./run.sh
   ```
   
   The script will:
   - Create a Python virtual environment
   - Install required dependencies
   - Create a `.env` file from `.env.example` if it doesn't exist
   - Load environment variables

3. Configure the `.env` file with your settings:
   ```ini
   # Google Cloud & AI Configuration
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_APPLICATION_CREDENTIALS=./secrets/qa-system-key.json
   API_KEY=your-gemini-api-key

   # Vector Database Configuration
   VECTOR_DB_PATH=./data/vectordb
   VECTOR_DB_TYPE=chromadb

   # Security Configuration
   AUTH_REQUIRED=true
   ```

## Google Cloud Authentication Setup

### 1. Service Account Creation and Configuration

1. Create service account:
```bash
gcloud iam service-accounts create qa-system-sa --display-name="QA System Service Account"
```

2. Grant required IAM roles:
```bash
gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:qa-system-sa@your-project-id.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:qa-system-sa@your-project-id.iam.gserviceaccount.com" \
    --role="roles/aiplatform.serviceAgent"
gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:qa-system-sa@your-project-id.iam.gserviceaccount.com" \
    --role="roles/vision.user"
```

3. Generate service account key:
```bash
gcloud iam service-accounts keys create secrets/qa-system-key.json \
    --iam-account=qa-system-sa@your-project-id.iam.gserviceaccount.com
```

### 2. API Enablement

Enable required Google Cloud APIs:
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable vision.googleapis.com
```

### 3. Gemini API Setup

1. Get your Gemini API key:
   - Visit https://makersuite.google.com/app/apikey
   - Create a new API key
   - Add the key to your `.env` file as `API_KEY`

### 4. Security Considerations

1. Service Account Key:
   - Store in `./secrets/qa-system-key.json`
   - Add `secrets/` to `.gitignore`
   - Set appropriate file permissions

2. Environment Variables:
   - Never commit `.env` to version control
   - Keep API keys secure
   - Regularly rotate credentials

## Usage

Start the application:
```bash
./run.sh
```

The application will:
1. Load environment variables from `.env`
2. Initialize the vector database
3. Set up API clients for Gemini and Vision AI
4. Start processing documents and handling queries

## Development

### Running Tests
```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run tests
pytest
```

### Directory Structure
```
.
├── .env                    # Environment configuration
├── .gitignore             # Git ignore file
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── run.sh                 # Setup and run script
├── secrets/               # Service account keys and credentials
│   └── qa-system-key.json
├── data/                  # Data storage
│   └── vectordb/         # Vector database files
└── qa_system/            # Application source code
```

## Troubleshooting

1. Environment Issues:
   - Verify `.env` file exists and is properly configured
   - Check service account key path is correct
   - Ensure virtual environment is activated

2. API Issues:
   - Verify API keys are valid
   - Check Google Cloud APIs are enabled
   - Review service account permissions

3. Vector Database Issues:
   - Check `VECTOR_DB_PATH` exists and is writable
   - Verify ChromaDB is properly initialized

## License

MIT License 