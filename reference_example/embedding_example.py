# Example: Using Gemini API to generate embeddings
# Reference: https://ai.google.dev/gemini-api/docs/embeddings

from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

# Replace with your actual Gemini API key
# https://aistudio.google.com/app/apikey
#
# GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# https://ai.google.dev/gemini-api/docs/embeddings
result = client.models.embed_content(
    model="gemini-embedding-exp-03-07",
    contents="What is the meaning of life?",
    config=types.EmbedContentConfig(task_type="QUESTION_ANSWERING")
)

print(result.embeddings) 