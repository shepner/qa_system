# Example: Using Gemini API for text generation
# Reference: https://ai.google.dev/gemini-api/docs/text-generation

from google import genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-1.5-flash-latest",
    contents="Write a story about a magic backpack."
)

print(response.text) 