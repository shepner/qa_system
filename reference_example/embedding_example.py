# Example: Using Gemini API to generate embeddings
# Reference: https://ai.google.dev/gemini-api/docs/embeddings

from google import genai

# Replace with your actual Gemini API key
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

client = genai.Client(api_key=GEMINI_API_KEY)

result = client.models.embed_content(
    model="gemini-embedding-exp-03-07",
    contents="What is the meaning of life?"
)

print(result.embeddings) 