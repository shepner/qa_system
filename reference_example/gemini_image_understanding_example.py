"""
@file: gemini_image_understanding_example.py
Example: Gemini API image understanding (captioning, object detection, segmentation)
Reference: https://ai.google.dev/gemini-api/docs/image-understanding

This script demonstrates:
- Uploading an image file using the Gemini Files API
- Generating a caption for the image
- Detecting objects and returning bounding boxes
- Segmenting objects and returning masks

Requirements:
- Place a sample image named 'sample.jpg' in the same directory as this script, or supply an image path as a CLI argument
- Set the GEMINI_API_KEY environment variable (see .env)
"""

import os
import sys
from dotenv import load_dotenv
from google import genai
import json

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment.")

client = genai.Client(api_key=GEMINI_API_KEY)

# Accept image path as a command-line argument, default to 'sample.jpg' in this directory
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "sample.jpg")

print(f"Using image: {IMAGE_PATH}")
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# 1. Upload the image file using the Files API
print("Uploading image file...")
my_file = client.files.upload(file=IMAGE_PATH)
print(f"Uploaded file URI: {my_file.uri}")

# 2. Image Captioning
print("\n--- Image Captioning ---")
caption_response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[my_file, "Caption this image."]
)
print("Caption:", caption_response.text)

# 3. Object Detection (bounding boxes)
print("\n--- Object Detection ---")
object_detection_prompt = (
    "Detect all of the prominent items in the image. "
    "The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000. "
    "Output a JSON list of objects with their labels and bounding boxes."
)
object_response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[my_file, object_detection_prompt]
)
try:
    object_json = json.loads(object_response.text)
    print(json.dumps(object_json, indent=2))
except Exception:
    print("Raw response:", object_response.text)

"""
# 4. Image Segmentation (masks)
print("\n--- Image Segmentation ---")
segmentation_prompt = (
    "Give the segmentation masks for the prominent items. "
    "Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d', "
    "the segmentation mask in key 'mask', and the text label in the key 'label'. Use descriptive labels."
)
segmentation_response = client.models.generate_content(
    model="gemini-2.5-pro-latest",  # Segmentation is supported in 2.5 models
    contents=[my_file, segmentation_prompt]
)
try:
    segmentation_json = json.loads(segmentation_response.text)
    print(json.dumps(segmentation_json, indent=2))
except Exception:
    print("Raw response:", segmentation_response.text)
"""