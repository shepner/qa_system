#!/usr/bin/env python3

"""
@file: wtpsplit_example.py
Example: Sentence and paragraph segmentation with wtpsplit (SaT model)
Reference: https://github.com/segment-any-text/wtpsplit

This script demonstrates:
- Loading the SaT model (sat-12l-sm) from wtpsplit
- Segmenting text into sentences using the model
- Segmenting text into paragraphs and sentences (paragraph segmentation)
- Printing the segmented sentences and paragraphs for a few examples
- Optionally, segmenting paragraphs from a user-provided file

Requirements:
- wtpsplit >= 2.1.5 (pip install wtpsplit)

"""

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from wtpsplit import SaT

# Initialize the SaT model (small, fast multilingual model)
model = SaT("sat-12l-sm")

def run_paragraph_segmentation(text, label="Paragraph Segmentation"):
    print(f"\n{label}")
    print("Input (first 500 chars):", text[:500].replace("\n", "\\n"))
    paragraphs = model.split(
        text,
        do_paragraph_segmentation=True,
    )
    print("Segmented paragraphs and sentences:")
    for p_idx, para in enumerate(paragraphs, 1):
        print(f"  Paragraph {p_idx}:")
        for sent in para:
            print(f"    - {sent}")

if len(sys.argv) > 1:
    # If a file path is provided, use it for paragraph segmentation
    file_path = sys.argv[1]
    print(f"[INFO] Running paragraph segmentation on file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_text = f.read()
        run_paragraph_segmentation(file_text, label=f"Paragraph Segmentation for {file_path}")
    except Exception as e:
        print(f"[ERROR] Could not read file: {e}")
else:
    # Run built-in examples
    texts = [
        "The quick brown fox jumps over the lazy dog. A second sentence follows! And a third? Yes.",
        "Artificial intelligence is transforming the world. It enables new applications in every industry.",
        "Dr. Smith went to Washington. He arrived at 10 a.m. on Jan. 5th, 2024. What a trip!"
    ]

    for i, text in enumerate(texts, 1):
        print(f"\nExample {i}: Sentence Segmentation")
        print("Input:", text)
        sentences = model.split(text)
        print("Segmented sentences:")
        for sent in sentences:
            print(f"- {sent}")

    # Paragraph segmentation example
    paragraph_text = (
        "Paragraph one. It has two sentences.\n\n"
        "Paragraph two starts here. It also has two sentences! Is that enough?\n\n"
        "Final paragraph. Short."
    )
    run_paragraph_segmentation(paragraph_text, label="Example 4: Paragraph Segmentation") 