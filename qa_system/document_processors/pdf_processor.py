"""
@file: pdf_processor.py
PDFDocumentProcessor: Processes PDF files for document ingestion in the QA system.

This module defines PDFDocumentProcessor, a subclass of BaseDocumentProcessor, for handling PDF (.pdf) files. It extracts metadata, text per page, detects section headers, chunks text, and returns structured results for downstream processing. It also extracts URLs, supports section hierarchy, and provides chunk-level metadata including context and summaries.

Major Components:
- PDFDocumentProcessor: Main processor class for .pdf files
- _list_to_csv: Utility for serializing lists as CSV
- _detect_section_headers: Utility for detecting section headers in text
- process: Main entry point for processing a PDF file

Dependencies:
- pypdf for PDF parsing
- csv, io, re for parsing and serialization

Usage Example:
    processor = PDFDocumentProcessor(config)
    result = processor.process("example.pdf")
    print(result['metadata'])
    print(result['chunks'][0]['text'])

Limitations:
- Section header detection is heuristic and may not match all use cases
- Topics extraction is a placeholder
- Summarization is basic (first sentence or first 20 words)

Version History:
- 1.0: Initial version
"""

from .base_processor import BaseDocumentProcessor
import re
import csv
import io
import os

def debug_log(message):
    with open("pdf_debug.log", "a") as debug_file:
        debug_file.write(message + "\n")

def clean_pdf_text(text):
    """
    Cleans up PDF-extracted text by:
    - Removing hyphenation at line breaks
    - Joining lines that are split mid-sentence
    - Collapsing multiple spaces
    - Preserving paragraph breaks
    """
    # Remove hyphenation at line breaks (e.g., "infor-\nmation" -> "information")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Join lines that are not paragraph breaks (single newlines not preceded/followed by another newline)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    return text

SECTION_HEADER_REGEX = re.compile(r'^(\d+(?:\.\d+)*)(?:\s+|\.)+(.+)')
BULLET_CHARS = {'■', '-', '*', '•', 'I', '—', '–', '·', '●'}
BULLET_REGEX = re.compile(r'^(?:[■\-*•I—–·●]|\d+\.|[a-zA-Z]\))\s+')

def is_section_header(text, size, bold, header1_size):
    # Heuristic: section headers are often short, large, and/or bold, or match section number pattern
    if SECTION_HEADER_REGEX.match(text):
        return True
    if size >= header1_size or bold:
        # Also check if the line is not too long
        return len(text) < 80
    return False

class PDFDocumentProcessor(BaseDocumentProcessor):
    """
    Processor for PDF (.pdf) files using PyMuPDF (fitz).

    Converts PDF to Markdown, saving images to output_dir and referencing them in Markdown.
    Returns a dictionary with 'chunks', 'metadata', and 'page_texts'.
    """
    @staticmethod
    def pdf_to_markdown(pdf_path: str, output_dir: str = './tmp') -> tuple[str, list[str]]:
        """
        Convert a PDF file to Markdown, preserving formatting and images.
        Args:
            pdf_path (str): Path to the PDF file.
            output_dir (str): Directory to save images. Defaults to './tmp'.
        Returns:
            tuple: (Markdown string, list of page texts)
        """
        import fitz  # PyMuPDF
        BULLET_CHARS = {'■', '-', '*', '•'}
        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        markdown_lines = []
        image_count = 0
        font_sizes = []
        page_texts = []
        for page in doc:
            page_texts.append(page.get_text() or '')
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block['type'] == 0:
                    for line in block['lines']:
                        for span in line['spans']:
                            font_sizes.append(span['size'])
        if font_sizes:
            unique_sizes = sorted(set(font_sizes), reverse=True)
            header1_size = unique_sizes[0]
            header2_size = unique_sizes[1] if len(unique_sizes) > 1 else header1_size
        else:
            header1_size = header2_size = 12
        # For robust list nesting, track unique x0 values in order of appearance
        x0_to_level = []  # List of unique x0 values, in order encountered
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block['type'] == 0:  # text
                    for line in block['lines']:
                        # --- Section header merging logic ---
                        merged_parts = []
                        for i, span in enumerate(line['spans']):
                            text = span['text'].strip()
                            if not text:
                                continue
                            if i > 0:
                                prev = line['spans'][i-1]['text'].strip()
                                if (prev.replace('.', '').isdigit() and text.replace('.', '').isdigit()) or (prev.endswith('.') and text.isdigit()):
                                    merged_parts[-1] += text
                                    continue
                            merged_parts.append(text)
                        merged_text = ' '.join(merged_parts)
                        if not merged_text:
                            continue
                        # Debug: log merged_text
                        debug_log(f"[DEBUG] merged_text: '{merged_text}'")
                        # Fix for section numbers split by space (repeat until stable)
                        prev_text = None
                        while prev_text != merged_text:
                            prev_text = merged_text
                            merged_text = re.sub(r'(\d+(?:\.\d+)*)\s+(\d+)', r'\1.\2', merged_text)
                        debug_log(f"[DEBUG] after section fix: '{merged_text}'")
                        # Section header detection
                        section_match = SECTION_HEADER_REGEX.match(merged_text)
                        if section_match:
                            section_num = section_match.group(1)
                            # Always use header level 4 for section headers
                            header_level = 4
                            # Font size and x0 clues (still log for debug)
                            if line['spans']:
                                font_size = line['spans'][0]['size']
                                x0 = line['spans'][0]['origin'][0] if 'origin' in line['spans'][0] else line['spans'][0].get('x', 0)
                            else:
                                font_size = None
                                x0 = None
                            debug_log(f"[DEBUG] Section header detected: '{merged_text}' | section_num: '{section_num}' | font_size: {font_size} | x0: {x0} | header_level: {header_level}")
                            # Output the header with extra newlines before and after
                            markdown_lines.append(f"\n{'#' * header_level} {merged_text}\n")
                            continue
                        # --- Nested list detection with x0 mapping ---
                        if line['spans']:
                            x0 = line['spans'][0]['origin'][0] if 'origin' in line['spans'][0] else line['spans'][0].get('x', 0)
                        else:
                            x0 = 0
                        if BULLET_REGEX.match(merged_text):
                            debug_log(f"[DEBUG] List item detected: x0={x0}, text='{merged_text}'")
                            # Remove bullet and leading space
                            item_text = BULLET_REGEX.sub('', merged_text, count=1)
                            # Map x0 to indent level (order of appearance)
                            found = False
                            for idx, val in enumerate(x0_to_level):
                                if abs(x0 - val) < 1.0:  # Allow for small floating point differences
                                    indent_level = idx
                                    found = True
                                    break
                            if not found:
                                x0_to_level.append(x0)
                                indent_level = len(x0_to_level) - 1
                            markdown_lines.append(f"{'  ' * indent_level}- {item_text.strip()}")
                            continue
                        # Otherwise, process as before (span-level formatting)
                        for span in line['spans']:
                            text = span['text'].strip()
                            if not text:
                                continue
                            size = span['size']
                            bold = span.get('flags', 0) & 2
                            italic = span.get('flags', 0) & 1
                            if size >= header1_size:
                                markdown_lines.append(f"# {text}")
                            elif size >= header2_size:
                                markdown_lines.append(f"## {text}")
                            elif text and text[0] in BULLET_CHARS:
                                markdown_lines.append(f"- {text[1:].strip()}")
                            elif bold and italic:
                                markdown_lines.append(f"***{text}***")
                            elif bold:
                                markdown_lines.append(f"**{text}**")
                            elif italic:
                                markdown_lines.append(f"*{text}*")
                            else:
                                markdown_lines.append(text)
                elif block['type'] == 1:  # image
                    img_xref = block.get('image')
                    if isinstance(img_xref, int):
                        try:
                            img = doc.extract_image(img_xref)
                            img_bytes = img['image']
                            ext = img.get('ext', 'png')
                            img_name = f"image_{page_num+1}_{image_count+1}.{ext}"
                            img_path = os.path.join(output_dir, img_name)
                            with open(img_path, 'wb') as f:
                                f.write(img_bytes)
                            markdown_lines.append(f"![Image]({img_path})")
                            image_count += 1
                        except Exception as e:
                            markdown_lines.append(f"<!-- Failed to extract image: {e} -->")
        # Debug: log the final markdown_lines before joining
        debug_log('[DEBUG] FINAL MARKDOWN_LINES:')
        for idx, line in enumerate(markdown_lines):
            debug_log(f'[DEBUG] LINE {idx}: {repr(line)}')
        # Output exactly as detected, no further reformatting
        markdown = '\n'.join(markdown_lines)
        return markdown, page_texts

    def process(self, file_path, metadata=None, output_dir: str = './tmp'):
        """
        Process a PDF file, converting it to Markdown and extracting images.
        Args:
            file_path (str): Path to the PDF file
            metadata (dict, optional): Additional metadata to include
            output_dir (str): Directory to save images (default: './tmp')
        Returns:
            dict: {
                'chunks': List of chunk dicts with 'text' (Markdown) and 'metadata',
                'metadata': Document-level metadata,
                'page_texts': List of text per page (for reference)
            }
        """
        self.logger.debug(f"Processing PDF file to Markdown: {file_path}")
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        else:
            extracted = self.extract_metadata(file_path)
            metadata = {**extracted, **metadata}
        try:
            markdown, page_texts = self.pdf_to_markdown(file_path, output_dir)
        except Exception as e:
            self.logger.warning(f"Failed to convert PDF to Markdown: {e}")
            metadata['skipped'] = True
            metadata['skip_reason'] = f'conversion-error: {e}'
            return {
                'metadata': metadata,
                'chunks': [],
                'page_texts': []
            }
        
        # --- Header-based chunking ---
        import re
        header_pattern = re.compile(r"^#### (.+)$", re.MULTILINE)
        lines = markdown.splitlines()
        header_indices = []
        headers = []
        for idx, line in enumerate(lines):
            m = header_pattern.match(line.strip())
            if m:
                header_indices.append(idx)
                headers.append(m.group(1).strip())
        # Add a sentinel for the end
        header_indices.append(len(lines))
        chunks = []
        chunk_index = 0
        for i in range(len(header_indices) - 1):
            start = header_indices[i]
            end = header_indices[i+1]
            block_lines = lines[start:end]
            block_text = '\n'.join(block_lines).strip()
            if not block_text:
                continue
            # If block is too large, split further
            if len(block_text) > self.chunk_size:
                # Try to split evenly by lines
                n_splits = (len(block_text) // self.chunk_size) + 1
                split_size = max(1, len(block_lines) // n_splits)
                for j in range(0, len(block_lines), split_size):
                    chunk_lines = block_lines[j:j+split_size]
                    chunk_text = '\n'.join(chunk_lines).strip()
                    if not chunk_text:
                        continue
                    summary = chunk_text.split(". ")[0]
                    if len(summary.split()) < 5:
                        summary = " ".join(chunk_text.split()[:20])
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            **metadata,
                            'chunk_index': chunk_index,
                            'section_header': headers[i],
                            'summary': summary.strip()
                        }
                    })
                    chunk_index += 1
            else:
                summary = block_text.split(". ")[0]
                if len(summary.split()) < 5:
                    summary = " ".join(block_text.split()[:20])
                chunks.append({
                    'text': block_text,
                    'metadata': {
                        **metadata,
                        'chunk_index': chunk_index,
                        'section_header': headers[i],
                        'summary': summary.strip()
                    }
                })
                chunk_index += 1
        document_metadata = dict(metadata)
        document_metadata['chunk_count'] = len(chunks)
        document_metadata['total_tokens'] = sum(len(chunk['text']) for chunk in chunks)
        document_metadata['page_count'] = len(page_texts)
        return {
            'chunks': chunks,
            'metadata': document_metadata,
            'page_texts': page_texts
        } 