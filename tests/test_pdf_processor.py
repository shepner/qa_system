import pytest
from pathlib import Path
import sys
from unittest.mock import patch

try:
    import pypdf
except ImportError:
    pypdf = None

from qa_system.document_processors.pdf_processor import PDFDocumentProcessor

class DummyConfig:
    def get_nested(self, key, default=None):
        if key == 'DOCUMENT_PROCESSING.MAX_CHUNK_SIZE':
            return 50
        if key == 'DOCUMENT_PROCESSING.MIN_CHUNK_SIZE':
            return 10
        if key == 'DOCUMENT_PROCESSING.CHUNK_OVERLAP':
            return 10
        if key == 'DOCUMENT_PROCESSING.PRESERVE_SENTENCES':
            return True
        return default

@pytest.mark.skipif(pypdf is None, reason="pypdf not installed")
def test_pdf_processor_basic(tmp_path):
    # Create a simple PDF file with 2 pages
    pdf_path = tmp_path / 'sample.pdf'
    writer = pypdf.PdfWriter()
    # Add blank pages
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    # Add text to each page using reportlab
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        rl_pdf_path = tmp_path / 'rl_sample.pdf'
        c = canvas.Canvas(str(rl_pdf_path), pagesize=letter)
        c.drawString(100, 700, "Page 1 text.")
        c.showPage()
        c.drawString(100, 700, "Page 2 text.")
        c.save()
        pdf_path = rl_pdf_path
    except ImportError:
        # Fallback: blank pages
        with open(pdf_path, 'wb') as f:
            writer.write(f)
    proc = PDFDocumentProcessor(DummyConfig())
    result = proc.process(str(pdf_path))
    assert 'metadata' in result
    assert 'chunks' in result
    assert result['metadata']['filename_full'].endswith('.pdf')
    assert result['metadata']['file_type'] == 'pdf'
    assert result['metadata']['chunk_count'] == len(result['chunks'])
    assert result['metadata']['page_count'] >= 1
    assert all(isinstance(chunk, str) for chunk in result['chunks'])
    assert sum(len(chunk) for chunk in result['chunks']) == result['metadata']['total_tokens']

@pytest.mark.skipif(pypdf is None, reason="pypdf not installed")
def test_pdf_processor_metadata_override(tmp_path):
    # Create a simple PDF file
    pdf_path = tmp_path / 'meta.pdf'
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, 'wb') as f:
        writer.write(f)
    proc = PDFDocumentProcessor(DummyConfig())
    custom_meta = {'custom': 'value'}
    result = proc.process(str(pdf_path), metadata=custom_meta)
    assert result['metadata']['custom'] == 'value'
    assert result['metadata']['filename_full'] == 'meta.pdf'

def test_pdf_processor_import_error():
    # Simulate pypdf not being installed
    with patch.dict('sys.modules', {'pypdf': None}):
        proc = PDFDocumentProcessor(DummyConfig())
        with pytest.raises(ImportError) as exc_info:
            proc.process('test.pdf')
        assert "pypdf is required for PDF processing" in str(exc_info.value)

@pytest.mark.skipif(pypdf is None, reason="pypdf not installed")
def test_pdf_processor_file_error(tmp_path):
    # Test with a non-existent file
    proc = PDFDocumentProcessor(DummyConfig())
    with pytest.raises(FileNotFoundError):
        proc.process(str(tmp_path / 'nonexistent.pdf'))

@pytest.mark.skipif(pypdf is None, reason="pypdf not installed")
def test_pdf_processor_corrupted_file(tmp_path):
    # Create a corrupted PDF file
    pdf_path = tmp_path / 'corrupted.pdf'
    with open(pdf_path, 'wb') as f:
        f.write(b'This is not a valid PDF file')
    proc = PDFDocumentProcessor(DummyConfig())
    with pytest.raises(Exception) as exc_info:
        proc.process(str(pdf_path))
    assert any(msg in str(exc_info.value) for msg in [
        "Error processing PDF file",
        "EOF marker not found",
        "Stream has ended unexpectedly",
        "invalid pdf header"
    ]) 