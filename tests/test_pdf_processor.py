import pytest
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

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

@pytest.mark.skipif(PyPDF2 is None, reason="PyPDF2 not installed")
def test_pdf_processor_basic(tmp_path):
    # Create a simple PDF file with 2 pages
    pdf_path = tmp_path / 'sample.pdf'
    writer = PyPDF2.PdfWriter()
    writer.add_page(PyPDF2.pdf.PageObject.create_blank_page(None, 72, 72))
    writer.add_page(PyPDF2.pdf.PageObject.create_blank_page(None, 72, 72))
    # Add text to each page
    # PyPDF2 can't add text directly, so we use a workaround: create a PDF with text using reportlab if available
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

@pytest.mark.skipif(PyPDF2 is None, reason="PyPDF2 not installed")
def test_pdf_processor_metadata_override(tmp_path):
    # Create a simple PDF file
    pdf_path = tmp_path / 'meta.pdf'
    writer = PyPDF2.PdfWriter()
    writer.add_page(PyPDF2.pdf.PageObject.create_blank_page(None, 72, 72))
    with open(pdf_path, 'wb') as f:
        writer.write(f)
    proc = PDFDocumentProcessor(DummyConfig())
    custom_meta = {'custom': 'value'}
    result = proc.process(str(pdf_path), metadata=custom_meta)
    assert result['metadata']['custom'] == 'value'
    assert result['metadata']['filename_full'] == 'meta.pdf' 