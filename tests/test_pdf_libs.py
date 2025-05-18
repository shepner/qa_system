import sys
from pathlib import Path

PDF_PATH = Path("/Users/shepner/src/qa_system/docs/Library/Reference/The TOGAF Standard 10th Edition/Guides/_resources/TOGAF - Define and Govern SOA.pdf")

results = {}

# Test PyPDF
try:
    import pypdf
    try:
        reader = pypdf.PdfReader(str(PDF_PATH))
        if reader.is_encrypted:
            decrypt_result = reader.decrypt("")
            if decrypt_result == 1:
                text = reader.pages[0].extract_text()
                results['pypdf'] = f"Success (decrypted with empty password): {text[:100]!r}"
            else:
                results['pypdf'] = "Failed: Encrypted and could not decrypt with empty password."
        else:
            text = reader.pages[0].extract_text()
            results['pypdf'] = f"Success (not encrypted): {text[:100]!r}"
    except Exception as e:
        results['pypdf'] = f"Exception: {e}"
except ImportError:
    results['pypdf'] = "Not installed"

# Test PyMuPDF
try:
    import fitz  # PyMuPDF
    try:
        doc = fitz.open(str(PDF_PATH))
        text = doc[0].get_text()
        results['pymupdf'] = f"Success: {text[:100]!r}"
    except Exception as e:
        results['pymupdf'] = f"Exception: {e}"
except ImportError:
    results['pymupdf'] = "Not installed"

# Test pdfminer
try:
    from pdfminer.high_level import extract_text
    try:
        text = extract_text(str(PDF_PATH), maxpages=1)
        results['pdfminer'] = f"Success: {text[:100]!r}"
    except Exception as e:
        results['pdfminer'] = f"Exception: {e}"
except ImportError:
    results['pdfminer'] = "Not installed"

print("\nPDF Library Read Test Results:")
for lib, result in results.items():
    print(f"{lib}: {result}") 