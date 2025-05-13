#!/usr/bin/env python3
"""
Scan the document directory and generate a markdown report with recommendations for
DOCUMENT_PROCESSING, EMBEDDING_MODEL, and QUERY settings based on corpus analysis and config strategies.

This script does not accept any command-line arguments. All configuration is read from config/config.yaml.
"""
import os
import sys
import yaml
import statistics
from pathlib import Path
from collections import Counter

try:
    import pypdf
except ImportError:
    pypdf = None

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
REPORT_PATH = os.path.join(os.path.dirname(__file__), '../config/config_recommendations.md')

# Utility: Load YAML config
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Utility: Recursively scan for allowed files
def scan_files(root, allowed_exts, exclude_patterns):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Exclude directories
        dirnames[:] = [d for d in dirnames if not any(
            Path(os.path.join(dirpath, d)).match(pattern) for pattern in exclude_patterns
        )]
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            ext = fname.lower().rsplit('.', 1)[-1] if '.' in fname else ''
            if ext in allowed_exts and not any(Path(fpath).match(pattern) for pattern in exclude_patterns):
                files.append(fpath)
    return files

# Utility: Get file stats
def get_file_stats(files):
    sizes = []
    type_counts = Counter()
    for f in files:
        try:
            stat = os.stat(f)
            sizes.append(stat.st_size)
            ext = f.lower().rsplit('.', 1)[-1]
            type_counts[ext] += 1
        except Exception:
            continue
    return sizes, type_counts

# Utility: Percentile
def percentile(data, pct):
    if not data:
        return None
    data = sorted(data)
    k = int(len(data) * pct / 100)
    k = min(k, len(data) - 1)
    return data[k]

# Utility: Analyze all files for line and paragraph lengths, and per-file stats
def analyze_file_lengths(files):
    line_lens = []
    para_lens = []
    file_stats = []  # List of dicts: {file, size, line_count, para_count, line_lens, para_lens}
    for f in files:
        try:
            ext = f.lower().rsplit('.', 1)[-1]
            stat = os.stat(f)
            if ext == 'pdf' and pypdf:
                lines = []
                paras = []
                with open(f, 'rb') as pdf_file:
                    reader = pypdf.PdfReader(pdf_file)
                    for page in reader.pages:
                        text = page.extract_text() or ''
                        paras += [p for p in text.split('\n\n') if p.strip()]
                        lines += [l for l in text.splitlines() if l.strip()]
                file_line_lens = [len(l) for l in lines]
                file_para_lens = [len(p) for p in paras]
            else:
                with open(f, 'r', encoding='utf-8', errors='ignore') as tfile:
                    text = tfile.read()
                    paras = [p for p in text.split('\n\n') if p.strip()]
                    lines = [l for l in text.splitlines() if l.strip()]
                    file_line_lens = [len(l) for l in lines]
                    file_para_lens = [len(p) for p in paras]
            line_lens.extend(file_line_lens)
            para_lens.extend(file_para_lens)
            file_stats.append({
                'file': f,
                'size': stat.st_size,
                'line_count': len(file_line_lens),
                'para_count': len(file_para_lens),
                'line_lens': file_line_lens,
                'para_lens': file_para_lens,
            })
        except Exception:
            continue
    return line_lens, para_lens, file_stats

# Utility: Markdown section
SECTION = lambda title: f"\n\n## {title}\n\n"

# Utility: Format stats for a list of numbers
def format_stats(label, data):
    if not data:
        return f"- **{label}:** No data"
    return (f"- **{label}:** min={min(data)}, max={max(data)}, "
            f"mean={int(statistics.mean(data))}, median={int(statistics.median(data))}, "
            f"80th percentile={percentile(data, 80)}")

# Main logic
def main():
    # Print error and exit if any arguments are given
    if len(sys.argv) > 1:
        print("This script does not accept any command-line arguments.")
        sys.exit(1)

    config = load_config(CONFIG_PATH)
    scanner = config.get('FILE_SCANNER', {})
    doc_path = scanner.get('DOCUMENT_PATH', './docs')
    allowed_exts = set(e.lower() for e in scanner.get('ALLOWED_EXTENSIONS', ['txt', 'md', 'pdf']))
    exclude_patterns = scanner.get('EXCLUDE_PATTERNS', [])

    # Always resolve DOCUMENT_PATH relative to project root (parent of tools/) unless absolute
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if os.path.isabs(doc_path):
        scan_path = doc_path
    else:
        scan_path = os.path.abspath(os.path.join(project_root, doc_path))

    files = scan_files(scan_path, allowed_exts, exclude_patterns)
    sizes, type_counts = get_file_stats(files)
    line_lens, para_lens, file_stats = analyze_file_lengths(files) if files else ([], [], [])

    # Prepare report
    report = [
        "# QA System Configuration Recommendations",
        "",
        "This report analyzes the current document corpus and provides recommendations for initial settings in `DOCUMENT_PROCESSING`, `EMBEDDING_MODEL`, and `QUERY`. Recommendations are based on corpus statistics and the strategies/comments in your `config.yaml`."
    ]

    # Corpus summary
    report.append(SECTION("Corpus Summary"))
    if not files:
        report.append(
            f"**No files found in `{scan_path}` with allowed extensions {sorted(allowed_exts)}.**\n"
            "Recommendations below are based on sensible defaults and config strategies."
        )
    else:
        report.append(f"- **Total files:** {len(files)}")
        report.append(f"- **File types:** {dict(type_counts)}")
        if sizes:
            report.append(f"- **File size (bytes):** min={min(sizes)}, max={max(sizes)}, mean={int(statistics.mean(sizes))}, median={int(statistics.median(sizes))}")
        if line_lens:
            report.append(format_stats("Line length (all data)", line_lens))
        if para_lens:
            report.append(format_stats("Paragraph length (all data)", para_lens))

        # Typical file (median file size)
        if file_stats:
            sorted_by_size = sorted(file_stats, key=lambda x: x['size'])
            median_idx = len(sorted_by_size) // 2
            typical = sorted_by_size[median_idx]
            report.append("\n### Typical File (median file size)")
            report.append(f"- **File name:** {os.path.relpath(typical['file'], project_root)}")
            report.append(f"- **File size (bytes):** {typical['size']}")
            report.append(f"- **Line count:** {typical['line_count']}")
            report.append(f"- **Paragraph count:** {typical['para_count']}")
            report.append(format_stats("Line length", typical['line_lens']))
            report.append(format_stats("Paragraph length", typical['para_lens']))

        # Middle 80% of files by size
        if len(file_stats) >= 10:
            sorted_by_size = sorted(file_stats, key=lambda x: x['size'])
            n = len(sorted_by_size)
            start = n // 10
            end = n - n // 10
            middle_80 = sorted_by_size[start:end]
            mid80_line_lens = [l for f in middle_80 for l in f['line_lens']]
            mid80_para_lens = [p for f in middle_80 for p in f['para_lens']]
            report.append("\n### Middle 80% of Files (by size)")
            report.append(f"- **Files included:** {len(middle_80)}")
            report.append(format_stats("Line length (middle 80%)", mid80_line_lens))
            report.append(format_stats("Paragraph length (middle 80%)", mid80_para_lens))

    # DOCUMENT_PROCESSING recommendations
    report.append(SECTION("DOCUMENT_PROCESSING Recommendations"))
    doc_proc = config.get('DOCUMENT_PROCESSING', {})
    if para_lens:
        recommended_chunk = min(max(int(statistics.median(para_lens) * 2), 512), doc_proc.get('MAX_CHUNK_SIZE', 2048))
        chunk_strategy = (
            "Set to 2x the median paragraph length (for context preservation), "
            "but not less than 512 and not more than the configured MAX_CHUNK_SIZE. "
            "This balances chunk size for embedding efficiency and retrieval context."
        )
    else:
        recommended_chunk = doc_proc.get('MAX_CHUNK_SIZE', 2048)
        chunk_strategy = (
            "Defaulted to config value (MAX_CHUNK_SIZE) due to lack of paragraph data."
        )
    min_chunk = max(int(recommended_chunk // 2), 256)
    min_chunk_strategy = (
        "Set to half of MAX_CHUNK_SIZE, but not less than 256. "
        "Ensures small documents are not over-chunked and that minimum chunk size is practical for most models."
    )
    overlap = max(int(recommended_chunk * 0.2), 128)
    overlap_strategy = (
        "Set to 20% of MAX_CHUNK_SIZE, but not less than 128. "
        "This overlap helps preserve context across chunk boundaries, especially for long sentences or paragraphs."
    )
    batch_size = min(100, max(10, len(files)//2 if files else 50))
    batch_strategy = (
        "Set to min(100, max(10, N/2)), where N is the number of files. "
        "This balances throughput and memory usage for parallel processing."
    )
    concurrent_tasks = doc_proc.get('CONCURRENT_TASKS', 6)
    concurrent_strategy = (
        "Taken from config or default (6). "
        "Controls the number of parallel processing tasks. Increase for large corpora if system resources allow."
    )
    preserve_sentences = doc_proc.get('PRESERVE_SENTENCES', True)
    preserve_strategy = (
        "Taken from config or default (True). "
        "Preserves sentence boundaries during chunking for better semantic splits."
    )
    # Output YAML config block
    report.append("```yaml")
    report.append("DOCUMENT_PROCESSING:")
    report.append(f"  MAX_CHUNK_SIZE: {recommended_chunk}")
    report.append(f"  MIN_CHUNK_SIZE: {min_chunk}")
    report.append(f"  CHUNK_OVERLAP: {overlap}")
    report.append(f"  BATCH_SIZE: {batch_size}")
    report.append(f"  CONCURRENT_TASKS: {concurrent_tasks}")
    report.append(f"  PRESERVE_SENTENCES: {preserve_sentences}")
    report.append("```")
    # Output strategy explanations
    report.append(f"- `MAX_CHUNK_SIZE`: {recommended_chunk}\n  # Strategy: {chunk_strategy}")
    report.append(f"- `MIN_CHUNK_SIZE`: {min_chunk}\n  # Strategy: {min_chunk_strategy}")
    report.append(f"- `CHUNK_OVERLAP`: {overlap}\n  # Strategy: {overlap_strategy}")
    report.append(f"- `BATCH_SIZE`: {batch_size}\n  # Strategy: {batch_strategy}")
    report.append(f"- `CONCURRENT_TASKS`: {concurrent_tasks}\n  # Strategy: {concurrent_strategy}")
    report.append(f"- `PRESERVE_SENTENCES`: {preserve_sentences}\n  # Strategy: {preserve_strategy}")

    # EMBEDDING_MODEL recommendations
    report.append(SECTION("EMBEDDING_MODEL Recommendations"))
    emb = config.get('EMBEDDING_MODEL', {})
    max_length = max(recommended_chunk, emb.get('MAX_LENGTH', 2048))
    max_length_strategy = (
        "Set to the greater of MAX_CHUNK_SIZE and the configured model max length. "
        "Ensures that the embedding model can process the largest chunk."
    )
    dims = emb.get('DIMENSIONS', 768)
    dims_strategy = (
        "Taken from config or default (768). "
        "Should match the embedding model's output vector size."
    )
    emb_batch = min(emb.get('BATCH_SIZE', 100), batch_size)
    emb_batch_strategy = (
        "Set to the lower of embedding model batch size and document batch size. "
        "Optimizes throughput while respecting model and system limits."
    )
    model_name = emb.get('MODEL_NAME', 'text-embedding-004')
    model_name_strategy = (
        "Taken from config or default. "
        "Choose a model that supports the required max length and dimensions."
    )
    report.append("```yaml")
    report.append("EMBEDDING_MODEL:")
    report.append(f"  MAX_LENGTH: {max_length}")
    report.append(f"  DIMENSIONS: {dims}")
    report.append(f"  BATCH_SIZE: {emb_batch}")
    report.append(f"  MODEL_NAME: {model_name}")
    report.append("```")
    report.append(f"- `MAX_LENGTH`: {max_length}\n  # Strategy: {max_length_strategy}")
    report.append(f"- `DIMENSIONS`: {dims}\n  # Strategy: {dims_strategy}")
    report.append(f"- `BATCH_SIZE`: {emb_batch}\n  # Strategy: {emb_batch_strategy}")
    report.append(f"- `MODEL_NAME`: {model_name}\n  # Strategy: {model_name_strategy}")

    # QUERY recommendations
    report.append(SECTION("QUERY Recommendations"))
    query = config.get('QUERY', {})
    if files:
        if len(files) < 1000:
            top_k = 10 if recommended_chunk > 1500 else 20
            topk_strategy = (
                "Set to 10 if MAX_CHUNK_SIZE > 1500, else 20. "
                "Smaller corpora and larger chunks need fewer retrievals."
            )
        elif len(files) < 10000:
            top_k = 30 if recommended_chunk > 1500 else 50
            topk_strategy = (
                "Set to 30 if MAX_CHUNK_SIZE > 1500, else 50. "
                "Medium corpora benefit from more retrievals for coverage."
            )
        else:
            top_k = 75
            topk_strategy = (
                "Set to 75 for very large corpora. "
                "Ensures broad retrieval coverage in large datasets."
            )
    else:
        top_k = query.get('TOP_K', 75)
        topk_strategy = (
            "Defaulted to config value (TOP_K) due to lack of corpus data."
        )
    min_sim = query.get('MIN_SIMILARITY', 0.1)
    min_sim_strategy = (
        "Taken from config or default (0.1). "
        "Lower for small corpora, higher for large or noisy corpora."
    )
    temp = query.get('TEMPERATURE', 0.95)
    temp_strategy = (
        "Taken from config or default (0.95). "
        "Controls randomness of LLM output. 0.0 = deterministic, 1.0 = creative."
    )
    max_tokens = query.get('MAX_TOKENS', 1024)
    max_tokens_strategy = (
        "Taken from config or default (1024). "
        "Maximum tokens to generate in LLM response. Tune for your use case."
    )
    context_window = max_length * 4  # Heuristic: 4x max_length
    context_strategy = (
        "Set to 4x MAX_LENGTH as a heuristic for context window size. "
        "Do not exceed the model's input token limit."
    )
    report.append("```yaml")
    report.append("QUERY:")
    report.append(f"  TOP_K: {top_k}")
    report.append(f"  MIN_SIMILARITY: {min_sim}")
    report.append(f"  TEMPERATURE: {temp}")
    report.append(f"  MAX_TOKENS: {max_tokens}")
    report.append(f"  CONTEXT_WINDOW: {context_window}")
    report.append("```")
    report.append(f"- `TOP_K`: {top_k}\n  # Strategy: {topk_strategy}")
    report.append(f"- `MIN_SIMILARITY`: {min_sim}\n  # Strategy: {min_sim_strategy}")
    report.append(f"- `TEMPERATURE`: {temp}\n  # Strategy: {temp_strategy}")
    report.append(f"- `MAX_TOKENS`: {max_tokens}\n  # Strategy: {max_tokens_strategy}")
    report.append(f"- `CONTEXT_WINDOW`: {context_window}\n  # Strategy: {context_strategy}")

    # Write report
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"Report written to {REPORT_PATH}")

if __name__ == '__main__':
    main() 