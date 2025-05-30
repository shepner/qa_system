import os
import re
import sys
import pkgutil
import importlib.util
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PREFIXES = ("qa_system",)

# Use sys.stdlib_module_names if available, else fallback
try:
    STDLIB_MODULES = set(sys.stdlib_module_names)
except AttributeError:
    import distutils.sysconfig
    stdlib_path = distutils.sysconfig.get_python_lib(standard_lib=True)
    STDLIB_MODULES = set(
        name for _, name, _ in pkgutil.iter_modules([stdlib_path])
    )

# Allowlist for packages that may be required but not directly imported
ALLOWED_EXTRA_PACKAGES = {
    "setuptools", "wheel", "distutils", "pkg_resources",  # build tools
    # Add others as needed for your stack
    "chromadb_rust_bindings", "dry_attr", "pil", "platformdirs", "inflect", "more_itertools", "typeguard", "pytz",
    "_distutils_hack", "gevent", "greenlet", "chardet", "pandas", "yaml", "dateutil", "markdown_it", "grpc",
    "pil",  # prevent false positive for PIL
    "markdown-it-py", "grpcio", "pypdf2", "python-dateutil",  # allow as extras in requirements.txt
    "pillow",  # for VisionDocumentProcessor
}

IGNORED_MODULES = {"_pytest"}

IMPORT_RE = re.compile(r"^(?:import|from)\s+([\w\.]+)", re.MULTILINE)

def find_imported_modules(pyfile):
    with open(pyfile) as f:
        content = f.read()
    modules = set()
    for match in IMPORT_RE.finditer(content):
        mod = match.group(1).split(".")[0]
        modules.add(mod)
    return modules

def is_valid_module_name(mod):
    if not mod or mod == '.' or mod == '__pycache__':
        return False
    if not mod.isidentifier():
        return False
    return True

def is_third_party(mod):
    if mod in STDLIB_MODULES:
        return False
    if mod in LOCAL_PREFIXES:
        return False
    if mod in ALLOWED_EXTRA_PACKAGES:
        return False
    # Try to find the module spec; if it's not found, assume third-party
    spec = importlib.util.find_spec(mod)
    if spec is None:
        return False  # Could be a typo or missing, but not third-party
    if spec.origin and 'site-packages' in spec.origin:
        return True
    return False

def test_requirements_dynamic():
    pyfiles = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for fname in files:
            if fname.endswith('.py'):
                pyfiles.append(os.path.join(root, fname))
    imported = set()
    for pyfile in pyfiles:
        imported |= find_imported_modules(pyfile)
    imported = set(mod for mod in imported if is_valid_module_name(mod))
    third_party = set(mod for mod in imported if is_third_party(mod))
    third_party -= IGNORED_MODULES
    # Parse requirements.txt
    reqs = set()
    with open(os.path.join(PROJECT_ROOT, 'requirements.txt')) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            pkg = re.split(r'[<>=]', line)[0].strip().lower()
            if pkg:
                reqs.add(pkg)
    # Lowercase for comparison
    third_party = set(m.lower() for m in third_party)
    third_party -= {pkg.lower() for pkg in ALLOWED_EXTRA_PACKAGES}
    missing = third_party - reqs
    extra = reqs - third_party - {pkg.lower() for pkg in ALLOWED_EXTRA_PACKAGES}
    assert not missing, f"Missing required packages in requirements.txt: {missing}"
    assert not extra, f"Extra packages in requirements.txt: {extra}" 