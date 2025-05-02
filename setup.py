from setuptools import setup, find_packages

setup(
    name="qa_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml",  # For config file parsing
    ],
    python_requires=">=3.13",
) 