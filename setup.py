from setuptools import setup, find_packages

setup(
    name="edge-rag",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "qdrant-client",
        "sentence-transformers",
        "python-dotenv",
        "requests",
        "ollama",
        "numpy",
        "nltk"
    ]
) 