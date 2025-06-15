#!/usr/bin/env bash

# Install Python dependencies
pip install -r requirements.txt

# Create a directory for NLTK data
mkdir -p ./nltk_data

# Download NLTK resources into that directory
python -m nltk.downloader -d ./nltk_data punkt stopwords wordnet
