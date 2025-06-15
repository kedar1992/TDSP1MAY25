#!/usr/bin/env bash

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet
