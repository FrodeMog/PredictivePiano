#!/bin/bash

# Activate your virtual environment
source venv/bin/activate

# Install PyTorch with the correct CUDA version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the rest of your requirements
pip3 install -r requirements.txt