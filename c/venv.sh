#!/usr/bin/env bash
# colibri virtual environment helper
# Usage: source venv.sh

if [ -z "$COLI_VENV_ACTIVE" ]; then
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        export COLI_VENV_ACTIVE=1
        echo "✓ colibri venv activated (numpy, huggingface_hub, safetensors)"
    else
        echo "Creating venv..."
        python3 -m venv .venv
        source .venv/bin/activate
        python -m pip install --quiet --upgrade pip
        python -m pip install --quiet numpy huggingface_hub safetensors
        export COLI_VENV_ACTIVE=1
        echo "✓ venv created and activated"
    fi
fi
