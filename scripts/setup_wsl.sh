#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "========================================"
echo "Smart LLM Router — WSL Setup"
echo "========================================"
echo "Project dir: $PROJECT_DIR"

# Ensure python3-venv is available
if ! python3 -m venv --help &>/dev/null; then
    echo "[*] Installing python3-venv..."
    sudo apt-get update && sudo apt-get install -y python3-venv
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[*] Virtual environment already exists."
fi

source "$VENV_DIR/bin/activate"

echo "[*] Upgrading pip..."
pip install --upgrade pip

echo "[*] Installing requirements..."
pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the venv:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To start all models:"
echo "  python scripts/serve_models.py"
echo ""
echo "To test models:"
echo "  python scripts/test_models.py"
echo ""
echo "To benchmark models:"
echo "  python scripts/benchmark_models.py"
