#!/usr/bin/env bash
set -e

###############################################################################
# SETUP SCRIPT – VIDEO CLASSIFICATION PROJECT
# Uses a Python virtual environment to comply with PEP 668
###############################################################################

echo "🔍 Detecting operating system..."

OS_TYPE="unknown"
case "$OSTYPE" in
  darwin*)  OS_TYPE="macos" ;;
  linux*)   OS_TYPE="linux" ;;
  msys*|cygwin*) OS_TYPE="windows" ;;
esac

echo "🖥️  OS detected: $OS_TYPE"
echo

###############################################################################
# 1️⃣ CHECK PYTHON
###############################################################################
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python3 not found"
    echo "➡ Please install Python 3.9+ from https://python.org"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo

###############################################################################
# 2️⃣ CREATE PROJECT STRUCTURE
###############################################################################
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$PROJECT_ROOT/code"
DATASET_DIR="$PROJECT_ROOT/dataset"
VENV_DIR="$PROJECT_ROOT/.venv"

mkdir -p "$CODE_DIR" "$DATASET_DIR" "$DATASET_DIR/splits"
echo "📁 Folder structure ready"
echo

###############################################################################
# 3️⃣ CREATE & ACTIVATE VIRTUAL ENVIRONMENT
###############################################################################
if [[ ! -d "$VENV_DIR" ]]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "⏭️ Virtual environment already exists"
fi

echo "🔑 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "✅ Using Python: $(which python)"
echo

###############################################################################
# 4️⃣ INSTALL PYTHON DEPENDENCIES
###############################################################################
REQ_FILE="$CODE_DIR/requirements.txt"

if [[ ! -f "$REQ_FILE" ]]; then
    echo "❌ requirements.txt not found at $REQ_FILE"
    exit 1
fi

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r "$REQ_FILE"
echo "✅ Dependencies installed"
echo

###############################################################################
# 5️⃣ CHECK SYSTEM TOOL (7z)
###############################################################################
echo "🔎 Checking 7z extractor..."

if ! command -v 7z >/dev/null 2>&1; then
    echo "❌ 7z not found"
    if [[ "$OS_TYPE" == "macos" ]]; then
        echo "➡ Install using: brew install p7zip"
    elif [[ "$OS_TYPE" == "linux" ]]; then
        echo "➡ Install using: sudo apt install p7zip-full"
    else
        echo "➡ Install 7-Zip and add to PATH (Windows)"
    fi
    exit 1
fi

echo "✅ 7z found"
echo

###############################################################################
# 6️⃣ DOWNLOAD DATA + CREATE SUBSET + SPLITS
###############################################################################
DATA_LOADER="$CODE_DIR/data_loader.py"

if [[ ! -f "$DATA_LOADER" ]]; then
    echo "❌ data_loader.py not found"
    exit 1
fi

echo "🚀 Running dataset preparation..."
python "$DATA_LOADER"

echo
echo "🎉 SETUP COMPLETED SUCCESSFULLY"
echo "➡ Virtual environment: .venv"
echo "➡ Activate later using: source .venv/bin/activate"
