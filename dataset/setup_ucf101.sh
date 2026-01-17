#!/usr/bin/env bash

# ============================================================
# UCF-101 DATASET SETUP SCRIPT (WITH PROGRESS + SAFE RE-RUN)
# ============================================================
# PURPOSE:
#   This script prepares the UCF-101 dataset required by
#   data_loader.py. It performs the following steps:
#
#   1) Downloads the official UCF101.rar archive (only once)
#   2) Extracts the dataset into the expected directory layout
#   3) Verifies dataset availability before exiting
#
# KEY DESIGN FEATURES:
#   - Idempotent: Safe to run multiple times
#   - Bandwidth-efficient: Avoids re-downloading large files
#   - Transparent: Shows real-time download progress
#
# SUPPORTED ENVIRONMENTS:
#   - Windows Terminal (Git Bash)
#   - Windows Terminal (WSL - Ubuntu)
#   - MSYS2
#
# DATASET SOURCE (OFFICIAL):
#   https://www.crcv.ucf.edu/data/UCF101.php
# ============================================================


# ------------------------------------------------------------
# SAFETY SETTING
# ------------------------------------------------------------
# Exit immediately if any command fails.
# This prevents silent failures and partial downloads/extracts.
set -e


# ------------------------------------------------------------
# PROJECT PATH CONFIGURATION
# ------------------------------------------------------------

# Absolute path of the directory where the script is executed.
# This ensures paths are correct regardless of terminal location.
PROJECT_ROOT="$(pwd)"

# Dataset directory expected by data_loader.py
DATASET_DIR="$PROJECT_ROOT"

# Location where the downloaded .rar file will be stored
RAR_FILE="$DATASET_DIR/UCF101.rar"

# Location of the extracted dataset folder
# IMPORTANT: data_loader.py assumes this exact folder name
EXTRACT_DIR="$DATASET_DIR/UCF-101"

# Official UCF-101 dataset download URL
UCF_URL="https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"


# ------------------------------------------------------------
# STEP 1: CREATE DATASET DIRECTORY (IF NEEDED)
# ------------------------------------------------------------

echo "📁 Checking dataset directory..."

# mkdir -p:
#   - Creates the directory if it does not exist
#   - Does nothing if it already exists
#   - Avoids errors on repeated runs
mkdir -p "$DATASET_DIR"


# ------------------------------------------------------------
# STEP 2: DOWNLOAD DATASET WITH PROGRESS (ONLY IF REQUIRED)
# ------------------------------------------------------------

# Check whether the dataset archive already exists.
# If it does, downloading again is unnecessary and wasteful.

echo "$RAR_FILE"

if [ -f "$RAR_FILE" ]; then
    echo "✔ UCF101.rar already exists — skipping download"
else
    echo "⬇ Downloading UCF101.rar (~6.5 GB)"
    echo "⏳ Real-time download progress will be shown below"

    # Use wget if available
    if command -v wget >/dev/null 2>&1; then
        wget --progress=bar \
             --no-check-certificate \
             -O "$RAR_FILE" "$UCF_URL"

    # Fallback to curl
    elif command -v curl >/dev/null 2>&1; then
        curl -L -k "$UCF_URL" -o "$RAR_FILE"

    else
        echo "❌ Neither wget nor curl found"
        exit 1
    fi

    echo "✔ Download completed"
fi


# ------------------------------------------------------------
# STEP 3: EXTRACT DATASET (ONLY IF NOT ALREADY EXTRACTED)
# ------------------------------------------------------------
if [ -d "$EXTRACT_DIR" ]; then
    echo "✔ Dataset already extracted — skipping extraction"
else
    echo "📦 Extracting UCF101.rar..."

    # Check for unrar or 7z
    if command -v unrar >/dev/null 2>&1; then
        unrar x "$RAR_FILE" "$DATASET_DIR"
    elif command -v 7z >/dev/null 2>&1; then
        7z x "$RAR_FILE" -o"$DATASET_DIR"
    else
        # Auto-install extraction tools on Linux (apt-based)
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo "⚡ Installing 'unrar' and 'p7zip-full' automatically..."
            sudo apt update
            sudo apt install -y unrar p7zip-full
            echo "✔ Extraction tools installed, retrying extraction..."
            if command -v unrar >/dev/null 2>&1; then
                unrar x "$RAR_FILE" "$DATASET_DIR"
            elif command -v 7z >/dev/null 2>&1; then
                7z x "$RAR_FILE" -o"$DATASET_DIR"
            fi
        else
            echo "❌ Neither unrar nor 7z found on system"
            echo "👉 Please install WinRAR or 7-Zip manually"
            exit 1
        fi
    fi

    echo "✔ Extraction completed successfully"
fi


# ------------------------------------------------------------
# STEP 4: FINAL VERIFICATION
# ------------------------------------------------------------

# Final sanity check to ensure the dataset folder exists
# This confirms that data_loader.py can safely run
if [ -d "$EXTRACT_DIR" ]; then
    echo "✅ UCF-101 dataset is ready for use"
    echo "📂 Dataset path: $EXTRACT_DIR"
else
    echo "❌ Dataset setup failed — UCF-101 folder not found"
    exit 1
fi
