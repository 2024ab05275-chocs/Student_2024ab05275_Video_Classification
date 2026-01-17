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
DATASET_DIR="$PROJECT_ROOT/dataset"

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
if [ -f "$RAR_FILE" ]; then
    echo "✔ UCF101.rar already exists — skipping download"
else
    echo "⬇ Downloading UCF101.rar (~6.5 GB)"
    echo "⏳ Real-time download progress will be shown below"

    # Prefer wget if available (common in Git Bash / WSL)
    if command -v wget >/dev/null 2>&1; then

        # --progress=bar:
        #   Displays a clean progress bar with percentage, speed, and ETA
        wget --progress=bar -O "$RAR_FILE" "$UCF_URL"

    # Fallback to curl if wget is not installed
    elif command -v curl >/dev/null 2>&1; then

        # -L:
        #   Follows HTTP redirects (required for some servers)
        # curl shows a progress bar by default
        curl -L "$UCF_URL" -o "$RAR_FILE"

    # If neither tool is available, exit with a clear message
    else
        echo "❌ Neither wget nor curl found on system"
        echo "👉 Install Git Bash, WSL, or curl to proceed"
        exit 1
    fi

    echo "✔ Download completed successfully"
fi


# ------------------------------------------------------------
# STEP 3: EXTRACT DATASET (ONLY IF NOT ALREADY EXTRACTED)
# ------------------------------------------------------------

# Check whether the extracted dataset directory already exists.
# If it does, extraction is skipped to prevent overwriting files.
if [ -d "$EXTRACT_DIR" ]; then
    echo "✔ Dataset already extracted — skipping extraction"
else
    echo "📦 Extracting UCF101.rar..."

    # Prefer unrar (installed with WinRAR or unrar package)
    if command -v unrar >/dev/null 2>&1; then

        # x:
        #   Extract with full directory structure
        unrar x "$RAR_FILE" "$DATASET_DIR"

    # Fallback to 7z (7-Zip)
    elif command -v 7z >/dev/null 2>&1; then

        # -o:
        #   Specifies output directory
        7z x "$RAR_FILE" -o"$DATASET_DIR"

    # If no extraction tool is available, exit cleanly
    else
        echo "❌ Neither unrar nor 7z found on system"
        echo "👉 Install WinRAR or 7-Zip to extract .rar files"
        exit 1
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
