#!/usr/bin/env bash

# ============================================================
# BATCH JUPYTER NOTEBOOK → PDF (ROBUST MODE)
# ============================================================

ROOT_DIR="${1:-.}"

echo "📁 Root directory: $ROOT_DIR"
echo "--------------------------------------------"

echo "🔍 Checking dependencies..."
for cmd in jupyter pandoc xelatex; do
    if ! command -v "$cmd" >/dev/null; then
        echo "❌ Missing dependency: $cmd"
        exit 1
    fi
done
echo "✅ All dependencies available"
echo "--------------------------------------------"

mapfile -t NOTEBOOKS < <(
    find "$ROOT_DIR" -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*"
)

echo "📘 Found ${#NOTEBOOKS[@]} notebooks"
echo "--------------------------------------------"

SUCCESS=0
FAILED=0

for NB in "${NOTEBOOKS[@]}"; do
    echo ""
    echo "📘 Converting: $NB"

    jupyter nbconvert "$NB" \
        --to pdf \
        --PDFExporter.run_bibtex=False \
        --PDFExporter.latex_command="['xelatex', '{filename}', '-quiet']"

    EXIT_CODE=$?

    PDF="${NB%.ipynb}.pdf"

    if [ $EXIT_CODE -eq 0 ] && [ -f "$PDF" ]; then
        echo "✅ PDF generated: $PDF"
        ((SUCCESS++))
    else
        echo "❌ Conversion failed: $NB"
        ((FAILED++))
    fi
done

echo ""
echo "============================================"
echo "📊 CONVERSION SUMMARY"
echo "============================================"
echo "✅ Successful: $SUCCESS"
echo "❌ Failed:     $FAILED"
echo "============================================"
