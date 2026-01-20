# ============================================================
# JUPYTER → PDF CONVERSION IMAGE
# ============================================================

FROM jupyter/scipy-notebook:latest

USER root

# ------------------------------------------------------------
# Install system dependencies for PDF export
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    pandoc \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-latex-extra \
    texlive-lang-english \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Switch back to jovyan user
# ------------------------------------------------------------
USER jovyan

WORKDIR /home/jovyan/work
