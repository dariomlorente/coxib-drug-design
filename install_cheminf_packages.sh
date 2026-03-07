#!/bin/bash
# Extra cheminformatics packages for the broader TFG workflow.
# These are NOT required to run the coxibs pipeline, but are useful
# for retrosynthesis, molecular visualization, protein analysis, etc.
#
# Usage:
#   conda activate cheminf
#   bash install_cheminf_packages.sh

set -e

echo "================================================"
echo "Installing extra cheminformatics packages"
echo "================================================"
echo ""
echo "Make sure you activated the environment first:"
echo "  conda activate coxibs"
echo ""

# --- Conda packages (need conda-forge) ---

echo "[conda] Installing openbabel, xtb, and system libs..."
conda install -y -c conda-forge \
    openbabel=3.1.1 \
    xtb=6.7.1 \
    glib gtk3 pango mscorefonts \
    libgfortran=14.2.0

# --- Pip packages ---

echo ""
echo "1/14 - AQME (conformer generation, QM input preparation)..."
pip install aqme==2.0.0

echo "2/14 - ROBERT (ML for reactivity prediction)..."
pip install robert==2.1.0

echo "3/14 - PySide6 (GUI framework for ROBERT)..."
pip install PySide6==6.9.2 PySide6-Addons==6.9.2 PySide6-Essentials==6.9.2

echo "4/14 - psutil (system monitoring)..."
pip install psutil

echo "5/14 - aizynthfinder (retrosynthetic analysis)..."
pip install aizynthfinder

echo "6/14 - rdchiral (chiral chemistry)..."
pip install rdchiral

echo "7/14 - chemcrow (chemistry knowledge tool)..."
pip install chemcrow

echo "8/14 - rmrkl (orchestration framework)..."
pip install rmrkl

echo "9/14 - paper-qa (scientific paper analysis tool)..."
pip install paper-qa

echo "10/14 - RXN4Chemistry (IBM RXN API)..."
pip install RXN4Chemistry

echo "11/14 - molbloom (molecular search)..."
pip install molbloom

echo "12/14 - synspace (synthesizable chemical space)..."
pip install synspace

echo "13/14 - py3Dmol (3D molecular visualization in notebooks)..."
pip install py3Dmol

echo "14/14 - pdb2pqr + propka (protein structure analysis)..."
pip install pdb2pqr propka

echo ""
echo "================================================"
echo "Done! Extra packages installed:"
echo "================================================"
echo "  Conformers & QM:   aqme, openbabel, xtb"
echo "  ML:                robert, PySide6"
echo "  Retrosynthesis:    aizynthfinder, rdchiral"
echo "  Cheminformatics:   chemcrow, rmrkl, paper-qa"
echo "  APIs:              RXN4Chemistry"
echo "  Chemical space:    molbloom, synspace"
echo "  Visualization:     py3Dmol"
echo "  Proteins:          pdb2pqr, propka"
echo ""
