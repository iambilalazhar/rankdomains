#!/usr/bin/env bash
set -euo pipefail

# Simple venv bootstrapper
PY=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv in $VENV_DIR"
  $PY -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "\nDone. Activate with: source $VENV_DIR/bin/activate"
