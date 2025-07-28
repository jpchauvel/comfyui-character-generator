#!/bin/sh
set -euo pipefail

# Check for required arguments
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <venv_path> <PYTHONPATH> <config_idx> <prompt_idx>"
  exit 1
fi

VENV_PATH="$1"
PYTHON_PATH="$2"
CONFIG_IDX="$3"
PROMPT_IDX="$4"

# Read stdin safely (non-blocking if no input piped)
if [ -t 0 ]; then
  echo "Error: No input received from stdin." >&2
  exit 1
fi

stdin_input=$(cat)

# Activate virtualenv
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "Error: Virtual environment not found at ${VENV_PATH}/bin/activate"
  exit 1
fi
# shellcheck source=/dev/null
. "${VENV_PATH}/bin/activate"

# Check if script exists
SCRIPT_PATH="${PYTHON_PATH}/comfyui_character_generator/generate.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
  echo "Error: Script not found at ${SCRIPT_PATH}"
  exit 1
fi

# Execute script with stdin as input
printf "%s" "$stdin_input" | PYTHONPATH="${PYTHON_PATH}" python3 "${SCRIPT_PATH}" --config_idx "$CONFIG_IDX" --prompt_idx "$PROMPT_IDX"
