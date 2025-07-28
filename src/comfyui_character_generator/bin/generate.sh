#!/bin/sh

stdin_input=$(cat -)

source "${1}/bin/activate"

printf "$stdin_input" | PYTHONPATH="${2}" python3 src/comfyui_character_generator/generate.py --prompt_idx ${3}
