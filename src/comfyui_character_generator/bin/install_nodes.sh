#!/bin/sh
set -euo pipefail

# Check for required arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <venv_path> <custom_nodes_base_path>"
  exit 1
fi

VENV_PATH="$1"
CUSTOM_NODES_PATH="$2/custom_nodes"

# Check if virtualenv exists and activate
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "Error: Virtual environment not found at ${VENV_PATH}/bin/activate"
  exit 1
fi

# Activate the virtual environment
# shellcheck source=/dev/null
. "${VENV_PATH}/bin/activate"

echo "Installing Custom Nodes..."

# Create custom_nodes directory if it doesn't exist
mkdir -p "${CUSTOM_NODES_PATH}"

# Save current dir and switch to custom nodes
pushd "${CUSTOM_NODES_PATH}" > /dev/null

# Clone helper
clone_and_install() {
  repo_url="$1"
  target_dir="$2"

  if [ -d "${target_dir}" ]; then
    echo "Directory ${target_dir} already exists. Skipping clone."
  else
    echo "Cloning ${repo_url} into ${target_dir}..."
    git clone "${repo_url}" "${target_dir}" || {
      echo "Failed to clone ${repo_url}"
      return 1
    }
  fi

  if [ -f "${target_dir}/requirements.txt" ]; then
    echo "Installing requirements for ${target_dir}..."
    pip install -r "${target_dir}/requirements.txt" || {
      echo "Failed to install requirements for ${target_dir}"
      return 1
    }
  else
    echo "No requirements.txt found in ${target_dir}, skipping pip install."
  fi
}

# Repositories to clone
clone_and_install "git@github.com:rgthree/rgthree-comfy.git" "rgthree-comfy"
clone_and_install "git@github.com:yolain/ComfyUI-Easy-Use.git" "comfyui-easy-use"
clone_and_install "git@github.com:ssitu/ComfyUI_UltimateSDUpscale.git" "comfyui_ultimatesdupscale"
clone_and_install "git@github.com:Fannovel16/comfyui_controlnet_aux.git" "comfyui_controlnet_aux"

# Return to original directory
popd > /dev/null

echo "Done Installing Custom Nodes..."
