# Automated ComfyUI Character Generator

This tool replicates the ComfyUI workflow that is in the `workflows`
subdirectory.

This workflow does the following:

1. Loads a checkpoint
2. Loads a character LoRA (or any LoRA)
3. Loads a pose controlnet
4. Concatenates the system prompt, the dressing prompt, the face expression
   prompt and the pose and environment settings prompt
5. It also concatenates the system and user negative prompts
6. Evaluates a switch to either apply the controlnet pose or just bypass it
7. If the controlnet pose is to be applied, it calculates the pose provided by
   the user and applies it to the generation
8. Finally, after generation, it does a face swap, using the face image
   provided by the user, and applies it to the generated image
9. It then saves the image to the output directory

The script does exactly the same thing as the ComfyUI workflow but it allows
configuration of some parameters (you just have to run the script with the
`--help` option to learn more about the parameters).

To run this tool, just type:

```sh
uvx --from git+https://github.com/jpchauvel/comfyui-character-generator comfyui-character-generator --help
```

It will prompt with all the necessary information.

**NOTE**: You can install the custom nodes using the following command:

```sh
uvx --from git+https://github.com/jpchauvel/comfyui-character-generator comfyui-character-generator --install_nodes --confyui_path /path/to/comfyui --venv_path relative/path/to/venv
```

Take a look at the [sample configuration file](/config.toml.example).
