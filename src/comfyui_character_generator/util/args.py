import argparse
import sys

from comfyui_character_generator.util.constants import (
    ASPECT_RATIO, DEFAULT_ASPECT_RATIO, DEFAULT_BATCH, DEFAULT_GUIDANCE_SCALE,
    DEFAULT_LOOP_COUNT, DEFAULT_POSE_DETECTION_TYPE, DEFAULT_SEED_GENERATION,
    DEFAULT_STEPS, DEFAULT_VENV_PATH, DEFAULT_WIDTH, SEED)
from comfyui_character_generator.util.enums import (PoseDetectionType,
                                                    SeedGenerationMethod)


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Automated ComfyUI Character Generator.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to TOML config file.",
    )
    parser.add_argument(
        "--comfyui_path",
        type=str,
        default=None,
        help="Path to ComfyUI directory.",
    )
    parser.add_argument(
        "--venv_path",
        type=str,
        default=DEFAULT_VENV_PATH,
        help=f"Relative path to env directory. (default {DEFAULT_VENV_PATH})",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Single .safetensors checkpoint file. Relative to checkpoints directory.",
    )
    parser.add_argument(
        "--lora_paths",
        nargs="*",
        type=str,
        default=[],
        help="Paths to LoRA .safetensors files. Relative to loras directory.",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default=None,
        help="Path to ControlNet .safetensors file. Relative to controlnet directory.",
    )
    parser.add_argument(
        "--upscaler_path",
        type=str,
        default=None,
        help="Path to Upscaler model .safetensors file. Relative to upscale_models directory.",
    )
    parser.add_argument(
        "--disable_controlnet",
        action="store_true",
        help="Disable ControlNet.",
    )
    parser.add_argument(
        "--pose_detection_type",
        type=str,
        default=DEFAULT_POSE_DETECTION_TYPE,
        choices=PoseDetectionType.__members__.values(),
        help="Pose detection type. OpenPose: 0, Realistic LineArt: 1, Depth: 2.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Denoising steps (default {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Base RNG seed (index is added per prompt).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help=f"Guidance scale (default {DEFAULT_GUIDANCE_SCALE}).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Images *per* prompt (default {DEFAULT_BATCH}).",
    )
    parser.add_argument(
        "--lora_strengths",
        nargs="*",
        type=float,
        default=[],
        help="LoRA strengths. If provided should match the number of --lora_paths.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Image width (default {DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default=DEFAULT_ASPECT_RATIO,
        choices=ASPECT_RATIO.keys(),
        help=f"Image aspect ratio (default {DEFAULT_ASPECT_RATIO}).",
    )
    parser.add_argument(
        "--system_prompt_path",
        default=None,
        help="Path to system prompt text file.",
    )
    parser.add_argument(
        "--system_neg_prompt_path",
        default=None,
        help="Path to system negative prompt text file.",
    )
    parser.add_argument(
        "--neg_prompts_path",
        default=None,
        help="Path to negative prompts text file. If provided should match the number of sub prompts.",
    )
    parser.add_argument(
        "--sub_prompts_path",
        default=None,
        help="Path to sub-prompts text file. Prompts are separeted by newlines.",
    )
    parser.add_argument(
        "--face_swap_image_paths",
        nargs="*",
        type=str,
        default=[],
        help="Path to face swap image files. If provided should match the number of sub prompts.",
    )
    parser.add_argument(
        "--pose_image_paths",
        nargs="*",
        type=str,
        default=[],
        help="Path to pose image files. If provided should match the number of sub prompts.",
    )
    parser.add_argument(
        "--loop_count",
        type=int,
        default=DEFAULT_LOOP_COUNT,
        help=f"Number of generations (default {DEFAULT_LOOP_COUNT}).",
    )
    parser.add_argument(
        "--seed_generation",
        type=int,
        default=DEFAULT_SEED_GENERATION,
        choices=SeedGenerationMethod.__members__.values(),
        help="Seed generation method. Increment: 1, Decrement: 2, Random: 3.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Path to output directory.",
    )
    parser.add_argument(
        "--install_nodes",
        action="store_true",
        help="Install custom nodes. (Must provide --comfyui_path and --venv_path)",
    )
    return parser.parse_known_args(sys.argv)[0]
