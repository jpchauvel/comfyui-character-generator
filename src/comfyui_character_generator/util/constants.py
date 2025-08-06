import random

from comfyui_character_generator.util.enums import (PoseDetectionType,
                                                    SeedGenerationMethod)

ASPECT_RATIO: dict[str, float] = {
    "1:1": 1.0,
    "4:3": 3 / 4,
    "3:4": 4 / 3,
    "5:4": 4 / 5,
    "4:5": 5 / 4,
    "16:9": 9 / 16,
    "9:16": 16 / 9,
}

MODEL_DIRECTORY: str = "models"
CHECKPOINT_DIRECTORY: str = "checkpoints"
LORA_DIRECTORY: str = "loras"
CONTROLNET_DIRECTORY: str = "controlnet"
UPSCALER_DIRECTORY: str = "upscale_models"
DEFAULT_VENV_PATH: str = ".venv"
DEFAULT_DISABLE_CONTROLNET: bool = False
DEFAULT_STEPS: int = 35
SEED: int = random.randint(1, 2**64)
DEFAULT_GUIDANCE_SCALE: float = 8.0
DEFAULT_BATCH: int = 1
DEFAULT_WIDTH: int = 1024
DEFAULT_ASPECT_RATIO: str = "1:1"
DEFAULT_LOOP_COUNT: int = 1
DEFAULT_SEED_GENERATION: SeedGenerationMethod = (
    SeedGenerationMethod.INCREMENT
)  # 1: Increment, 2: Decrement, 3: Random
DEFAULT_POSE_DETECTION_TYPE: PoseDetectionType = (
    PoseDetectionType.OPENPOSE
)  # 0: OpenPose, 1: Realistic Lineart, 2: Depth
