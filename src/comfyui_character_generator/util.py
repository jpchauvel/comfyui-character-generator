import argparse
import os
import pathlib
import random
import shutil
import sys
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

from tomlkit import TOMLDocument, document, dumps, loads, table
from tomlkit.items import Table

ASPECT_RATIO: dict[str, float] = {
    "1:1": 1.0,
    "4:3": 3 / 4,
    "3:4": 4 / 3,
    "16:9": 9 / 16,
    "9:16": 16 / 9,
}


class SeedGenerationMethod(IntEnum):
    INCREMENT = auto()
    DECREMENT = auto()
    RANDOM = auto()


MODEL_DIRECTORY: str = "models"
CHECKPOINT_DIRECTORY: str = "checkpoints"
LORA_DIRECTORY: str = "loras"
CONTROLNET_DIRECTORY: str = "controlnet"
DEFAULT_VENV_PATH: str = ".venv"
DEFAULT_DISABLE_CONTROLNET: bool = False
DEFAULT_STEPS: int = 35
SEED: int = random.randint(1, 2**64)
DEFAULT_GUIDANCE_SCALE: float = 8.0
DEFAULT_BATCH: int = 1
DEFAULT_LORA_STRENGTH: float = 1.5
DEFAULT_WIDTH: int = 1024
DEFAULT_ASPECT_RATIO: str = "1:1"
DEFAULT_LOOP_COUNT: int = 1
DEFAULT_SEED_GENERATION: SeedGenerationMethod = (
    SeedGenerationMethod.INCREMENT
)  # 1: Increment, 2: Decrement, 3: Random


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
        help="Single .safetensors checkpoint file. Relative to checkpoint directory.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to character LoRA .safetensors file. Relative to lora directory.",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default=None,
        help="Path to ControlNet .safetensors file. Relative to controlnet directory.",
    )
    parser.add_argument(
        "--disable_controlnet",
        action="store_true",
        help="Disable ControlNet.",
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
        "--lora_strength",
        type=float,
        default=DEFAULT_LORA_STRENGTH,
        help=f"Character LoRA strength (default {DEFAULT_LORA_STRENGTH}).",
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
        "--neg_prompts_path",
        default=None,
        help="Path to negative prompts text file.",
    )
    parser.add_argument(
        "--sub_prompts_path",
        default=None,
        help="Path to sub-prompts text file. Prompts are separeted by newlines.",
    )
    parser.add_argument(
        "--face_swap_image_path",
        default=None,
        help="Path to face swap image file.",
    )
    parser.add_argument(
        "--pose_image_path",
        default=None,
        help="Path to pose image file.",
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
        help="Seed generation method (1: Increment, 2: Decrement, 3: Random).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Path to output directory.",
    )
    return parser.parse_known_args(sys.argv)[0]


@dataclass()
class Config:
    comfyui_path: pathlib.Path | None = None
    venv_path: pathlib.Path | None = None
    ckpt: str | None = None
    lora: str | None = None
    controlnet: str | None = None
    disable_controlnet: bool = DEFAULT_DISABLE_CONTROLNET
    steps: int = DEFAULT_STEPS
    seed: int = SEED
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    batch: int = DEFAULT_BATCH
    lora_strength: float = DEFAULT_LORA_STRENGTH
    width: int = DEFAULT_WIDTH
    height: int = int(DEFAULT_WIDTH * ASPECT_RATIO[DEFAULT_ASPECT_RATIO])
    aspect_ratio: str = DEFAULT_ASPECT_RATIO
    system_prompt: str = ""
    neg_prompts: list[str] = field(default_factory=list)
    sub_prompts: list[str] = field(default_factory=list)
    face_swap_image: str | None = None
    pose_image: str | None = None
    loop_count: int = DEFAULT_LOOP_COUNT
    seed_generation: SeedGenerationMethod = DEFAULT_SEED_GENERATION
    output_path: pathlib.Path = pathlib.Path("")

    def __init__(self) -> None:
        super().__init__()
        self._neg_prompt_iter = None
        self._sub_prompt_iter = None

    @property
    def next_neg_prompt(self) -> str:
        if self._neg_prompt_iter is None:
            self._neg_prompt_iter = iter(self.neg_prompts)
        return next(self._neg_prompt_iter, "")

    @property
    def next_sub_prompt(self) -> str:
        if self._sub_prompt_iter is None:
            self._sub_prompt_iter = iter(self.sub_prompts)
        return next(self._sub_prompt_iter, "")


def add_with_rotate_64(a, b) -> int:
    full_sum: int = a + b
    base: int = full_sum & 0xFFFFFFFFFFFFFFFF  # Normal 64-bit truncation

    # Number of bits overflowed (max possible: up to 64)
    overflow_bit_count: int = (
        full_sum.bit_length() - 64 if full_sum.bit_length() > 64 else 0
    )

    # Rotate-left base by number of overflow bits
    rotated: int = (
        (base << overflow_bit_count) | (base >> (64 - overflow_bit_count))
    ) & 0xFFFFFFFFFFFFFFFF
    return rotated


def sub_with_rotate_64(a, b) -> int:
    diff: int = a - b

    if diff >= 0:
        # No underflow — just normal subtraction
        return diff & 0xFFFFFFFFFFFFFFFF
    else:
        # Underflow occurred
        # Simulate 64-bit wraparound (as unsigned would do)
        wrapped: int = (diff + (1 << 64)) & 0xFFFFFFFFFFFFFFFF

        # How many bits underflowed?
        # Use the absolute difference to estimate underflow “depth”
        borrow: int = abs(diff)
        overflow_bit_count: int = borrow.bit_length()

        # Rotate right the wrapped result by number of overflow bits
        rotated: int = (
            (wrapped >> overflow_bit_count)
            | (wrapped << (64 - overflow_bit_count))
        ) & 0xFFFFFFFFFFFFFFFF
        return rotated


class AppManager:
    def __init__(self, toml_data: str | None = None) -> None:
        self._input = None
        if toml_data is not None:
            self._config = load_toml(toml_data)
            self._args = None
            self._chdir()
        else:
            self._config = Config()
            self._args = get_args()
            if self._args.config_path is not None:
                self._config = load_toml_config(self._args.config_path)
            elif None not in (
                self._args.comfyui_path,
                self._args.ckpt_path,
                self._args.lora_path,
                self._args.controlnet_path,
                self._args.system_prompt_path,
                self._args.neg_prompts_path,
                self._args.sub_prompts_path,
                self._args.face_swap_image_path,
                self._args.pose_image_path,
            ):
                self._set_config_from_args()
            else:
                raise ValueError(
                    "All of the following must be provided: "
                    "--comfyui_path, --ckpt_path, --lora_path, "
                    "--controlnet_path, --system_prompt_path, --neg_prompts_path, "
                    "--sub_prompts_path, --face_swap_image_path, --pose_image_path"
                )

    def _set_config_from_args(self) -> None:
        if self._args is None:
            return
        self._set_comfyui_path()
        self._set_venv()
        self._set_ckpt()
        self._set_lora()
        self._set_controlnet()
        self.config.disable_controlnet = self._args.disable_controlnet
        self.config.steps = self._args.steps
        self.config.seed = self._args.seed
        self.config.guidance_scale = self._args.guidance_scale
        self.config.batch = self._args.batch
        self.config.lora_strength = self._args.lora_strength
        self._set_resolution()
        self.config.aspect_ratio = self._args.aspect_ratio
        self._set_system_prompt()
        self._set_neg_prompts()
        self._set_sub_prompts()
        self._set_face_swap_image()
        self._set_pose_image()
        self.config.loop_count = self._args.loop_count
        self.config.seed_generation = SeedGenerationMethod(
            self._args.seed_generation
        )
        self.config.output_path = pathlib.Path(self._args.output_path)

    def _set_comfyui_path(self) -> None:
        if self._args is None:
            return
        self.config.comfyui_path = pathlib.Path(
            self._args.comfyui_path
        ).expanduser()
        if not os.path.isdir(self.config.comfyui_path):
            raise ValueError(
                f"ComfyUI directory not found: {self.config.comfyui_path}"
            )
        self._input_path: pathlib.Path = self.config.comfyui_path / "input"

    def _set_venv(self) -> None:
        if self._args is None:
            return
        venv_path: pathlib.Path = (
            self._args.venv_path
            if self.config.venv_path is None
            else self.config.venv_path
        ).expanduser()
        venv_path = self.config.comfyui_path / self._args.venv_path
        if not os.path.isdir(venv_path):
            raise ValueError(f"Environment directory not found: {venv_path}")
        self.config.venv_path = venv_path

    def _set_ckpt(self) -> None:
        if self._args is None or self.config.comfyui_path is None:
            return
        ckpt_path: pathlib.Path = (
            self.config.comfyui_path
            / MODEL_DIRECTORY
            / CHECKPOINT_DIRECTORY
            / self._args.ckpt_path
        ).expanduser()
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint file not found: {ckpt_path}")
        self.config.ckpt = self._args.ckpt_path

    def _set_lora(self) -> None:
        if self._args is None or self.config.comfyui_path is None:
            return
        lora_path: pathlib.Path = (
            self.config.comfyui_path
            / MODEL_DIRECTORY
            / LORA_DIRECTORY
            / self._args.lora_path
        ).expanduser()
        if not os.path.isfile(lora_path):
            raise ValueError(f"Lora file not found: {lora_path}")
        self.config.lora = self._args.lora_path

    def _set_controlnet(self) -> None:
        if self._args is None or self.config.comfyui_path is None:
            return
        controlnet_path: pathlib.Path = (
            self.config.comfyui_path
            / MODEL_DIRECTORY
            / CONTROLNET_DIRECTORY
            / self._args.controlnet_path
        ).expanduser()
        if not os.path.isfile(controlnet_path):
            raise ValueError(f"Controlnet file not found: {controlnet_path}")
        self.config.controlnet = self._args.controlnet_path

    def _set_resolution(self) -> None:
        if self._args is None:
            return
        self.config.width = self._args.width
        self.config.height = int(
            self._args.width * ASPECT_RATIO[self._args.aspect_ratio]
        )

    def _set_system_prompt(self) -> None:
        if self._args is None:
            return
        with open(
            pathlib.Path(self._args.system_prompt_path).expanduser(), "r"
        ) as fd:
            self.config.system_prompt = fd.read().strip()

    def _set_neg_prompts(self) -> None:
        if self._args is None:
            return
        with open(
            pathlib.Path(self._args.neg_prompts_path).expanduser(), "r"
        ) as fd:
            self.config.neg_prompts = fd.read().splitlines()

    def _set_sub_prompts(self) -> None:
        if self._args is None:
            return
        with open(
            pathlib.Path(self._args.sub_prompts_path).expanduser(), "r"
        ) as fd:
            self.config.sub_prompts = fd.read().splitlines()

    def _set_face_swap_image(self) -> None:
        if self._args is None:
            return
        face_swap_image_path: pathlib.Path = pathlib.Path(
            self._args.face_swap_image_path
        ).expanduser()
        if not os.path.isfile(face_swap_image_path):
            raise ValueError(
                f"Face swap image file not found: {face_swap_image_path}"
            )
        shutil.copyfile(
            face_swap_image_path, self._input_path / face_swap_image_path.name
        )
        self.config.face_swap_image = face_swap_image_path.name

    def _set_pose_image(self) -> None:
        if self._args is None:
            return
        pose_image_path: pathlib.Path = pathlib.Path(
            self._args.pose_image_path
        ).expanduser()
        if not os.path.isfile(pose_image_path):
            raise ValueError(f"Pose image file not found: {pose_image_path}")
        shutil.copyfile(
            pose_image_path, self._input_path / pose_image_path.name
        )
        self.config.pose_image = pose_image_path.name

    def _chdir(self) -> None:
        if self.config.comfyui_path is None:
            return
        print(self.config.comfyui_path)
        os.chdir(self.config.comfyui_path)

    def generate_new_seed(self) -> int:
        match self.config.seed_generation:
            case SeedGenerationMethod.INCREMENT:
                return add_with_rotate_64(self.config.seed, 1)
            case SeedGenerationMethod.DECREMENT:
                return sub_with_rotate_64(self.config.seed, 1)
            case SeedGenerationMethod.RANDOM:
                return random.randint(1, 2**64)

    @property
    def config(self) -> Config:
        return self._config

    @property
    def basedir(self) -> pathlib.Path:
        return pathlib.Path(__file__).parent

    @property
    def pythonpath(self) -> pathlib.Path:
        return self.basedir.parent


def dump_toml(manager: AppManager) -> str:
    doc: TOMLDocument = document()
    doc_config: Table = table()
    doc_config.add("comfyui_path", str(manager.config.comfyui_path))
    doc_config.add("venv_path", str(manager.config.venv_path))
    doc_config.add("ckpt", manager.config.ckpt)
    doc_config.add("lora", manager.config.lora)
    doc_config.add("controlnet", manager.config.controlnet)
    doc_config.add("disable_controlnet", manager.config.disable_controlnet)
    doc_config.add("steps", manager.config.steps)
    doc_config.add("seed", manager.config.seed)
    doc_config.add("guidance_scale", manager.config.guidance_scale)
    doc_config.add("lora_strength", manager.config.lora_strength)
    doc_config.add("batch", manager.config.batch)
    doc_config.add("width", manager.config.width)
    doc_config.add("height", manager.config.height)
    doc_config.add("aspect_ratio", manager.config.aspect_ratio)
    doc_config.add("system_prompt", manager.config.system_prompt)
    doc_config.add("neg_prompts", manager.config.neg_prompts)
    doc_config.add("sub_prompts", manager.config.sub_prompts)
    doc_config.add("face_swap_image", manager.config.face_swap_image)
    doc_config.add("pose_image", manager.config.pose_image)
    doc_config.add("loop_count", manager.config.loop_count)
    doc_config.add("seed_generation", manager.config.seed_generation)
    doc_config.add("output_path", str(manager.config.output_path))
    doc.add("config", doc_config)
    return dumps(doc)


def load_toml(toml_data: str) -> Config:
    doc: TOMLDocument = loads(toml_data)
    doc_dict: dict[str, Any] = doc.value
    config: Config = Config()
    config.comfyui_path = pathlib.Path(
        doc_dict["config"]["comfyui_path"]
    ).expanduser()
    config.venv_path = pathlib.Path(
        doc_dict["config"]["venv_path"]
    ).expanduser()
    config.ckpt = doc_dict["config"]["ckpt"]
    config.lora = doc_dict["config"]["lora"]
    config.controlnet = doc_dict["config"]["controlnet"]
    config.disable_controlnet = doc_dict["config"]["disable_controlnet"]
    config.steps = doc_dict["config"]["steps"]
    config.seed = doc_dict["config"]["seed"]
    config.guidance_scale = doc_dict["config"]["guidance_scale"]
    config.batch = doc_dict["config"]["batch"]
    config.lora_strength = doc_dict["config"]["lora_strength"]
    config.width = doc_dict["config"]["width"]
    config.height = doc_dict["config"]["height"]
    config.aspect_ratio = doc_dict["config"]["aspect_ratio"]
    config.system_prompt = doc_dict["config"]["system_prompt"]
    config.neg_prompts = doc_dict["config"]["neg_prompts"]
    config.sub_prompts = doc_dict["config"]["sub_prompts"]
    config.face_swap_image = doc_dict["config"]["face_swap_image"]
    config.pose_image = doc_dict["config"]["pose_image"]
    config.loop_count = doc_dict["config"]["loop_count"]
    config.seed_generation = doc_dict["config"]["seed_generation"]
    config.output_path = pathlib.Path(doc_dict["config"]["output_path"])
    return config


def load_toml_config(config_path: pathlib.Path) -> Config:
    with open(config_path, "r") as f:
        doc: TOMLDocument = loads(f.read())
    doc_dict: dict[str, Any] = doc.value
    config: Config = Config()
    config.comfyui_path = pathlib.Path(
        doc_dict["config"]["comfyui_path"]
    ).expanduser()
    if not os.path.isdir(config.comfyui_path):
        raise ValueError(f"ComfyUI directory not found: {config.comfyui_path}")
    input_path: pathlib.Path = config.comfyui_path / "input"
    venv_path: str = doc_dict["config"].get("venv_path", DEFAULT_VENV_PATH)
    config.venv_path = config.comfyui_path / venv_path
    if not os.path.isdir(config.venv_path):
        raise ValueError(f"Environment directory not found: {venv_path}")
    ckpt: str = doc_dict["config"]["ckpt_path"]
    ckpt_path: pathlib.Path = (
        config.comfyui_path / MODEL_DIRECTORY / CHECKPOINT_DIRECTORY / ckpt
    ).expanduser()
    if not os.path.isfile(ckpt_path):
        raise ValueError(f"Checkpoint file not found: {ckpt_path}")
    config.ckpt = ckpt
    lora: str = doc_dict["config"]["lora_path"]
    lora_path: pathlib.Path = (
        config.comfyui_path / MODEL_DIRECTORY / LORA_DIRECTORY / lora
    ).expanduser()
    if not os.path.isfile(lora_path):
        raise ValueError(f"Lora file not found: {lora_path}")
    config.lora = lora
    controlnet: str = doc_dict["config"]["controlnet_path"]
    controlnet_path: pathlib.Path = (
        config.comfyui_path
        / MODEL_DIRECTORY
        / CONTROLNET_DIRECTORY
        / controlnet
    ).expanduser()
    if not os.path.isfile(controlnet_path):
        raise ValueError(f"Controlnet file not found: {controlnet_path}")
    config.controlnet = controlnet
    config.disable_controlnet = doc_dict["config"].get(
        "disable_controlnet", DEFAULT_DISABLE_CONTROLNET
    )
    config.steps = doc_dict["config"].get("steps", DEFAULT_STEPS)
    seed: int | None = doc_dict["config"].get("seed")
    if seed is None:
        seed = random.randint(1, 2**64)
    config.seed = seed
    config.guidance_scale = doc_dict["config"].get(
        "guidance_scale", DEFAULT_GUIDANCE_SCALE
    )
    config.batch = doc_dict["config"].get("batch", DEFAULT_BATCH)
    config.lora_strength = doc_dict["config"].get(
        "lora_strength", DEFAULT_LORA_STRENGTH
    )
    config.width = doc_dict["config"].get("width", DEFAULT_WIDTH)
    aspect_ratio: str = doc_dict["config"].get(
        "aspect_ratio", DEFAULT_ASPECT_RATIO
    )
    config.height = int(config.width * ASPECT_RATIO[aspect_ratio])
    config.system_prompt = doc_dict["config"]["system_prompt"]
    config.neg_prompts = doc_dict["config"]["neg_prompts"]
    config.sub_prompts = doc_dict["config"]["sub_prompts"]
    face_swap_image_path: pathlib.Path = pathlib.Path(
        doc_dict["config"]["face_swap_image_path"]
    ).expanduser()
    if not os.path.isfile(face_swap_image_path):
        raise ValueError(
            f"Face swap image file not found: {face_swap_image_path}"
        )
    shutil.copyfile(
        face_swap_image_path, input_path / face_swap_image_path.name
    )
    config.face_swap_image = face_swap_image_path.name
    pose_image_path: pathlib.Path = pathlib.Path(
        doc_dict["config"]["pose_image_path"]
    ).expanduser()
    if not os.path.isfile(pose_image_path):
        raise ValueError(f"Pose image file not found: {pose_image_path}")
    shutil.copyfile(pose_image_path, input_path / pose_image_path.name)
    config.pose_image = pose_image_path.name
    config.loop_count = doc_dict["config"].get(
        "loop_count", DEFAULT_LOOP_COUNT
    )
    config.seed_generation = SeedGenerationMethod(
        doc_dict["config"]["seed_generation"]
    )
    config.output_path = pathlib.Path(
        doc_dict["config"].get("output_path", "")
    )
    return config
