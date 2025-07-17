import argparse
import json
import os
import pathlib
import random
import shutil
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from enum import IntEnum, auto
from typing import Any, Self

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
        "--lora_paths",
        nargs="*",
        type=str,
        default=[],
        help="Paths to .safetensors files. Relative to lora directory.",
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


@dataclass
class Config:
    comfyui_path: pathlib.Path | None = None
    venv_path: pathlib.Path | None = None
    ckpt: str | None = None
    loras: list[str] = field(default_factory=list)
    controlnet: str | None = None
    disable_controlnet: bool = DEFAULT_DISABLE_CONTROLNET
    steps: int = DEFAULT_STEPS
    seed: int = SEED
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    batch: int = DEFAULT_BATCH
    lora_strengths: list[float] = field(default_factory=list)
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

    def __init__(self, validate: bool = True, **attrs) -> None:
        self._neg_prompt_iter = None
        self._sub_prompt_iter = None
        self._input_path: pathlib.Path | None = None
        if validate:
            self._set_comfyui_path(attrs["comfyui_path"])
            self._set_venv(attrs["venv_path"])
            self._set_ckpt(attrs["ckpt"])
            self._set_loras_and_strengths(
                attrs["loras"], attrs["lora_strengths"]
            )
            self._set_controlnet(attrs["controlnet"])
            self.disable_controlnet = attrs["disable_controlnet"]
            self.steps = attrs["steps"]
            self.seed = attrs["seed"]
            self.guidance_scale = attrs["guidance_scale"]
            self.batch = attrs["batch"]
            self._set_resolution(attrs["width"])
            self.aspect_ratio = attrs["aspect_ratio"]
            self.system_prompt = attrs["system_prompt"]
            self.neg_prompts = attrs["neg_prompts"]
            self.sub_prompts = attrs["sub_prompts"]
            self._set_face_swap_image(attrs["face_swap_image"])
            self._set_pose_image(attrs["pose_image"])
            self.loop_count = attrs["loop_count"]
            self.seed_generation = SeedGenerationMethod(
                attrs["seed_generation"]
            )
            self.output_path = pathlib.Path(attrs["output_path"])
        else:
            for key, value in attrs.items():
                setattr(self, key, value)

    def _set_comfyui_path(self, value: str) -> None:
        comfyui_path: pathlib.Path = pathlib.Path(value).expanduser()
        if not os.path.isdir(comfyui_path):
            raise ValueError(f"ComfyUI directory not found: {comfyui_path}")
        self.comfyui_path = comfyui_path
        self._input_path = comfyui_path / "input"

    def _set_venv(self, value: str) -> None:
        if self.comfyui_path is None:
            raise ValueError("ComfyUI path is not set")
        venv_path = self.comfyui_path / value
        if not os.path.isdir(venv_path):
            raise ValueError(f"Environment directory not found: {venv_path}")
        self.venv_path = venv_path

    def _set_ckpt(self, value: str) -> None:
        if self.comfyui_path is None:
            raise ValueError("ComfyUI path is not set")
        ckpt_path: pathlib.Path = (
            self.comfyui_path / MODEL_DIRECTORY / CHECKPOINT_DIRECTORY / value
        ).expanduser()
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint file not found: {ckpt_path}")
        self.ckpt = value

    def _set_loras_and_strengths(
        self, loras: list[str], strengths: list[float]
    ) -> None:
        if self.comfyui_path is None:
            raise ValueError("ComfyUI path is not set")
        if len(loras) != len(strengths):
            raise ValueError(
                "Number of loras and lora strengths must be the same"
            )
        for lora in loras:
            lora_path: pathlib.Path = (
                self.comfyui_path / MODEL_DIRECTORY / LORA_DIRECTORY / lora
            ).expanduser()
            if not os.path.isfile(lora_path):
                raise ValueError(f"Lora file not found: {lora_path}")
        self.loras = loras
        self.lora_strengths = strengths

    def _set_controlnet(self, value: str) -> None:
        if self.comfyui_path is None:
            raise ValueError("ComfyUI path is not set")
        controlnet_path: pathlib.Path = (
            self.comfyui_path / MODEL_DIRECTORY / CONTROLNET_DIRECTORY / value
        ).expanduser()
        if not os.path.isfile(controlnet_path):
            raise ValueError(f"Controlnet file not found: {controlnet_path}")
        self.controlnet = value

    def _set_resolution(self, width: int) -> None:
        if self.aspect_ratio is None:
            raise ValueError("Aspect ratio is not set")
        self.width = width
        self.height = int(width * ASPECT_RATIO[self.aspect_ratio])

    def _set_face_swap_image(self, value: str) -> None:
        if self._input_path is None:
            raise ValueError("Input path is not set")
        face_swap_image_path: pathlib.Path = pathlib.Path(value).expanduser()
        if not os.path.isfile(face_swap_image_path):
            raise ValueError(
                f"Face swap image file not found: {face_swap_image_path}"
            )
        shutil.copyfile(
            face_swap_image_path, self._input_path / face_swap_image_path.name
        )
        self.face_swap_image = face_swap_image_path.name

    def _set_pose_image(self, value: str) -> None:
        if self._input_path is None:
            raise ValueError("Input path is not set")
        pose_image_path: pathlib.Path = pathlib.Path(value).expanduser()
        if not os.path.isfile(pose_image_path):
            raise ValueError(f"Pose image file not found: {pose_image_path}")
        shutil.copyfile(
            pose_image_path, self._input_path / pose_image_path.name
        )
        self.pose_image = pose_image_path.name

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

    @property
    def next_lora_and_stregnth(self) -> tuple[str, float]:
        return next(zip(self.loras, self.lora_strengths), ("", 0.0))

    @staticmethod
    def dict_factory(pairs: dict[str, Any] | Any) -> dict[str, Any] | Any:
        if isinstance(pairs, pathlib.Path):
            return pairs.as_posix()
        converted: dict[str, Any] = {}
        for key, value in pairs.items():
            if isinstance(value, pathlib.Path):
                converted[key] = value.as_posix()
            else:
                converted[key] = value
        return converted

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        for key, value in data.items():
            if key in ("comfyui_path", "venv_path", "output_path"):
                data[key] = pathlib.Path(value)
        return cls(validate=False, **data)

    def dump(self) -> str:
        return json.dumps(asdict(self), default=self.dict_factory)

    @classmethod
    def load(cls, data: str) -> Self:
        return cls.from_dict(json.loads(data))

    @classmethod
    def load_from_toml(cls, config_path: pathlib.Path) -> Self:
        with open(config_path, "rb") as fd:
            doc_dict: dict[str, Any] = tomllib.load(fd)

        config = cls(
            comfyui_path=pathlib.Path(doc_dict["config"]["comfyui_path"]),
            venv_path=pathlib.Path(doc_dict["config"]["venv_path"]),
            ckpt=doc_dict["config"]["ckpt_path"],
            loras=doc_dict["config"]["lora_paths"],
            lora_strengths=doc_dict["config"]["lora_strengths"],
            controlnet=doc_dict["config"]["controlnet_path"],
            disable_controlnet=doc_dict["config"].get(
                "disable_controlnet", DEFAULT_DISABLE_CONTROLNET
            ),
            steps=doc_dict["config"].get("steps", DEFAULT_STEPS),
            seed=doc_dict["config"].get("seed", random.randint(1, 2**64)),
            guidance_scale=doc_dict["config"].get(
                "guidance_scale", DEFAULT_GUIDANCE_SCALE
            ),
            batch=doc_dict["config"].get("batch", DEFAULT_BATCH),
            width=doc_dict["config"].get("width", DEFAULT_WIDTH),
            aspect_ratio=doc_dict["config"].get(
                "aspect_ratio", DEFAULT_ASPECT_RATIO
            ),
            system_prompt=doc_dict["config"]["system_prompt"],
            neg_prompts=doc_dict["config"]["neg_prompts"],
            sub_prompts=doc_dict["config"]["sub_prompts"],
            face_swap_image=doc_dict["config"]["face_swap_image_path"],
            pose_image=doc_dict["config"]["pose_image_path"],
            loop_count=doc_dict["config"].get(
                "loop_count", DEFAULT_LOOP_COUNT
            ),
            seed_generation=SeedGenerationMethod(
                doc_dict["config"].get(
                    "seed_generation", DEFAULT_SEED_GENERATION
                )
            ),
            output_path=pathlib.Path(
                doc_dict["config"].get("output_path", "")
            ),
        )
        return config


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
    def __init__(self, data: str | None = None) -> None:
        if data is not None:
            self._config = Config.load(data)
            self._args = None
            self._chdir()
        else:
            self._args = get_args()
            if self._args.config_path is not None:
                self._config = Config.load_from_toml(self._args.config_path)
            elif None not in (
                self._args.comfyui_path,
                self._args.ckpt_path,
                self._args.lora_paths,
                self._args.controlnet_path,
                self._args.system_prompt_path,
                self._args.neg_prompts_path,
                self._args.sub_prompts_path,
                self._args.face_swap_image_path,
                self._args.pose_image_path,
                self._args.lora_strengths,
            ):
                self._set_config_from_args()
            else:
                raise ValueError(
                    "All of the following must be provided: "
                    "--comfyui_path, --ckpt_path, --lora_paths, "
                    "--controlnet_path, --system_prompt_path, --neg_prompts_path, "
                    "--sub_prompts_path, --face_swap_image_path, --pose_image_path, "
                    "--lora_strengths"
                )

    def _set_config_from_args(self) -> None:
        if self._args is None:
            return
        self._config = Config(
            comfyui_path=self._args.comfyui_path,
            venv_path=self._args.venv_path,
            ckpt=self._args.ckpt_path,
            loras=self._args.lora_paths,
            lora_strengths=self._args.lora_strengths,
            controlnet_path=self._args.controlnet_path,
            disable_controlnet=self._args.disable_controlnet,
            steps=self._args.steps,
            seed=self._args.seed,
            guidance_scale=self._args.guidance_scale,
            batch=self._args.batch,
            width=self._args.width,
            system_prompt=self._get_system_prompt(
                self._args.system_prompt_path
            ),
            neg_prompts=self._get_prompts(self._args.neg_prompts_path),
            sub_prompts=self._get_prompts(self._args.sub_prompts_path),
            face_swap_image=self._args.face_swap_image_path,
            pose_image=self._args.pose_image_path,
            loop_count=self._args.loop_count,
            seed_generation=SeedGenerationMethod(self._args.seed_generation),
            output_path=self._args.output_path,
        )

    def _get_system_prompt(self, system_prompt_path: str) -> str:
        with open(pathlib.Path(system_prompt_path).expanduser(), "r") as fd:
            return fd.read().strip()

    def _get_prompts(self, prompts_path: str) -> list[str]:
        with open(pathlib.Path(prompts_path).expanduser(), "r") as fd:
            return fd.read().splitlines()

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
