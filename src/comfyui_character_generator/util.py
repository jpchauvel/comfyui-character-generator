import argparse
import os
import pathlib
import random
import shutil
import sys
from dataclasses import dataclass
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

MODEL_DIRECTORY: str = "models"
CHECKPOINT_DIRECTORY: str = "checkpoints"
LORA_DIRECTORY: str = "loras"
CONTROLNET_DIRECTORY: str = "controlnet"


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Automated ComfyUI Character Generator.",
    )
    parser.add_argument(
        "--comfyui_path",
        type=str,
        required=True,
        help="Path to ComfyUI directory.",
    )
    parser.add_argument(
        "--venv_path",
        type=str,
        default=".venv",
        help="Relative path to env directory. (default .venv)",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Single .safetensors checkpoint file. Relative to checkpoint directory.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to character LoRA .safetensors file. Relative to lora directory.",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        required=True,
        help="Path to ControlNet .safetensors file. Relative to controlnet directory.",
    )
    parser.add_argument(
        "--disable_controlnet",
        action="store_true",
        help="Disable ControlNet.",
    )
    parser.add_argument(
        "--steps", type=int, default=35, help="Denoising steps (default 35)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(1, 2**64),
        help="Base RNG seed (index is added per prompt).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=8.0,
        help="Guidance scale (default 8.0).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Images *per* prompt.",
    )
    parser.add_argument(
        "--lora_strength",
        type=float,
        default=1.5,
        help="Character LoRA strength.",
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Image width (default 1024)."
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="1:1",
        choices=ASPECT_RATIO.keys(),
        help="Image aspect ratio (default 1:1).",
    )
    parser.add_argument(
        "--system_prompt_path",
        required=True,
        help="Path to system prompt text file.",
    )
    parser.add_argument(
        "--neg_prompts_path",
        required=True,
        help="Path to negative prompts text file.",
    )
    parser.add_argument(
        "--sub_prompts_path",
        required=True,
        help="Path to sub-prompts text file. Prompts are separeted by newlines.",
    )
    parser.add_argument(
        "--face_swap_image_path",
        required=True,
        help="Path to face swap image file.",
    )
    parser.add_argument(
        "--pose_image_path",
        required=True,
        help="Path to pose image file.",
    )
    parser.add_argument(
        "--loop_count",
        type=int,
        default=1,
        help="Number of generations (default 1).",
    )
    return parser.parse_known_args(sys.argv)[0]


@dataclass()
class Config:
    comfyui_path: pathlib.Path
    venv_path: pathlib.Path
    ckpt: str
    lora: str
    controlnet: str
    disable_controlnet: bool
    steps: int
    seed: int
    guidance_scale: float
    batch: int
    lora_strength: float
    width: int
    height: int
    system_prompt: str
    neg_prompts: list[str]
    sub_prompts: list[str]
    face_swap_image: str
    pose_image: str
    loop_count: int

    def __init__(self):
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


class AppManager:
    def __init__(self, toml_data: str | None = None) -> None:
        self._is_config = toml_data is not None
        if toml_data is not None:
            self._config = load_toml(toml_data)
            self._set_properties_from_config()
            self._chdir()
        else:
            self._args = get_args()
            self._set_properties_from_args()

    def _set_properties_from_config(self) -> None:
        self._comfyui_path = self._config.comfyui_path
        self._set_venv()
        self._ckpt = self._config.ckpt
        self._lora = self._config.lora
        self._controlnet = self._config.controlnet
        self._disable_controlnet = self._config.disable_controlnet
        self._steps = self._config.steps
        self._seed = self._config.seed
        self._guidance_scale = self._config.guidance_scale
        self._batch = self._config.batch
        self._lora_strength = self._config.lora_strength
        self._width = self._config.width
        self._height = self._config.height
        self._system_prompt = self._config.system_prompt
        self._neg_prompts = self._config.neg_prompts
        self._sub_prompts = self._config.sub_prompts
        self._face_swap_image = self._config.face_swap_image
        self._pose_image = self._config.pose_image
        self._loop_count = self._config.loop_count

    def _set_properties_from_args(self) -> None:
        self._set_comfyui_path()
        self._set_venv()
        self._set_ckpt()
        self._set_lora()
        self._set_controlnet()
        self._disable_controlnet = self._args.disable_controlnet
        self._steps = self._args.steps
        self._seed = self._args.seed
        self._guidance_scale = self._args.guidance_scale
        self._batch = self._args.batch
        self._lora_strength = self._args.lora_strength
        self._set_resolution()
        self._set_system_prompt()
        self._set_neg_prompts()
        self._set_sub_prompts()
        self._set_face_swap_image()
        self._set_pose_image()
        self._loop_count = self._args.loop_count
        self._config = Config()
        self.config.comfyui_path = self._comfyui_path
        self.config.venv_path = self._venv_path
        self.config.ckpt = self._ckpt
        self.config.lora = self._lora
        self.config.controlnet = self._controlnet
        self.config.disable_controlnet = self._disable_controlnet
        self.config.steps = self._steps
        self.config.seed = self._seed
        self.config.guidance_scale = self._guidance_scale
        self.config.batch = self._batch
        self.config.lora_strength = self._lora_strength
        self.config.width = self._width
        self.config.height = self._height
        self.config.system_prompt = self._system_prompt
        self.config.neg_prompts = self._neg_prompts
        self.config.sub_prompts = self._sub_prompts
        self.config.face_swap_image = self._face_swap_image
        self.config.pose_image = self._pose_image
        self.config.loop_count = self._loop_count

    def _set_comfyui_path(self) -> None:
        self._comfyui_path = pathlib.Path(self._args.comfyui_path).expanduser()
        if not os.path.isdir(self._comfyui_path):
            raise ValueError(
                f"ComfyUI directory not found: {self._comfyui_path}"
            )
        self._input_path: pathlib.Path = self._comfyui_path / "input"

    def _set_venv(self) -> None:
        if self._is_config:
            self._venv_path = self._config.venv_path
        else:
            venv_path = self._comfyui_path / self._args.venv_path
            if not os.path.isdir(venv_path):
                raise ValueError(
                    f"Environment directory not found: {venv_path}"
                )
            self._venv_path = venv_path

    def _set_ckpt(self) -> None:
        ckpt_path: pathlib.Path = (
            self._comfyui_path
            / MODEL_DIRECTORY
            / CHECKPOINT_DIRECTORY
            / self._args.ckpt_path
        )
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint file not found: {ckpt_path}")
        self._ckpt = self._args.ckpt_path

    def _set_lora(self) -> None:
        lora_path: pathlib.Path = (
            self._comfyui_path
            / MODEL_DIRECTORY
            / LORA_DIRECTORY
            / self._args.lora_path
        )
        if not os.path.isfile(lora_path):
            raise ValueError(f"Lora file not found: {lora_path}")
        self._lora = self._args.lora_path

    def _set_controlnet(self) -> None:
        controlnet_path: pathlib.Path = (
            self._comfyui_path
            / MODEL_DIRECTORY
            / CONTROLNET_DIRECTORY
            / self._args.controlnet_path
        )
        if not os.path.isfile(controlnet_path):
            raise ValueError(f"Controlnet file not found: {controlnet_path}")
        self._controlnet = self._args.controlnet_path

    def _set_resolution(self) -> None:
        self._width = self._args.width
        self._height = int(
            self._args.width * ASPECT_RATIO[self._args.aspect_ratio]
        )

    def _set_system_prompt(self) -> None:
        with open(
            pathlib.Path(self._args.system_prompt_path).expanduser(), "r"
        ) as fd:
            self._system_prompt = fd.read().strip()

    def _set_neg_prompts(self) -> None:
        with open(
            pathlib.Path(self._args.neg_prompts_path).expanduser(), "r"
        ) as fd:
            self._neg_prompts = fd.read().splitlines()

    def _set_sub_prompts(self) -> None:
        with open(
            pathlib.Path(self._args.sub_prompts_path).expanduser(), "r"
        ) as fd:
            self._sub_prompts = fd.read().splitlines()

    def _set_face_swap_image(self) -> None:
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
        self._face_swap_image = face_swap_image_path.name

    def _set_pose_image(self) -> None:
        pose_image_path: pathlib.Path = pathlib.Path(
            self._args.pose_image_path
        ).expanduser()
        if not os.path.isfile(pose_image_path):
            raise ValueError(f"Pose image file not found: {pose_image_path}")
        shutil.copyfile(
            pose_image_path, self._input_path / pose_image_path.name
        )
        self._pose_image = pose_image_path.name

    def _chdir(self) -> None:
        print(self._comfyui_path)
        os.chdir(self._comfyui_path)

    @property
    def config(self) -> Config:
        return self._config


def dump_toml(manager: AppManager) -> str:
    doc: TOMLDocument = document()
    doc_manager: Table = table()
    doc_manager.add("comfyui_path", str(manager.config.comfyui_path))
    doc_manager.add("venv_path", str(manager.config.venv_path))
    doc_manager.add("ckpt", manager.config.ckpt)
    doc_manager.add("lora", manager.config.lora)
    doc_manager.add("controlnet", manager.config.controlnet)
    doc_manager.add("disable_controlnet", manager.config.disable_controlnet)
    doc_manager.add("steps", manager.config.steps)
    doc_manager.add("seed", manager.config.seed)
    doc_manager.add("guidance_scale", manager.config.guidance_scale)
    doc_manager.add("lora_strength", manager.config.lora_strength)
    doc_manager.add("batch", manager.config.batch)
    doc_manager.add("width", manager.config.width)
    doc_manager.add("height", manager.config.height)
    doc_manager.add("system_prompt", manager.config.system_prompt)
    doc_manager.add("neg_prompt", manager.config.neg_prompts)
    doc_manager.add("sub_prompt", manager.config.sub_prompts)
    doc_manager.add("face_swap_image", manager.config.face_swap_image)
    doc_manager.add("pose_image", manager.config.pose_image)
    doc_manager.add("loop_count", manager.config.loop_count)
    doc.add("manager", doc_manager)
    return dumps(doc)


def load_toml(toml_data: str) -> Config:
    doc: TOMLDocument = loads(toml_data)
    doc_dict: dict[str, Any] = doc.value
    config: Config = Config()
    config.comfyui_path = pathlib.Path(doc_dict["manager"]["comfyui_path"])
    config.venv_path = pathlib.Path(doc_dict["manager"]["venv_path"])
    config.ckpt = doc_dict["manager"]["ckpt"]
    config.lora = doc_dict["manager"]["lora"]
    config.controlnet = doc_dict["manager"]["controlnet"]
    config.disable_controlnet = doc_dict["manager"]["disable_controlnet"]
    config.steps = doc_dict["manager"]["steps"]
    config.seed = doc_dict["manager"]["seed"]
    config.guidance_scale = doc_dict["manager"]["guidance_scale"]
    config.batch = doc_dict["manager"]["batch"]
    config.lora_strength = doc_dict["manager"]["lora_strength"]
    config.width = doc_dict["manager"]["width"]
    config.height = doc_dict["manager"]["height"]
    config.system_prompt = doc_dict["manager"]["system_prompt"]
    config.neg_prompts = doc_dict["manager"]["neg_prompt"]
    config.sub_prompts = doc_dict["manager"]["sub_prompt"]
    config.face_swap_image = doc_dict["manager"]["face_swap_image"]
    config.pose_image = doc_dict["manager"]["pose_image"]
    config.loop_count = doc_dict["manager"]["loop_count"]
    return config
