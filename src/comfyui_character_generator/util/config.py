import json
import os
import pathlib
import random
import shutil
import tomllib
from dataclasses import asdict, dataclass, field
from typing import Any, Self

from comfyui_character_generator.util.constants import (
    ASPECT_RATIO, CHECKPOINT_DIRECTORY, CONTROLNET_DIRECTORY,
    DEFAULT_ASPECT_RATIO, DEFAULT_BATCH, DEFAULT_DISABLE_CONTROLNET,
    DEFAULT_GUIDANCE_SCALE, DEFAULT_LOOP_COUNT, DEFAULT_SEED_GENERATION,
    DEFAULT_STEPS, DEFAULT_WIDTH, LORA_DIRECTORY, MODEL_DIRECTORY, SEED)
from comfyui_character_generator.util.enums import SeedGenerationMethod


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
    system_neg_prompt: str = ""
    neg_prompts: list[str] = field(default_factory=list)
    sub_prompts: list[str] = field(default_factory=list)
    face_swap_images: list[str] = field(default_factory=list)
    pose_images: list[str] = field(default_factory=list)
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
            self.aspect_ratio = attrs["aspect_ratio"]
            self._set_resolution(attrs["width"])
            self.system_prompt = attrs["system_prompt"]
            self.system_neg_prompt = attrs["system_neg_prompt"]
            self._set_prompts(
                values=[attrs["sub_prompts"], attrs["neg_prompts"]]
            )
            self.neg_prompts = attrs["neg_prompts"]
            self.sub_prompts = attrs["sub_prompts"]
            self._set_face_swap_images(attrs["face_swap_images"])
            self._set_pose_images(attrs["pose_images"])
            self._validate_sub_prompt_length()
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

    def _set_prompts(self, values: list[list[str]]) -> None:
        if len(values[0]) != len(values[1]):
            raise ValueError(
                "Number of sub prompts and negative prompts must be the same"
            )
        self.sub_prompts = values[0]
        self.neg_prompts = values[1]

    def _set_face_swap_images(self, value: list[str]) -> None:
        if self._input_path is None:
            raise ValueError("Input path is not set")
        face_swap_images: list[str] = []
        for v in value:
            face_swap_image_path: pathlib.Path = pathlib.Path(v).expanduser()
            if not os.path.isfile(face_swap_image_path):
                raise ValueError(
                    f"Face swap image file not found: {face_swap_image_path}"
                )
            shutil.copyfile(
                face_swap_image_path,
                self._input_path / face_swap_image_path.name,
            )
            face_swap_images.append(face_swap_image_path.name)
        self.face_swap_images = face_swap_images

    def _set_pose_images(self, value: list[str]) -> None:
        if self._input_path is None:
            raise ValueError("Input path is not set")
        pose_images: list[str] = []
        for v in value:
            pose_image_path: pathlib.Path = pathlib.Path(v).expanduser()
            if not os.path.isfile(pose_image_path):
                raise ValueError(
                    f"Pose image file not found: {pose_image_path}"
                )
            shutil.copyfile(
                pose_image_path, self._input_path / pose_image_path.name
            )
            pose_images.append(pose_image_path.name)
        self.pose_images = pose_images

    def _validate_sub_prompt_length(self) -> None:
        if len(self.sub_prompts) not in (
            len(self.face_swap_images),
            len(self.pose_images),
        ):

            raise ValueError(
                "Number of sub prompts, face swap images and pose images must be the same"
            )

    @property
    def sub_prompt_count(self) -> int:
        return len(self.sub_prompts)

    @property
    def pose_and_face_swap_count(self) -> int:
        return len(self.pose_images)

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
            system_neg_prompt=doc_dict["config"]["system_neg_prompt"],
            neg_prompts=doc_dict["config"]["neg_prompts"],
            sub_prompts=doc_dict["config"]["sub_prompts"],
            face_swap_images=doc_dict["config"]["face_swap_image_paths"],
            pose_images=doc_dict["config"]["pose_image_paths"],
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
