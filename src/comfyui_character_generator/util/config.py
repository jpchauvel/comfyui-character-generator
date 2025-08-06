import json
import os
import pathlib
import random
import shutil
import tomllib
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Self

from comfyui_character_generator.util.constants import (
    ASPECT_RATIO, CHECKPOINT_DIRECTORY, CONTROLNET_DIRECTORY,
    DEFAULT_ASPECT_RATIO, DEFAULT_BATCH, DEFAULT_DISABLE_CONTROLNET,
    DEFAULT_GUIDANCE_SCALE, DEFAULT_LOOP_COUNT, DEFAULT_POSE_DETECTION_TYPE,
    DEFAULT_SEED_GENERATION, DEFAULT_STEPS, DEFAULT_WIDTH, LORA_DIRECTORY,
    MODEL_DIRECTORY, SEED, UPSCALER_DIRECTORY)
from comfyui_character_generator.util.enums import (PoseDetectionType,
                                                    SeedGenerationMethod)


@dataclass
class SystemPromptMixin:
    system_prompt: str = ""
    system_neg_prompt: str = ""


@dataclass
class PromptMixin(SystemPromptMixin):
    sub_prompts: list[str] = field(default_factory=list)
    neg_prompts: list[str] = field(default_factory=list)

    def _set_prompts(self, values: list[list[str]]) -> None:
        if len(values[0]) != len(values[1]):
            raise ValueError(
                "Number of sub prompts and negative prompts must be the same"
            )
        self.sub_prompts = values[0]
        self.neg_prompts = values[1]

    @property
    def sub_prompt_count(self) -> int:
        return len(self.sub_prompts)


@dataclass
class ImageMixin:
    face_swap_images: list[str] = field(default_factory=list)
    pose_images: list[str] = field(default_factory=list)

    def _set_face_swap_images(
        self, input_path: pathlib.Path | None, value: list[str]
    ) -> None:
        if input_path is None:
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
                input_path / face_swap_image_path.name,
            )
            face_swap_images.append(face_swap_image_path.name)
        self.face_swap_images = face_swap_images

    def _set_pose_images(
        self, input_path: pathlib.Path | None, value: list[str]
    ) -> None:
        if input_path is None:
            raise ValueError("Input path is not set")
        pose_images: list[str] = []
        for v in value:
            pose_image_path: pathlib.Path = pathlib.Path(v).expanduser()
            if not os.path.isfile(pose_image_path):
                raise ValueError(
                    f"Pose image file not found: {pose_image_path}"
                )
            shutil.copyfile(pose_image_path, input_path / pose_image_path.name)
            pose_images.append(pose_image_path.name)
        self.pose_images = pose_images


@dataclass
class BaseConfig:
    ckpt: str | None = None
    loras: list[str] = field(default_factory=list)
    controlnet: str | None = None
    upscaler: str | None = None
    disable_controlnet: bool = DEFAULT_DISABLE_CONTROLNET
    pose_detection_type: PoseDetectionType = DEFAULT_POSE_DETECTION_TYPE
    steps: int = DEFAULT_STEPS
    seed: int = SEED
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    batch: int = DEFAULT_BATCH
    lora_strengths: list[float] = field(default_factory=list)
    width: int = DEFAULT_WIDTH
    height: int = int(DEFAULT_WIDTH * ASPECT_RATIO[DEFAULT_ASPECT_RATIO])
    aspect_ratio: str = DEFAULT_ASPECT_RATIO
    loop_count: int = DEFAULT_LOOP_COUNT
    seed_generation: SeedGenerationMethod = DEFAULT_SEED_GENERATION
    output_path: pathlib.Path = pathlib.Path("")

    def __init__(
        self,
        comfyui_path: pathlib.Path | None,
        input_path: pathlib.Path | None,
        validate: bool = True,
        **attrs,
    ) -> None:
        if validate:
            self._set_global_config(comfyui_path, **attrs)
        else:
            self._set_attrs(**attrs)

    def _set_global_config(
        self, comfyui_path: pathlib.Path | None, **attrs
    ) -> None:
        self._set_ckpt(comfyui_path, attrs["ckpt"])
        self._set_loras_and_strengths(
            comfyui_path, attrs["loras"], attrs["lora_strengths"]
        )
        self._set_controlnet(comfyui_path, attrs["controlnet"])
        self._set_upscaler(comfyui_path, attrs["upscaler"])
        self.disable_controlnet = attrs["disable_controlnet"]
        self.pose_detection_type = PoseDetectionType(
            attrs["pose_detection_type"]
        )
        self.steps = attrs["steps"]
        self.seed = attrs["seed"]
        self.guidance_scale = attrs["guidance_scale"]
        self.batch = attrs["batch"]
        self.aspect_ratio = attrs["aspect_ratio"]
        self._set_resolution(attrs["width"])
        self.loop_count = attrs["loop_count"]
        self.seed_generation = SeedGenerationMethod(attrs["seed_generation"])
        self.output_path = pathlib.Path(attrs["output_path"])

    def _set_attrs(self, **attrs) -> None:
        for key, value in attrs.items():
            match key:
                case "pose_detection_type":
                    self.pose_detection_type = PoseDetectionType(value)
                case "seed_generation":
                    self.seed_generation = SeedGenerationMethod(value)
                case _:
                    setattr(self, key, value)

    def _set_ckpt(self, comfyui_path: pathlib.Path | None, value: str) -> None:
        if comfyui_path is None:
            raise ValueError("ComfyUI path is not set")
        ckpt_path: pathlib.Path = (
            comfyui_path / MODEL_DIRECTORY / CHECKPOINT_DIRECTORY / value
        ).expanduser()
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint file not found: {ckpt_path}")
        self.ckpt = value

    def _set_loras_and_strengths(
        self,
        comfyui_path: pathlib.Path | None,
        loras: list[str],
        strengths: list[float],
    ) -> None:
        if comfyui_path is None:
            raise ValueError("ComfyUI path is not set")
        if len(loras) != len(strengths):
            raise ValueError(
                "Number of loras and lora strengths must be the same"
            )
        for lora in loras:
            lora_path: pathlib.Path = (
                comfyui_path / MODEL_DIRECTORY / LORA_DIRECTORY / lora
            ).expanduser()
            if not os.path.isfile(lora_path):
                raise ValueError(f"Lora file not found: {lora_path}")
        self.loras = loras
        self.lora_strengths = strengths

    def _set_controlnet(
        self, comfyui_path: pathlib.Path | None, value: str
    ) -> None:
        if comfyui_path is None:
            raise ValueError("ComfyUI path is not set")
        controlnet_path: pathlib.Path = (
            comfyui_path / MODEL_DIRECTORY / CONTROLNET_DIRECTORY / value
        ).expanduser()
        if not os.path.isfile(controlnet_path):
            raise ValueError(f"Controlnet file not found: {controlnet_path}")
        self.controlnet = value

    def _set_upscaler(
        self, comfyui_path: pathlib.Path | None, value: str
    ) -> None:
        if comfyui_path is None:
            raise ValueError("ComfyUI path is not set")
        upscaler_path: pathlib.Path = (
            comfyui_path / MODEL_DIRECTORY / UPSCALER_DIRECTORY / value
        ).expanduser()
        if not os.path.isfile(upscaler_path):
            raise ValueError(f"Upscaler file not found: {upscaler_path}")
        self.upscaler = value

    def _set_resolution(self, width: int) -> None:
        if self.aspect_ratio is None:
            raise ValueError("Aspect ratio is not set")
        self.width = width
        self.height = int(width * ASPECT_RATIO[self.aspect_ratio])


@dataclass
class Config(BaseConfig, PromptMixin, ImageMixin):
    def __init__(
        self,
        comfyui_path: pathlib.Path | None,
        input_path: pathlib.Path | None,
        validate: bool = True,
        **attrs,
    ) -> None:
        if validate:
            self._set_global_config(comfyui_path, **attrs)
            self.system_prompt = attrs["system_prompt"]
            self.system_neg_prompt = attrs["system_neg_prompt"]
            self._set_prompts(
                values=[attrs["sub_prompts"], attrs["neg_prompts"]]
            )
            self._set_face_swap_images(input_path, attrs["face_swap_images"])
            self._set_pose_images(input_path, attrs["pose_images"])
            self._validate_sub_prompt_length()
        else:
            self._set_attrs(**attrs)

    def _validate_sub_prompt_length(self) -> None:
        if len(self.sub_prompts) not in (
            len(self.face_swap_images),
            len(self.pose_images),
        ):

            raise ValueError(
                "Number of sub prompts, face swap images and pose images must be the same"
            )

    @classmethod
    def from_dict_list(cls, data: list[dict[str, Any]]) -> list[Self]:
        configs: list[Self] = []
        for ele in data:
            for key, value in ele.items():
                if key == "output_path":
                    ele[key] = pathlib.Path(value)
            configs.append(cls(None, None, validate=False, **ele))
        return configs


@dataclass
class GlobalConfig(BaseConfig, SystemPromptMixin):
    comfyui_path: pathlib.Path | None = None
    venv_path: pathlib.Path | None = None
    sub_configs: list[Config] = field(default_factory=list)

    def __init__(self, validate: bool = True, **attrs) -> None:
        self.sub_configs = []
        self._input_path: pathlib.Path | None = None
        if validate:
            self._set_comfyui_path(attrs["comfyui_path"])
            self._set_venv(attrs["venv_path"])
            self.system_prompt = attrs["system_prompt"]
            self.system_neg_prompt = attrs["system_neg_prompt"]
            attrs["comfyui_path"] = self.comfyui_path
            attrs["venv_path"] = self.venv_path
        else:
            for key, value in attrs.items():
                setattr(self, key, value)
        attrs["input_path"] = self._input_path
        super().__init__(
            validate=False,
            **attrs,
        )

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

    @staticmethod
    def default_dict_value(value: Any) -> Any:
        if isinstance(value, pathlib.Path):
            return value.as_posix()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        for key, value in data.items():
            if key in ("comfyui_path", "venv_path", "output_path"):
                data[key] = pathlib.Path(value)
        config = cls(validate=False, **data)
        config.sub_configs = Config.from_dict_list(data.get("sub_configs", []))
        return config

    def dump(self) -> str:
        return json.dumps(asdict(self), default=self.default_dict_value)

    @classmethod
    def load(cls, data: str) -> Self:
        return cls.from_dict(json.loads(data))

    @classmethod
    def load_from_toml(cls, config_path: pathlib.Path) -> Self:
        with open(config_path, "rb") as fd:
            doc_dict: dict[str, Any] = tomllib.load(fd)

        config_attrs: dict[str, Any] = dict(
            comfyui_path=pathlib.Path(doc_dict["global"]["comfyui_path"]),
            venv_path=pathlib.Path(doc_dict["global"]["venv_path"]),
            ckpt=doc_dict["global"]["ckpt_path"],
            loras=doc_dict["global"]["lora_paths"],
            lora_strengths=doc_dict["global"]["lora_strengths"],
            controlnet=doc_dict["global"]["controlnet_path"],
            upscaler=doc_dict["global"]["upscaler_path"],
            disable_controlnet=doc_dict["global"].get(
                "disable_controlnet", DEFAULT_DISABLE_CONTROLNET
            ),
            pose_detection_type=PoseDetectionType(
                doc_dict["global"].get(
                    "pose_detection_type", DEFAULT_POSE_DETECTION_TYPE
                )
            ),
            steps=doc_dict["global"].get("steps", DEFAULT_STEPS),
            seed=doc_dict["global"].get("seed", random.randint(1, 2**64)),
            guidance_scale=doc_dict["global"].get(
                "guidance_scale", DEFAULT_GUIDANCE_SCALE
            ),
            batch=doc_dict["global"].get("batch", DEFAULT_BATCH),
            width=doc_dict["global"].get("width", DEFAULT_WIDTH),
            aspect_ratio=doc_dict["global"].get(
                "aspect_ratio", DEFAULT_ASPECT_RATIO
            ),
            system_prompt=doc_dict["global"].get("system_prompt", ""),
            system_neg_prompt=doc_dict["global"].get("system_neg_prompt", ""),
            loop_count=doc_dict["global"].get(
                "loop_count", DEFAULT_LOOP_COUNT
            ),
            seed_generation=SeedGenerationMethod(
                doc_dict["global"].get(
                    "seed_generation", DEFAULT_SEED_GENERATION
                )
            ),
            output_path=pathlib.Path(
                doc_dict["global"].get("output_path", "")
            ),
        )
        config = cls(**config_attrs)
        sub_doc_dict_keys: list[str] = [
            k for k in doc_dict.keys() if k.startswith("config")
        ]
        for key in sub_doc_dict_keys:
            sub_config_attrs: dict[str, Any] = deepcopy(config_attrs)
            sub_config_attrs.update(
                dict(
                    ckpt=doc_dict[key].get("ckpt_path", config.ckpt),
                    loras=doc_dict[key].get("lora_paths", config.loras),
                    lora_strengths=doc_dict[key].get(
                        "lora_strengths", config.lora_strengths
                    ),
                    controlnet=doc_dict[key].get(
                        "controlnet_path", config.controlnet
                    ),
                    upscaler=doc_dict[key].get(
                        "upscaler_path", config.upscaler
                    ),
                    disable_controlnet=doc_dict[key].get(
                        "disable_controlnet", config.disable_controlnet
                    ),
                    pose_detection_type=PoseDetectionType(
                        doc_dict[key].get(
                            "pose_detection_type", config.pose_detection_type
                        )
                    ),
                    steps=doc_dict[key].get("steps", config.steps),
                    seed=doc_dict[key].get("seed", config.seed),
                    guidance_scale=doc_dict[key].get(
                        "guidance_scale", config.guidance_scale
                    ),
                    batch=doc_dict[key].get("batch", config.batch),
                    width=doc_dict[key].get("width", config.width),
                    aspect_ratio=doc_dict[key].get(
                        "aspect_ratio", config.aspect_ratio
                    ),
                    system_prompt=doc_dict[key].get(
                        "system_prompt", config.system_prompt
                    ),
                    system_neg_prompt=doc_dict[key].get(
                        "system_neg_prompt", config.system_neg_prompt
                    ),
                    neg_prompts=doc_dict[key]["neg_prompts"],
                    sub_prompts=doc_dict[key]["sub_prompts"],
                    face_swap_images=doc_dict[key]["face_swap_image_paths"],
                    pose_images=doc_dict[key]["pose_image_paths"],
                    loop_count=doc_dict[key].get(
                        "loop_count", config.loop_count
                    ),
                    seed_generation=SeedGenerationMethod(
                        doc_dict[key].get(
                            "seed_generation", config.seed_generation
                        )
                    ),
                    output_path=pathlib.Path(
                        doc_dict[key].get("output_path", config.output_path)
                    ),
                )
            )
            config.sub_configs.append(
                Config(input_path=config.input_path, **sub_config_attrs)
            )

        return config

    @property
    def input_path(self) -> pathlib.Path | None:
        return self._input_path
