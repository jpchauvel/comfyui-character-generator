import os
import pathlib
import random

from comfyui_character_generator.util.args import get_args
from comfyui_character_generator.util.config import Config, GlobalConfig
from comfyui_character_generator.util.enums import SeedGenerationMethod
from comfyui_character_generator.util.operations import (add_with_rotate_64,
                                                         sub_with_rotate_64)


class AppManager:
    def __init__(self, data: str | None = None) -> None:
        self._should_install_nodes = False
        if data is not None:
            self.config = GlobalConfig.load(data)
            self._chdir()
        else:
            self._args = get_args()
            if self._args.install_nodes:
                if None in (
                    self._args.comfyui_path,
                    self._args.venv_path,
                ):
                    raise ValueError(
                        "Both --comfyui_path and --venv_path must be provided"
                    )
                self.config = GlobalConfig(
                    comfyui_path=self._args.comfyui_path,
                    venv_path=self._args.venv_path,
                    system_prompt="",
                    system_neg_prompt="",
                )
                self._should_install_nodes = True
            elif self._args.config_path is not None:
                self.config = GlobalConfig.load_from_toml(
                    self._args.config_path
                )
            elif None not in (
                self._args.comfyui_path,
                self._args.ckpt_path,
                self._args.lora_paths,
                self._args.controlnet_path,
                self._args.upscaler_path,
                self._args.system_prompt_path,
                self._args.system_neg_prompt_path,
                self._args.face_swap_image_paths,
                self._args.pose_image_paths,
                self._args.lora_strengths,
            ):
                self._set_config_from_args()
            else:
                raise ValueError(
                    "All of the following must be provided: "
                    "--comfyui_path, --ckpt_path, --lora_paths, "
                    "--controlnet_path, --upscaler_path, "
                    "--system_prompt_path, --system_neg_prompt_path, "
                    "--face_swap_image_paths, --pose_image_paths, "
                    "--lora_strengths"
                )

    def _set_config_from_args(self) -> None:
        if self._args is None:
            return
        self.config = GlobalConfig(
            comfyui_path=self._args.comfyui_path,
            venv_path=self._args.venv_path,
        )
        self.config.sub_configs.append(
            Config(
                comfyui_path=self.config.comfyui_path,
                input_path=self.config.input_path,
                ckpt=self._args.ckpt_path,
                loras=self._args.lora_paths,
                lora_strengths=self._args.lora_strengths,
                controlnet=self._args.controlnet_path,
                upscaler=self._args.upscaler_path,
                disable_controlnet=self._args.disable_controlnet,
                pose_detection_type=self._args.pose_detection_type,
                steps=self._args.steps,
                seed=self._args.seed,
                guidance_scale=self._args.guidance_scale,
                batch=self._args.batch,
                width=self._args.width,
                aspect_ratio=self._args.aspect_ratio,
                system_prompt=self._get_system_prompt(
                    self._args.system_prompt_path
                ),
                system_neg_prompt=self._get_system_prompt(
                    self._args.system_neg_prompt_path
                ),
                neg_prompts=self._get_prompts(self._args.neg_prompts_path),
                sub_prompts=self._get_prompts(self._args.sub_prompts_path),
                face_swap_images=self._args.face_swap_image_paths,
                pose_images=self._args.pose_image_paths,
                loop_count=self._args.loop_count,
                seed_generation=SeedGenerationMethod(
                    self._args.seed_generation
                ),
                output_path=self._args.output_path,
            )
        )

    def _get_system_prompt(self, system_prompt_path: str) -> str:
        with open(pathlib.Path(system_prompt_path).expanduser(), "r") as fd:
            return fd.read().strip()

    def _get_prompts(self, prompts_path: str | None) -> list[str]:
        if prompts_path is None:
            return []
        with open(pathlib.Path(prompts_path).expanduser(), "r") as fd:
            return fd.read().splitlines()

    def _chdir(self) -> None:
        if self.config.comfyui_path is None:
            return
        os.chdir(self.config.comfyui_path)

    @staticmethod
    def generate_new_seed(
        seed_generation: SeedGenerationMethod, seed: int
    ) -> int:
        match seed_generation:
            case SeedGenerationMethod.INCREMENT:
                return add_with_rotate_64(seed, 1)
            case SeedGenerationMethod.DECREMENT:
                return sub_with_rotate_64(seed, 1)
            case SeedGenerationMethod.RANDOM:
                return random.randint(1, 2**64)

    @property
    def should_install_nodes(self) -> bool:
        return self._should_install_nodes

    @property
    def config(self) -> GlobalConfig:
        return self._config

    @config.setter
    def config(self, value: GlobalConfig) -> None:
        self._config = value

    @property
    def basedir(self) -> pathlib.Path:
        return pathlib.Path(__file__).parent.parent

    @property
    def pythonpath(self) -> pathlib.Path:
        return self.basedir.parent
