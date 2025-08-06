import pathlib
import types
import unittest
from unittest import mock

from comfyui_character_generator.util.config import Config, GlobalConfig


class TestConfig(unittest.TestCase):
    @mock.patch("os.path.isdir", return_value=True)
    def test_global_config_init(self, mock_isdir):
        attrs = dict(
            comfyui_path="/tmp/comfyui",
            venv_path="venv",
            ckpt="model.safetensors",
            loras=["lora1.safetensors"],
            lora_strengths=[1.0],
            controlnet="controlnet.safetensors",
            upscaler="upscaler.safetensors",
            disable_controlnet=False,
            steps=10,
            seed=42,
            guidance_scale=7.5,
            batch=2,
            width=512,
            aspect_ratio="1:1",
            system_prompt="this is a system prompt",
            system_neg_prompt="this is a system negative prompt",
            loop_count=1,
            seed_generation=1,
        )
        config = GlobalConfig(**attrs)
        self.assertEqual(config.comfyui_path, pathlib.Path("/tmp/comfyui"))
        self.assertEqual(config.venv_path, pathlib.Path("/tmp/comfyui/venv"))
        self.assertEqual(
            config._input_path, pathlib.Path("/tmp/comfyui/input")
        )
        self.assertEqual(config.ckpt, "model.safetensors")
        self.assertEqual(config.loras, ["lora1.safetensors"])
        self.assertEqual(config.lora_strengths, [1.0])
        self.assertEqual(config.controlnet, "controlnet.safetensors")
        self.assertEqual(config.upscaler, "upscaler.safetensors")
        self.assertEqual(config.disable_controlnet, False)
        self.assertEqual(config.steps, 10)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.guidance_scale, 7.5)
        self.assertEqual(config.batch, 2)
        self.assertEqual(config.width, 512)
        self.assertEqual(config.aspect_ratio, "1:1")
        self.assertEqual(config.system_prompt, "this is a system prompt")
        self.assertEqual(
            config.system_neg_prompt, "this is a system negative prompt"
        )
        self.assertEqual(config.loop_count, 1)
        self.assertEqual(config.seed_generation, 1)

    @mock.patch("os.path.isfile", return_value=True)
    @mock.patch("os.path.isdir", return_value=True)
    @mock.patch("shutil.copyfile")
    def test_config_init(self, mock_copyfile, mock_isdir, mock_isfile):
        comfyui_path = pathlib.Path("/tmp/comfyui")
        input_path = comfyui_path / "input"
        attrs = dict(
            ckpt="model.safetensors",
            loras=["lora1.safetensors"],
            lora_strengths=[1.0],
            controlnet="controlnet.safetensors",
            upscaler="upscaler.safetensors",
            disable_controlnet=False,
            steps=10,
            seed=42,
            guidance_scale=7.5,
            batch=2,
            width=512,
            aspect_ratio="1:1",
            system_prompt="prompt",
            system_neg_prompt="neg",
            neg_prompts=["neg1"],
            sub_prompts=["sub1"],
            face_swap_images=["/tmp/face1.png"],
            pose_images=["/tmp/pose1.png"],
            pose_detection_type=0,
            loop_count=1,
            seed_generation=1,
            output_path="/tmp/output",
        )
        config = Config(comfyui_path, input_path, validate=True, **attrs)
        self.assertEqual(config.ckpt, "model.safetensors")
        self.assertEqual(config.loras, ["lora1.safetensors"])
        self.assertEqual(config.lora_strengths, [1.0])
        self.assertEqual(config.face_swap_images, ["face1.png"])
        self.assertEqual(config.pose_images, ["pose1.png"])
        self.assertEqual(config.upscaler, "upscaler.safetensors")

    def test_config_from_dict_list(self):
        data = [
            {
                "ckpt": "model.safetensors",
                "loras": ["lora1.safetensors"],
                "lora_strengths": [1.0],
                "controlnet": "controlnet.safetensors",
                "upscaler": "upscaler.safetensors",
                "disable_controlnet": False,
                "steps": 10,
                "seed": 42,
                "guidance_scale": 7.5,
                "batch": 2,
                "width": 512,
                "aspect_ratio": "1:1",
                "system_prompt": "prompt",
                "system_neg_prompt": "neg",
                "neg_prompts": ["neg1"],
                "sub_prompts": ["sub1"],
                "face_swap_images": ["face1.png"],
                "pose_images": ["pose1.png"],
                "pose_detection_type": 0,
                "loop_count": 1,
                "seed_generation": 1,
                "output_path": pathlib.Path("/tmp/output"),
            }
        ]
        configs = Config.from_dict_list(data)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].ckpt, "model.safetensors")
        self.assertEqual(configs[0].loras, ["lora1.safetensors"])
        self.assertEqual(configs[0].output_path, pathlib.Path("/tmp/output"))
        self.assertEqual(configs[0].upscaler, "upscaler.safetensors")

    def test_global_config_from_dict(self):
        data = {
            "ckpt": "model.safetensors",
            "loras": ["lora1.safetensors"],
            "lora_strengths": [1.0],
            "controlnet": "controlnet.safetensors",
            "upscaler": "upscaler.safetensors",
            "disable_controlnet": False,
            "steps": 10,
            "seed": 42,
            "guidance_scale": 7.5,
            "batch": 2,
            "width": 512,
            "aspect_ratio": "1:1",
            "system_prompt": "prompt",
            "system_neg_prompt": "neg",
            "neg_prompts": ["neg1"],
            "sub_prompts": ["sub1"],
            "face_swap_images": ["face1.png"],
            "pose_images": ["pose1.png"],
            "pose_detection_type": 0,
            "loop_count": 1,
            "seed_generation": 1,
            "output_path": "/tmp/output",
            "comfyui_path": "/tmp/comfyui",
            "venv_path": "/tmp/comfyui/venv",
            "sub_configs": [
                {
                    "ckpt": "model.safetensors",
                    "loras": ["lora1.safetensors"],
                    "lora_strengths": [1.0],
                    "controlnet": "controlnet.safetensors",
                    "upscaler": "upscaler.safetensors",
                    "disable_controlnet": False,
                    "steps": 10,
                    "seed": 42,
                    "guidance_scale": 7.5,
                    "batch": 2,
                    "width": 512,
                    "aspect_ratio": "1:1",
                    "system_prompt": "prompt",
                    "system_neg_prompt": "neg",
                    "neg_prompts": ["neg1"],
                    "sub_prompts": ["sub1"],
                    "face_swap_images": ["face1.png"],
                    "pose_images": ["pose1.png"],
                    "pose_detection_type": 1,
                    "loop_count": 1,
                    "seed_generation": 1,
                    "output_path": "/tmp/output",
                }
            ],
        }
        with mock.patch("os.path.isdir", return_value=True):
            config = GlobalConfig.from_dict(data)
        self.assertEqual(config.comfyui_path, pathlib.Path("/tmp/comfyui"))
        self.assertEqual(config.venv_path, pathlib.Path("/tmp/comfyui/venv"))
        self.assertEqual(len(config.sub_configs), 1)
        self.assertEqual(config.sub_configs[0].ckpt, "model.safetensors")
        self.assertEqual(config.upscaler, "upscaler.safetensors")

    def test_global_config_dump_and_load(self):
        data = {
            "ckpt": "model.safetensors",
            "loras": ["lora1.safetensors"],
            "lora_strengths": [1.0],
            "controlnet": "controlnet.safetensors",
            "upscaler": "upscaler.safetensors",
            "disable_controlnet": False,
            "steps": 10,
            "seed": 42,
            "guidance_scale": 7.5,
            "batch": 2,
            "width": 512,
            "aspect_ratio": "1:1",
            "system_prompt": "prompt",
            "system_neg_prompt": "neg",
            "neg_prompts": ["neg1"],
            "sub_prompts": ["sub1"],
            "face_swap_images": ["face1.png"],
            "pose_images": ["pose1.png"],
            "loop_count": 1,
            "pose_detection_type": 0,
            "seed_generation": 1,
            "output_path": "/tmp/output",
            "comfyui_path": "/tmp/comfyui",
            "venv_path": "/tmp/comfyui/venv",
            "sub_configs": [],
        }
        with mock.patch("os.path.isdir", return_value=True):
            config = GlobalConfig.from_dict(data)
        dumped = config.dump()
        loaded = GlobalConfig.load(dumped)
        self.assertEqual(loaded.comfyui_path, pathlib.Path("/tmp/comfyui"))
        self.assertEqual(loaded.venv_path, pathlib.Path("/tmp/comfyui/venv"))
        self.assertEqual(loaded.ckpt, "model.safetensors")
        self.assertEqual(loaded.upscaler, "upscaler.safetensors")

    def test_global_default_dict_value(self):
        path = pathlib.Path("/tmp/foo")
        result = GlobalConfig.default_dict_value(path)
        self.assertEqual(result, "/tmp/foo")

    @mock.patch(
        "builtins.open", new_callable=mock.mock_open, read_data=b"fake toml"
    )
    @mock.patch("tomllib.load")
    @mock.patch("os.path.isdir", return_value=True)
    @mock.patch("os.path.isfile", return_value=True)
    def test_global_config_load_from_toml(
        self, mock_isfile, mock_isdir, mock_toml_load, mock_open
    ):
        fake_toml = {
            "global": {
                "comfyui_path": "/tmp/comfyui",
                "venv_path": "venv",
                "ckpt_path": "model.safetensors",
                "lora_paths": ["lora1.safetensors"],
                "lora_strengths": [1.0],
                "controlnet_path": "controlnet.safetensors",
                "upscaler_path": "upscaler.safetensors",
                "disable_controlnet": False,
                "steps": 10,
                "seed": 42,
                "guidance_scale": 7.5,
                "batch": 2,
                "width": 512,
                "aspect_ratio": "1:1",
                "loop_count": 1,
                "seed_generation": 1,
                "output_path": "/tmp/output",
            },
            "config1": {
                "system_prompt": "prompt",
                "system_neg_prompt": "neg",
                "neg_prompts": ["neg1"],
                "sub_prompts": ["sub1"],
                "face_swap_image_paths": ["face1.png"],
                "pose_image_paths": ["pose1.png"],
            },
        }
        mock_toml_load.return_value = fake_toml
        config = GlobalConfig.load_from_toml(pathlib.Path("/tmp/config.toml"))
        self.assertEqual(config.comfyui_path, pathlib.Path("/tmp/comfyui"))
        self.assertEqual(config.venv_path, pathlib.Path("/tmp/comfyui/venv"))
        self.assertEqual(config.ckpt, "model.safetensors")
        self.assertEqual(len(config.sub_configs), 1)
        self.assertEqual(config.sub_configs[0].face_swap_images, ["face1.png"])
