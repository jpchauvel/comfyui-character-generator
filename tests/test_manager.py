import pathlib
import types
import unittest
from unittest import mock

from comfyui_character_generator.util.manager import AppManager


class TestAppManager(unittest.TestCase):
    @mock.patch("comfyui_character_generator.util.manager.get_args")
    def test_install_nodes_flag(self, mock_get_args):
        mock_args = types.SimpleNamespace(
            install_nodes=True,
            comfyui_path="/tmp/comfyui",
            venv_path="venv",
            config_path=None,
            ckpt_path=None,
            lora_paths=None,
            controlnet_path=None,
            upscaler_path=None,
            system_prompt_path=None,
            system_neg_prompt_path=None,
            face_swap_image_paths=None,
            pose_image_paths=None,
            lora_strengths=None,
        )
        mock_get_args.return_value = mock_args
        with (
            mock.patch("os.chdir") as mock_chdir,
            mock.patch(
                "comfyui_character_generator.util.config.GlobalConfig.__init__",
                return_value=None,
            ),
        ):
            manager = AppManager()
            self.assertTrue(manager.should_install_nodes)

    @mock.patch("comfyui_character_generator.util.manager.get_args")
    def test_missing_required_args(self, mock_get_args):
        mock_args = types.SimpleNamespace(
            install_nodes=False,
            comfyui_path=None,
            config_path=None,
            ckpt_path=None,
            lora_paths=None,
            controlnet_path=None,
            upscaler_path=None,
            system_prompt_path=None,
            system_neg_prompt_path=None,
            face_swap_image_paths=None,
            pose_image_paths=None,
            lora_strengths=None,
        )
        mock_get_args.return_value = mock_args
        with self.assertRaises(ValueError):
            AppManager()
