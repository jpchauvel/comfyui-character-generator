import sys
import unittest
from unittest import mock

from comfyui_character_generator.util import args as args_module


class TestArgs(unittest.TestCase):
    def test_default_args(self):
        test_argv = ["prog"]
        with mock.patch.object(sys, "argv", test_argv):
            parsed = args_module.get_args()
            self.assertIsNotNone(parsed)
            self.assertEqual(
                parsed.pose_detection_type,
                args_module.DEFAULT_POSE_DETECTION_TYPE,
            )
            self.assertEqual(parsed.venv_path, args_module.DEFAULT_VENV_PATH)
            self.assertEqual(parsed.steps, args_module.DEFAULT_STEPS)
            self.assertEqual(
                parsed.guidance_scale, args_module.DEFAULT_GUIDANCE_SCALE
            )
            self.assertEqual(parsed.batch, args_module.DEFAULT_BATCH)
            self.assertEqual(parsed.width, args_module.DEFAULT_WIDTH)
            self.assertEqual(
                parsed.aspect_ratio, args_module.DEFAULT_ASPECT_RATIO
            )
            self.assertEqual(parsed.loop_count, args_module.DEFAULT_LOOP_COUNT)
            self.assertEqual(
                parsed.seed_generation, args_module.DEFAULT_SEED_GENERATION
            )

    def test_custom_args(self):
        test_argv = [
            "prog",
            "--steps",
            "99",
            "--guidance_scale",
            "7.5",
            "--batch",
            "3",
            "--width",
            "512",
            "--aspect_ratio",
            "4:3",
        ]
        with mock.patch.object(sys, "argv", test_argv):
            parsed = args_module.get_args()
            self.assertEqual(parsed.steps, 99)
            self.assertEqual(parsed.guidance_scale, 7.5)
            self.assertEqual(parsed.batch, 3)
            self.assertEqual(parsed.width, 512)
            self.assertEqual(parsed.aspect_ratio, "4:3")
