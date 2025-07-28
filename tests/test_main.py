import pathlib
import unittest
from unittest import mock

import pytest

import comfyui_character_generator.main as main_mod


def make_mock_manager(
    should_install_nodes=True,
    venv_path="/venv",
    comfyui_path="/comfyui",
    sub_configs=None,
):
    # Use real pathlib.Path to mock venv_path and comfyui_path
    mock_cfg = mock.MagicMock()
    mock_cfg.venv_path = pathlib.Path(venv_path)
    mock_cfg.comfyui_path = pathlib.Path(comfyui_path)
    mock_cfg.sub_configs = sub_configs or []
    mock_cfg.dump.return_value = "dumped-config"

    # Patch `basedir / "bin" / "generate.sh"` and similar
    basedir = mock.MagicMock(spec=pathlib.Path)
    basedir.__truediv__.side_effect = lambda x: pathlib.Path(f"/base/{x}")

    return mock.Mock(
        config=mock_cfg,
        should_install_nodes=should_install_nodes,
        basedir=basedir,
        pythonpath=pathlib.Path("/python"),
        generate_new_seed=lambda a, b: 42,
    )


class TestMain(unittest.TestCase):
    def setUp(self):
        self.patcher_app_manager = mock.patch.object(main_mod, "AppManager")
        self.mock_app_manager = self.patcher_app_manager.start()
        self.patcher_run = mock.patch.object(main_mod.subprocess, "run")
        self.mock_run = self.patcher_run.start()

    def tearDown(self):
        self.patcher_app_manager.stop()
        self.patcher_run.stop()

    def test_should_install_nodes(self):
        self.mock_app_manager.return_value = make_mock_manager(True)
        main_mod.main()
        self.assertEqual(self.mock_run.call_count, 1)
        args = self.mock_run.call_args[0][0]
        self.assertIn("/base/bin/install_nodes.sh", args[0])
        self.assertIn("/venv", args[1])
        self.assertIn("/comfyui", args[2])

    def test_generate_branch(self):
        sub_config = mock.Mock()
        sub_config.sub_prompt_count = 2
        sub_config.loop_count = 2
        sub_config.seed = 1
        sub_config.seed_generation = "foo"
        sub_configs = [sub_config]
        self.mock_app_manager.return_value = make_mock_manager(
            False, sub_configs=sub_configs
        )
        main_mod.main()
        self.assertEqual(self.mock_run.call_count, 4)
        for call in self.mock_run.call_args_list:
            args = call[0][0]
            self.assertIn("/base/bin/generate.sh", args[0])
            self.assertIn("/venv", args[1])
            self.assertIn("/python", args[2])
            self.assertEqual(call[1]["input"], b"dumped-config")

    def test_raises_no_venv(self):
        manager = make_mock_manager()
        manager.config.venv_path = None
        self.mock_app_manager.return_value = manager
        with pytest.raises(ValueError, match="No venv path specified"):
            main_mod.main()

    def test_raises_no_comfyui(self):
        manager = make_mock_manager()
        manager.should_install_nodes = True
        manager.config.comfyui_path = None
        self.mock_app_manager.return_value = manager
        with pytest.raises(ValueError, match="No comfyui path specified"):
            main_mod.main()
