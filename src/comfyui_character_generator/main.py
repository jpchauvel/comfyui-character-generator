import shlex
import subprocess

from comfyui_character_generator.util.manager import AppManager


def main() -> None:
    manager: AppManager = AppManager()

    if manager.config.venv_path is None:
        raise ValueError("No venv path specified")

    if manager.should_install_nodes:

        if manager.config.comfyui_path is None:
            raise ValueError("No comfyui path specified")

        command: str = shlex.quote(
            (manager.basedir / "bin" / "install_nodes.sh").as_posix()
        )
        args: tuple[str, ...] = (
            shlex.quote(manager.config.venv_path.as_posix()),
            shlex.quote(manager.config.comfyui_path.as_posix()),
        )
        subprocess.run([command, *args], shell=False)
    else:
        for config_idx, config in enumerate(manager.config.sub_configs):
            for prompt_idx in range(config.sub_prompt_count):
                for _ in range(config.loop_count):
                    command = shlex.quote(
                        (manager.basedir / "bin" / "generate.sh").as_posix()
                    )
                    args = (
                        shlex.quote(manager.config.venv_path.as_posix()),
                        shlex.quote(manager.pythonpath.as_posix()),
                        shlex.quote(str(config_idx)),
                        shlex.quote(str(prompt_idx)),
                    )

                    subprocess.run(
                        [command, *args],
                        input=manager.config.dump().encode("utf-8"),
                        shell=False,
                    )

                    config.seed = manager.generate_new_seed(
                        config.seed_generation, config.seed
                    )


if __name__ == "__main__":
    main()
