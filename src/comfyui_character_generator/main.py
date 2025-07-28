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
        for _ in range(manager.config.loop_count):
            for prompt_idx in range(manager.config.sub_prompt_count):
                command = shlex.quote(
                    (manager.basedir / "bin" / "generate.sh").as_posix()
                )
                args = (
                    shlex.quote(manager.config.venv_path.as_posix()),
                    shlex.quote(manager.pythonpath.as_posix()),
                    shlex.quote(str(prompt_idx)),
                )

                subprocess.run(
                    [command, *args],
                    input=manager.config.dump().encode("utf-8"),
                    shell=False,
                )

                manager.config.seed = manager.generate_new_seed()


if __name__ == "__main__":
    main()
