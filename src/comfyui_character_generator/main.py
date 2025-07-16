import shlex
import subprocess

from comfyui_character_generator.util import AppManager


def main() -> None:
    manager: AppManager = AppManager()

    if manager.config.venv_path is None:
        raise ValueError("No venv path specified")

    for _ in range(manager.config.loop_count):
        command: str = shlex.quote(
            (manager.basedir / "bin" / "generate.sh").as_posix()
        )
        args: tuple[str, ...] = (
            shlex.quote(manager.config.venv_path.as_posix()),
            shlex.quote(manager.pythonpath.as_posix()),
        )

        subprocess.run(
            [command, *args],
            input=manager.config.dump().encode("utf-8"),
            shell=False,
        )

        manager.config.seed = manager.generate_new_seed()


if __name__ == "__main__":
    main()
