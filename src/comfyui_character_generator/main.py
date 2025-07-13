import pathlib
import shlex
import subprocess

from comfyui_character_generator.util import AppManager, dump_toml


def main() -> None:
    manager: AppManager = AppManager()
    basedir: pathlib.Path = pathlib.Path(__file__).parent
    pythonpath: pathlib.Path = basedir.parent
    if manager.config.venv_path is None:
        raise ValueError("No venv path specified")
    for _ in range(manager.config.loop_count):
        command: str = shlex.quote((basedir / "generate.sh").as_posix())
        args: tuple[str, ...] = (
            shlex.quote(manager.config.venv_path.as_posix()),
            shlex.quote(pythonpath.as_posix()),
        )
        subprocess.run(
            [command, *args],
            input=dump_toml(manager).encode("utf-8"),
            shell=False,
        )
        manager.config.seed = manager.generate_new_seed()


if __name__ == "__main__":
    main()
