from enum import IntEnum, auto
from typing import Any


class SeedGenerationMethod(IntEnum):
    INCREMENT = auto()
    DECREMENT = auto()
    RANDOM = auto()


class PoseDetectionType(IntEnum):
    OPENPOSE = auto()
    REALISTIC_LINEART = auto()
    DEPTH = auto()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._value_ -= 1
