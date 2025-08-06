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

    def __new__(cls: type, value: int) -> Any:
        value -= 1
        member: IntEnum = int.__new__(cls, value)
        member._value_ = value
        return member
