import unittest

from comfyui_character_generator.util.operations import (add_with_rotate_64,
                                                         sub_with_rotate_64)


class TestOperations(unittest.TestCase):
    def test_add_with_rotate_64_basic(self):
        self.assertEqual(add_with_rotate_64(1, 2), 3)
        self.assertEqual(add_with_rotate_64(0, 0), 0)

    def test_add_with_rotate_64_overflow(self):
        # 0xFFFFFFFFFFFFFFFF + 1 = 0 (overflow), rotate by 1
        result = add_with_rotate_64(0xFFFFFFFFFFFFFFFF, 1)
        # After truncation: 0, rotate-left by 1: still 0
        self.assertEqual(result, 0)

    def test_sub_with_rotate_64_basic(self):
        self.assertEqual(sub_with_rotate_64(5, 3), 2)
        self.assertEqual(sub_with_rotate_64(0, 0), 0)

    def test_sub_with_rotate_64_underflow(self):
        # 0 - 1 = -1, wraps to 0xFFFFFFFFFFFFFFFF, rotate right by 1
        result = sub_with_rotate_64(0, 1)
        # Rotating 0xFFFFFFFFFFFFFFFF right by 1 is still 0xFFFFFFFFFFFFFFFF
        self.assertEqual(result, 0xFFFFFFFFFFFFFFFF)
