def add_with_rotate_64(a, b) -> int:
    full_sum: int = a + b
    base: int = full_sum & 0xFFFFFFFFFFFFFFFF  # Normal 64-bit truncation

    # Number of bits overflowed (max possible: up to 64)
    overflow_bit_count: int = (
        full_sum.bit_length() - 64 if full_sum.bit_length() > 64 else 0
    )

    # Rotate-left base by number of overflow bits
    rotated: int = (
        (base << overflow_bit_count) | (base >> (64 - overflow_bit_count))
    ) & 0xFFFFFFFFFFFFFFFF
    return rotated


def sub_with_rotate_64(a, b) -> int:
    diff: int = a - b

    if diff >= 0:
        # No underflow — just normal subtraction
        return diff & 0xFFFFFFFFFFFFFFFF
    else:
        # Underflow occurred
        # Simulate 64-bit wraparound (as unsigned would do)
        wrapped: int = (diff + (1 << 64)) & 0xFFFFFFFFFFFFFFFF

        # How many bits underflowed?
        # Use the absolute difference to estimate underflow “depth”
        borrow: int = abs(diff)
        overflow_bit_count: int = borrow.bit_length()

        # Rotate right the wrapped result by number of overflow bits
        rotated: int = (
            (wrapped >> overflow_bit_count)
            | (wrapped << (64 - overflow_bit_count))
        ) & 0xFFFFFFFFFFFFFFFF
        return rotated
