"""Compare frozen renderer settings without rejecting float32 storage."""
import struct


def setting_matches(actual, expected):
    if isinstance(expected, float):
        # Carb may preserve a double or store the exact IEEE-754 float32 value.
        # Accept those two representations only, rather than a loose tolerance.
        single = struct.unpack('f', struct.pack('f', expected))[0]
        return type(actual) in (int, float) and actual in (expected, single)
    return type(actual) is type(expected) and actual == expected
