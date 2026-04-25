"""Binary wire format for drone position frames.

Little-endian, NumPy structured dtypes so packing N drones is a single
``.tobytes()`` call on a contiguous array — the layout on disk/wire matches
the layout in memory exactly.

Header (16B) + drone_count * record (30B each).
"""

from __future__ import annotations

import numpy as np

DRONE_RECORD_DTYPE = np.dtype(
    [
        ("id", "<u2"),
        ("type", "<u1"),
        ("state", "<u1"),
        ("intent", "<u1"),
        ("_pad", "<u1"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("vx", "<f4"),
        ("vy", "<f4"),
        ("vz", "<f4"),
    ],
    align=False,
)

HEADER_DTYPE = np.dtype(
    [
        ("frame_number", "<u4"),
        ("timestamp", "<f8"),
        ("drone_count", "<u2"),
        ("_pad", "<u2"),
    ],
    align=False,
)

DRONE_RECORD_SIZE = DRONE_RECORD_DTYPE.itemsize  # 30
HEADER_SIZE = HEADER_DTYPE.itemsize  # 16

assert DRONE_RECORD_SIZE == 30, DRONE_RECORD_SIZE
assert HEADER_SIZE == 16, HEADER_SIZE


def pack_frame(frame_number: int, timestamp: float, records: np.ndarray) -> bytes:
    """Concatenate envelope header + records into one binary blob.

    `records` must be a 1-D array with dtype DRONE_RECORD_DTYPE.
    """
    if records.dtype != DRONE_RECORD_DTYPE:
        raise TypeError(f"records dtype {records.dtype} != {DRONE_RECORD_DTYPE}")
    if records.ndim != 1:
        raise ValueError("records must be 1-D")

    header = np.zeros(1, dtype=HEADER_DTYPE)
    header["frame_number"] = frame_number
    header["timestamp"] = timestamp
    header["drone_count"] = records.shape[0]
    return header.tobytes() + np.ascontiguousarray(records).tobytes()
