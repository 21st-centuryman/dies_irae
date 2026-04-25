"""Sanity checks for the wire format.

Locks in the sizes and byte layout so a frontend implementer can rely on
the spec in ARCHITECTURE.md.
"""

from __future__ import annotations

import struct

import numpy as np

from app.protocol import (
    DRONE_RECORD_DTYPE,
    DRONE_RECORD_SIZE,
    HEADER_DTYPE,
    HEADER_SIZE,
    pack_frame,
)


def test_record_and_header_sizes():
    assert DRONE_RECORD_SIZE == 30
    assert HEADER_SIZE == 16


def test_pack_frame_layout():
    records = np.zeros(2, dtype=DRONE_RECORD_DTYPE)
    records[0]["id"] = 7
    records[0]["type"] = 1
    records[0]["state"] = 2
    records[0]["intent"] = 3
    records[0]["x"] = 1.0
    records[0]["y"] = 2.0
    records[0]["z"] = 3.0
    records[0]["vx"] = 4.0
    records[0]["vy"] = 5.0
    records[0]["vz"] = 6.0
    records[1]["id"] = 42

    blob = pack_frame(frame_number=123, timestamp=9.5, records=records)
    assert len(blob) == HEADER_SIZE + 2 * DRONE_RECORD_SIZE

    frame_number, timestamp, drone_count, _pad = struct.unpack_from("<IdHH", blob, 0)
    assert frame_number == 123
    assert timestamp == 9.5
    assert drone_count == 2

    # First record, field-by-field
    off = HEADER_SIZE
    rid, rtype, rstate, rintent, _rpad = struct.unpack_from("<HBBBB", blob, off)
    assert (rid, rtype, rstate, rintent) == (7, 1, 2, 3)
    x, y, z, vx, vy, vz = struct.unpack_from("<ffffff", blob, off + 6)
    assert (x, y, z, vx, vy, vz) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    # Second record id lands where expected.
    off2 = HEADER_SIZE + DRONE_RECORD_SIZE
    (rid2,) = struct.unpack_from("<H", blob, off2)
    assert rid2 == 42


def test_pack_frame_rejects_wrong_dtype():
    bad = np.zeros(1, dtype=np.float32)
    try:
        pack_frame(0, 0.0, bad)
    except TypeError:
        return
    raise AssertionError("expected TypeError")
