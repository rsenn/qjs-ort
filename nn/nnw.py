"""Reads .nnw files (layout in export.py) into numpy arrays."""
import json
import struct

import numpy as np


def read_nnw(path):
    """Returns (meta, {name: ndarray}); arrays are memory-mapped views."""
    with open(path, 'rb') as f:
        if f.read(4) != b'NNW1':
            raise ValueError(f'{path}: not an .nnw file')
        n = struct.unpack('<I', f.read(4))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for name, t in header['tensors'].items():
        out[name] = np.memmap(path, dtype='<f4', mode='r', offset=base + t['offset'], shape=tuple(t['shape']))
    return header['meta'], out
