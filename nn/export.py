#!/usr/bin/env python3
"""Writes a PyTorch state_dict (or any name -> array dict) as an .nnw file.

File layout, all integers little endian:

    "NNW1"            4 bytes magic
    header_len        uint32, length of the JSON header in bytes
    header            UTF-8 JSON, padded with spaces to a multiple of 4
    blob              float32 data, every tensor starts on a 4 byte boundary

The header is {"meta": {...}, "tensors": {name: {"shape": [...],
"offset": n}}}; offset counts bytes from the start of the blob.

    from export import write_nnw
    write_nnw('model.nnw', model.state_dict(), meta={'hidden': 64})

Only numpy is needed; torch tensors are accepted when torch is installed.
"""
import json
import struct

import numpy as np

MAGIC = b'NNW1'


def _to_array(v):
    if hasattr(v, 'detach'):
        v = v.detach().cpu().numpy()
    return np.ascontiguousarray(v, dtype='<f4')


def write_nnw(path, tensors, meta=None):
    tensors = {k: _to_array(v) for k, v in tensors.items()}
    index, offset = {}, 0
    for name, a in tensors.items():
        index[name] = {'shape': list(a.shape), 'offset': offset}
        offset += a.nbytes
    header = json.dumps({'meta': meta or {}, 'tensors': index}).encode()
    header += b' ' * (-(8 + len(header)) % 4)
    with open(path, 'wb') as f:
        f.write(MAGIC + struct.pack('<I', len(header)) + header)
        for a in tensors.values():
            f.write(a.tobytes())


if __name__ == '__main__':
    import sys
    import torch
    src, dst = sys.argv[1:3]
    obj = torch.load(src, map_location='cpu')
    write_nnw(dst, obj.get('state_dict', obj) if isinstance(obj, dict) else obj.state_dict())
