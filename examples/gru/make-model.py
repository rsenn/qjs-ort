#!/usr/bin/env python3
"""Exports a tiny GRU step model for the gru-loop.js example.

The model takes one time step x [1, 1, 8] and the hidden state h [1, 1, 16]
and returns y [1, 1, 16] and the new hidden state hn [1, 1, 16]. Feeding hn
back as h is how a sequence model runs one block at a time in a loop.

Run it with the audio-ml virtual environment (torch and onnxruntime):

    python make-model.py

It writes gru-step.onnx and gru-expected.json (the inputs and the outputs
that ONNX Runtime for Python gives, which gru-loop.js compares against).
"""
import json
import numpy as np
import onnxruntime as ort
import torch

STEPS, IN, HID = 12, 8, 16

torch.manual_seed(1)
gru = torch.nn.GRU(IN, HID, batch_first=True).eval()

x = torch.zeros(1, 1, IN)
h = torch.zeros(1, 1, HID)

torch.onnx.export(
    gru,
    (x, h),
    'gru-step.onnx',
    input_names=['x', 'h'],
    output_names=['y', 'hn'],
    opset_version=17,
    dynamo=False,
)

rng = np.random.default_rng(7)
xs = rng.standard_normal((STEPS, IN)).astype(np.float32)

sess = ort.InferenceSession('gru-step.onnx', providers=['CPUExecutionProvider'])
hs = np.zeros((1, 1, HID), np.float32)
ys = []

for t in range(STEPS):
    y, hs = sess.run(None, {'x': xs[t].reshape(1, 1, IN), 'h': hs})
    ys.append(y.reshape(-1).tolist())

with open('gru-expected.json', 'w') as f:
    json.dump({'steps': STEPS, 'in': IN, 'hidden': HID, 'x': xs.tolist(), 'y': ys}, f)

print('wrote gru-step.onnx and gru-expected.json')
