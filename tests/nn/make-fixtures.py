#!/usr/bin/env python3
"""Makes tests/nn/nn-test.nnw: random weights, inputs and the expected
outputs of every nn/nn.js layer, computed with plain numpy (an
independent implementation of the PyTorch formulas). Needs only numpy.

    python3 tests/nn/make-fixtures.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'nn'))
from export import write_nnw

rng = np.random.default_rng(7)
f32 = lambda *s: rng.standard_normal(s).astype('<f4') * 0.5
sig = lambda x: 1 / (1 + np.exp(-x))
T = {}


def softmax(x, t=1.0):
    e = np.exp((x - x.max()) / t)
    return e / e.sum()


def run_cell(kind, p, x, h, c):
    H = h.shape[0]
    gi = p['wi'] @ x + p['bi']
    gh = p['wh'] @ h + p['bh']
    if kind == 'gru':
        r = sig(gi[:H] + gh[:H])
        z = sig(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        return (1 - z) * n + z * h, c
    i, f = sig(gi[:H] + gh[:H]), sig(gi[H:2 * H] + gh[H:2 * H])
    g, o = np.tanh(gi[2 * H:3 * H] + gh[2 * H:3 * H]), sig(gi[3 * H:] + gh[3 * H:])
    c = f * c + i * g
    return o * np.tanh(c), c


def make_rnn(name, kind, inp, H, layers, bidir):
    G = 3 if kind == 'gru' else 4
    dirs = 2 if bidir else 1
    P = {}
    for l in range(layers):
        for d in range(dirs):
            s = f'_l{l}' + ('_reverse' if d else '')
            i = inp if l == 0 else H * dirs
            P[l, d] = dict(wi=f32(G * H, i), wh=f32(G * H, H), bi=f32(G * H), bh=f32(G * H))
            for k, v in zip(('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'), P[l, d].values()):
                T[f'{name}.{k}{s}'] = v
    x = f32(9, inp)
    T[f'{name}.x'] = x
    cur = x
    for l in range(layers):
        out = np.zeros((len(cur), H * dirs), '<f4')
        for d in range(dirs):
            h, c = np.zeros(H, '<f4'), np.zeros(H, '<f4')
            for i in range(len(cur)):
                t = len(cur) - 1 - i if d else i
                h, c = run_cell(kind, P[l, d], cur[t], h, c)
                out[t, d * H:(d + 1) * H] = h
        cur = out
    T[f'{name}.y'] = cur


# Linear
T['fc.weight'], T['fc.bias'] = f32(5, 7), f32(5)
T['fc.x'] = f32(3, 7)
T['fc.y'] = T['fc.x'] @ T['fc.weight'].T + T['fc.bias']

# Embedding
T['emb.weight'] = f32(11, 4)
T['emb.ids'] = np.array([3, 0, 10, 7], '<f4')
T['emb.y'] = T['emb.weight'][[3, 0, 10, 7]]

# LayerNorm
T['ln.weight'], T['ln.bias'] = f32(6) + 1, f32(6)
T['ln.x'] = f32(4, 6)
x = T['ln.x']
T['ln.y'] = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5) * T['ln.weight'] + T['ln.bias']

# Conv1d, stride 2, padding 2, dilation 2
T['conv.weight'], T['conv.bias'] = f32(4, 3, 3), f32(4)
T['conv.x'] = f32(13, 3)
x, w, b = T['conv.x'], T['conv.weight'], T['conv.bias']
Tout = (13 + 4 - 2 * 2 - 1) // 2 + 1
y = np.zeros((Tout, 4), '<f4')
for t in range(Tout):
    for o in range(4):
        s = b[o]
        for j in range(3):
            p = t * 2 - 2 + j * 2
            if 0 <= p < 13:
                s += w[o, :, j] @ x[p]
        y[t, o] = s
T['conv.y'] = y

# activations and softmax
a = f32(16) * 4
T['act.x'] = a
T['act.relu'], T['act.tanh'], T['act.sigmoid'] = np.maximum(a, 0), np.tanh(a), sig(a)
T['act.silu'] = a * sig(a)
T['act.gelu'] = np.array([0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in a], '<f4')
T['act.softmax'] = softmax(a)
T['act.softmax_t'] = softmax(a, 0.5)

make_rnn('gru', 'gru', 5, 8, 1, False)
make_rnn('bigru', 'gru', 5, 6, 2, True)
make_rnn('lstm', 'lstm', 5, 7, 2, False)

out = os.path.join(os.path.dirname(__file__), 'nn-test.nnw')
write_nnw(out, T, meta={'generator': 'make-fixtures.py'})
print('wrote', out, len(T), 'tensors')
