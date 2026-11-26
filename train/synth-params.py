#!/usr/bin/env python3
"""Trains the Case 4 model: log-mel features of a rendered note -> synth
parameters (see doc/AM-DNN-CASES.md). Data comes from tools/render-dataset.js.

    python3 train/synth-params.py data/synth out/synth-params.nnw [--epochs 60]

data/synth must hold train-*.nnw and val.nnw. Writes the model (.nnw, loads
with nn/models/synth-params.js) and <out>.check.nnw with a few validation
inputs and this script's outputs, which tests/synth-params-test.js compares
against the JS inference.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nn'))
from export import write_nnw
from nnw import read_nnw


class Net(nn.Module):
    """Names and shapes match nn/models/synth-params.js."""

    def __init__(self, n_mels, frames, n_out):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, 64, 5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(64, 96, 5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(96, 64, 5, stride=2, padding=2)
        t = frames
        for _ in range(2):
            t = (t + 4 - 4 - 1) // 2 + 1
        self.fc1 = nn.Linear(64 * t + 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, n_out)

    def forward(self, x, pitch):  # x [B, T, M] normalised, pitch [B]
        h = x.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.transpose(1, 2).flatten(1)  # time-major, like the JS layout
        h = torch.cat([h, pitch[:, None]], 1)
        return self.head(F.relu(self.fc2(F.relu(self.fc1(h)))))


def load(paths):
    metas, feats, labs, midis = None, [], [], []
    for p in paths:
        meta, t = read_nnw(p)
        metas = metas or meta
        feats.append(np.asarray(t['features']))
        labs.append(np.asarray(t['labels']))
        midis.append(np.asarray(t['midi']))
    return metas, np.concatenate(feats), np.concatenate(labs), np.concatenate(midis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('out')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3)
    a = ap.parse_args()

    meta, X, Y, M = load(sorted(glob.glob(os.path.join(a.data, 'train-*.nnw'))))
    _, Xv, Yv, Mv = load([os.path.join(a.data, 'val.nnw')])
    spec = meta['spec']
    kinds = [p['type'] for p in spec['params']]
    fidx = [i for i, k in enumerate(kinds) if k == 'float']
    cidx = [i for i, k in enumerate(kinds) if k == 'choice']
    csize = [len(spec['params'][i]['choices']) for i in cidx]
    n_out = len(fidx) + sum(csize)

    mean = X.mean((0, 1)).astype('f4')
    std = X.std((0, 1)).astype('f4') + 1e-3
    pitch = lambda m: (m - 60.0) / 24.0
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{len(X)} train, {len(Xv)} val, {dev}, outputs {n_out}')

    def prep(x, m, y):
        return (torch.tensor((x - mean) / std, device=dev), torch.tensor(pitch(m), device=dev),
                torch.tensor(y, device=dev))

    xt, pt, yt = prep(X, M, Y)
    xv, pv, yv = prep(Xv, Mv, Yv)
    net = Net(X.shape[2], X.shape[1], n_out).to(dev)
    print('parameters', sum(p.numel() for p in net.parameters()))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    steps = a.epochs * ((len(X) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)

    def loss_fn(out, y):
        f = torch.sigmoid(out[:, :len(fidx)])
        l = F.l1_loss(f, y[:, fidx], reduction='none').mean(0)
        o, ce = len(fidx), 0.0
        for i, n in zip(cidx, csize):
            ce = ce + F.cross_entropy(out[:, o:o + n], y[:, i].long())
            o += n
        return l.mean() + 0.3 * ce, l

    best = 1e9
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), a.batch):
            b = perm[i:i + a.batch]
            x = xt[b] + 0.05 * torch.randn_like(xt[b])
            loss, _ = loss_fn(net(x, pt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            vl, per = loss_fn(net(xv, pv), yv)
        print(f'epoch {ep + 1:3d} val {vl.item():.4f} (param L1 {per.mean().item():.4f})', flush=True)
        if vl.item() < best:
            best = vl.item()
            torch.save(net.state_dict(), a.out + '.pt')

    net.load_state_dict(torch.load(a.out + '.pt'))
    net.eval()
    with torch.no_grad():
        _, per = loss_fn(net(xv, pv), yv)
    names = [spec['params'][i]['name'] for i in fidx]
    print('per-parameter val L1:', {n: round(float(v), 3) for n, v in zip(names, per.cpu())})

    sd = {k: v for k, v in net.state_dict().items()}
    sd['norm.mean'], sd['norm.std'] = mean, std
    write_nnw(a.out, sd, meta={'synth': spec['name'], 'spec': spec, 'features': meta['features'],
                               'pitch': {'center': 60.0, 'scale': 24.0}, 'frames': int(X.shape[1]),
                               'floats': fidx, 'choices': cidx, 'choiceSizes': csize, 'valLoss': best})
    k = 16
    with torch.no_grad():
        out = net(xv[:k], pv[:k]).cpu().numpy()
    write_nnw(a.out + '.check.nnw', {'features': Xv[:k], 'midi': Mv[:k], 'labels': Yv[:k], 'out': out})
    print('wrote', a.out)


if __name__ == '__main__':
    main()
