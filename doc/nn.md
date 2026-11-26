# nn/ - pure-JS inference

Train in PyTorch, export, run in qjsm/node/bun/deno with no native code.

    from export import write_nnw           # nn/export.py
    write_nnw('model.nnw', model.state_dict(), meta={'hidden': 64})

    import * as nn from './nn/nn.js';
    const w = nn.loadWeights(buffer);       // buffer: ArrayBuffer of the .nnw
    const gru = nn.Recurrent.from(w, 'rnn', 'gru', { bidirectional: true });
    const { y, state } = gru.forward(frames);   // frames: Float32Array [T*in]

Layers: `Linear`, `Embedding`, `LayerNorm`, `Conv1d`, `Recurrent` ('gru' | 'lstm'), plus `apply(x, 'relu'|
'tanh'|'sigmoid'|'silu'|'gelu')`, `softmax`, `sample`, `argmax`. Batch is always 1; sequences are `[T, features]`
row-major. `Recurrent.forward(x, state)` continues from `state` for frame-by-frame realtime use.
Tests: `qjsm tests/nn-test.js` (regenerate fixtures with `tests/nn/make-fixtures.py`).
