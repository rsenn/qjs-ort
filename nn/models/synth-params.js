/*
 * synth-params.js - inference for train/synth-params.py: log-mel features
 * of a note (plus its MIDI pitch) -> synth parameter labels.
 *
 *   const model = SynthParams.load(arrayBuffer);
 *   const labels = model.predict(melData, midi);   // see nn/synths/params.js
 *   const values = denormalize(model.spec, labels);
 *
 * Pure JS, no runtime imports.
 */
import { loadWeights, Conv1d, Linear, apply, softmax, argmax } from '../nn.js';

export class SynthParams {
  constructor(w) {
    const m = (this.meta = w.meta);
    this.spec = m.spec;
    this.features = m.features;
    this.mean = w.tensors['norm.mean'].data;
    this.std = w.tensors['norm.std'].data;
    this.conv = ['conv1', 'conv2', 'conv3'].map((n, i) => Conv1d.from(w, n, { stride: i ? 2 : 1, padding: 2 }));
    this.fc = ['fc1', 'fc2', 'head'].map(n => Linear.from(w, n));
  }
  static load(buffer) {
    return new SynthParams(loadWeights(buffer));
  }
  /* head output -> label vector in spec order (floats sigmoid, choices argmax) */
  decode(out) {
    const { floats, choices, choiceSizes } = this.meta, labels = new Float32Array(this.spec.params.length);
    floats.forEach((p, i) => (labels[p] = 1 / (1 + Math.exp(-out[i]))));
    let o = floats.length;
    choices.forEach((p, i) => {
      labels[p] = argmax(out.subarray(o, o + choiceSizes[i]));
      o += choiceSizes[i];
    });
    return labels;
  }
  /* raw head output for a [frames * nMels] log-mel and a MIDI note */
  forward(mel, midi) {
    const M = this.mean.length, x = new Float32Array(mel.length);
    for(let i = 0; i < mel.length; i++) x[i] = (mel[i] - this.mean[i % M]) / this.std[i % M];
    let h = x;
    for(const c of this.conv) h = apply(c.forward(h), 'relu');
    const flat = new Float32Array(h.length + 1);
    flat.set(h);
    flat[h.length] = (midi - this.meta.pitch.center) / this.meta.pitch.scale;
    h = apply(this.fc[0].forward(flat), 'relu');
    h = apply(this.fc[1].forward(h), 'relu');
    return this.fc[2].forward(h);
  }
  predict(mel, midi) {
    return this.decode(this.forward(mel, midi));
  }
}
