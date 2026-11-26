# Music/Audio deep learning use cases

Decided 2026-11-26: models are trained in PyTorch (audio-ml venv), exported with `nn/export.py` to `.nnw`, and run
by the pure-JS `nn/nn.js` (see `doc/nn.md`). qjs-ort (ONNX Runtime) is the cross-check and the fallback for
models too big for pure JS. Audio and MIDI plumbing comes from the qjs-sound modules already installed:
`aubio.so` (onset, pitch, tempo, notes, MFCC, filterbank), `sndfile.so` (read/write audio), `portmidi.so`
(MIDI in/out), `portaudio.so` (realtime), `stk.so`, `sndobj.so`, `labsound.so` (synths/effects that can be
rendered offline for dataset generation), `rubberband.so`/`samplerate.so` (augmentation).

The original list:

 - a model that can compose/jam/improv, either melodies or accompaniment
 - a model that can transcribe notes into arrangements from Aubio generated data
 - a model which measures dynamics and spectral composition which can find mud/phase-error/etc.  in mixes and provides advice
 - a model which learns which parameters of a synth/effect it must tweak to achieve a specific audio result

## Common groundwork (all four cases)

Order matters: the shared pieces make every case cheaper, so they come first.

1. **Dataset layout.** `data/<case>/` (git-ignored, big) holds raw material, `data/<case>/cache/*.npz` the
   extracted features. Every script takes `--seed`, every dataset has a `manifest.json` (source files, feature
   config, git revision of the extractor) so a model can be reproduced.
2. **One feature definition, two implementations.** Features are computed in JS at inference time (aubio via
   `aubio.so`, or plain JS DSP) and must equal what training saw. Rule: the extractor is written **once in JS**,
   run under qjsm to dump the training features (`tools/extract-<case>.js` writes `.nnw`/`.npy`-style float32
   files that PyTorch loads with numpy), so there is no Python port to drift. Where Python needs the
   same function (e.g. synthetic data generation), a fixture test compares both on a few seconds of audio.
3. **Train/export/verify loop** per model, always the same four steps:
   `train.py` -> `export.py` (`.nnw` + `meta` with feature config, vocab, normalisation stats) ->
   `tests/<case>-test.js` runs the JS model on 5-10 stored inputs and compares with PyTorch's saved outputs
   (tolerance 1e-4) -> optional ONNX export compared via `ort.so` (catches layout mistakes).
4. **Normalisation lives in the file.** Means/stds, vocabularies, and sample rate go into `meta`, never into
   JS constants.
5. **Speed budget.** Measure early (TODO item 6): a 2-layer GRU(256) step costs about 0.8 MFLOP; QuickJS does
   roughly 30-100 MFLOP/s in this kind of loop, so ~10-30 ms per step. Budgets below are checked against that;
   if a case needs more, add a native `matvec` (`ort`-independent, tiny) rather than growing the model.
6. **Metrics first.** Every case defines a dumb baseline (below) and a number the model must beat, otherwise
   it is not worth shipping.

---

## Case 1: compose / jam / improvise (melody or accompaniment)

### Goal
Given what has been played so far (live from `portmidi` or from a file), continue it as melody, or generate a
backing part (bass, chords, drums) that follows the played part.

### Sub-modes
- **1a. Melody continuation** - input: past notes of one voice, output: next events. Autoregressive.
- **1b. Accompaniment** - input: melody + chord/key context, output: bass/chord/drum events aligned to it.
  Same architecture, conditioning input added.

### Data
- Source: the author's own MIDI (and whatever exports exist from his DAW projects), converted with a small
  JS script (`tools/midi-to-events.js`). Public sets (Lakh MIDI cleaned, Nottingham, JSB chorales) for
  pre-training; fine-tune on own material. Needed: a few thousand bars minimum for a useful GRU, tens of
  thousands for style.
- Augmentation: transpose +-6 semitones (invert nothing else), tempo jitter. Always split by **song**, never
  by window, or validation is meaningless.

### Representation
Event tokens (one vocabulary, ~150-400 ids), in the style of Performance-RNN:
`NOTE_ON(p)` (128), `NOTE_OFF(p)` (128), `TIME_SHIFT(k)` (k = 1..32 steps of 1/16-note at the file's tempo,
quantised grid) , `VELOCITY(b)` (8-16 bins), plus `BAR`, `BOS`, `EOS`. For 1b add a second token stream
(`COND`: melody events and a chord label per bar) fed through its own `Embedding`, summed or concatenated.
Monophonic melody can use the shorter `(pitch, duration-class)` pairs; decide after the first experiment.

### Model
Embedding(V,64) -> GRU(2 layers, 256) -> Linear(256,V). About 0.7M parameters for V=300. LSTM only if GRU
underfits. No attention (cost, and the context needed is short for jamming). Training: teacher forcing,
cross-entropy, Adam 1e-3, truncated BPTT 128-256 tokens, dropout 0.2 between layers, early stop on validation
loss. 1b: encoder GRU over the conditioning stream, its final state initialises the decoder (or concat per step).

### Inference in JS
`Recurrent.forward(x, state)` one token at a time, `sample(logits, {temperature, topK})` for the next token.
Realtime loop: `portmidi` input -> token -> step -> maybe emit -> `portmidi` output, with the state carried.
Rules that make it musical rather than random: temperature 0.7-1.0, top-k 8-20, forbid `NOTE_OFF` for notes
not sounding, force a legal `TIME_SHIFT` after `BAR`, optional scale mask (zero out logits outside the key).

### Evaluation
- Perplexity on held-out songs vs a bigram baseline.
- Objective musicality checks on 200 generated clips: % notes in key, pitch range, repetition ratio, note
  density vs training distribution.
- Listening test: A/B against the baseline, rated by the author (the only metric that finally matters).

### Milestones
1. `tools/midi-to-events.js` + `tools/events-to-midi.js` roundtrip test (lossless on a quantised file).
2. Train on one genre of own material, perplexity below bigram.
3. Export, JS sampler script that writes a `.mid` (offline), listen.
4. Realtime `portmidi` jam loop with latency measurement (< 20 ms per step target on workstation, check Pi).
5. 1b conditioning, then chord detection as the first conditioning source.

### Risks
Small data -> memorised phrases (check n-gram overlap with training set); timing drift in the quantised grid
(keep swing as a separate token or pre-quantise with a groove template); step time on the Pi.

---

## Case 2: notes into arrangements from aubio data

### Goal
Turn the noisy stream aubio produces (onsets, pitch tracks, tempo/beat, MFCC/energy) into clean note events and
then into an arrangement: note start/end/pitch, then part assignment (bass / lead / chords / drums) and bar
structure. Output is MIDI-like data that Case 1's tokens can consume.

### Stages (each a separate small model so each can be tested alone)
- **2a. Frame tagger.** Per analysis frame decide: note-on, note-sustained (which pitch), silence. Replaces the
  hand-tuned thresholds of aubio's `notes` object.
- **2b. Part assignment.** Group the 2a notes into voices/roles from register, rhythm and timbre features.
- **2c. Quantise/arrange.** Snap to beat grid (aubio tempo), detect bars, emit tokens. Mostly algorithmic;
  a model only if rules fail (decide after 2a/2b).

### Data
- Best supervised source is **synthetic**: take MIDI files, render them with the offline synths
  (`stk.so`/`labsound.so`/`sndobj.so`, several timbres and a little noise/reverb), run aubio over the audio,
  and the MIDI is the ground truth. Unlimited labelled data and the exact JS pipeline is used.
- Real check set: 20-30 recordings of the author's own playing with hand-corrected MIDI.
- Augmentation: random timbre, detune up to 30 cents, reverb, background noise, `rubberband` tempo changes.

### Features (per frame, hop 256 or 512 at 44.1 kHz, same in training and inference)
aubio pitch (Hz -> MIDI float) and confidence, onset detection function, RMS/energy in dB, spectral
centroid/rolloff, 13 MFCC, 12-bin chroma (JS from aubio's spectrum or filterbank). About 30-40 floats/frame.
Normalise with stats stored in `meta`. Add delta features for onsets.

### Model (2a)
Conv1d(feat, 64, k=5) -> ReLU -> BiGRU(2 layers, 96) -> Linear to per-frame logits: `onset` (sigmoid),
`active` (sigmoid), `pitch` (129-way softmax, 128 = none). About 0.3M parameters. Loss: BCE for onset and
active (positive-weighted, onsets are rare) + CE for pitch. Bidirectional is fine because transcription is
offline; for a live variant train a causal GRU with 4-8 frames lookahead.

### Decoding in JS
Threshold onset (peak picking with min distance), take the mode of the pitch posterior over the active span,
end at `active` falling below 0.5, drop notes shorter than 40 ms, merge repeated same-pitch notes unless an
onset separates them. Output: `{start, end, pitch, velocity}` list.

### Evaluation
`mir_eval`-style note F1 (onset tolerance 50 ms, pitch tolerance 50 cents), onset-only F1, frame accuracy;
baseline = aubio's own `notes` output on the same files. Target: F1 clearly above aubio on the real check set.

### Milestones
1. `tools/render-dataset.js`: MIDI -> audio -> aubio frames -> `.npz`-readable floats + labels.
2. Baseline F1 of aubio `notes` on synthetic and real sets.
3. Train 2a on synthetic, test on real, then fine-tune with augmentation / a little real data.
4. JS inference script `tools/transcribe.js in.wav out.mid` and a test with stored frames.
5. 2b role assignment (start with rules: lowest register -> bass), only add a model if rules < 90 %.
6. Feed output through Case 1 tokenisation to prove the chain.

### Risks
Polyphony: aubio pitch is monophonic, so chords need a chroma/multi-pitch head (add a 12- or 88-way sigmoid
"pitch roll" output instead of one softmax); synthetic-to-real gap (mix in real recordings early); frame hop
mismatch between Python and JS (fixture test from the common groundwork).

---

## Case 3: mix analysis (mud, phase errors, dynamics) with advice

### Goal
Analyse a mix or stem and report problems with a location (time range, frequency band) and a plain suggestion,
e.g. "250-400 Hz buildup 0:45-1:20, cut 3 dB around 320 Hz", "L/R correlation -0.4 below 120 Hz: check phase",
"crest factor 5 dB: over-compressed".

### Approach: measurement first, model second
Most of these are measurable. The model's job is the **judgement** (is this buildup a problem *for this
material*?), not the measurement. So:
- **3a. Measurement layer (no ML).** Pure JS DSP, deterministic: per-band energy vs time (octave or 1/3-octave),
  LUFS (short-term/integrated), crest factor, dynamic range, stereo correlation per band, mid/side energy,
  DC offset, clipping count, spectral tilt. Ship this first; it is useful alone and is the model's input.
- **3b. Judgement model.** Classifies segments and bands as OK / problem types, and regresses a suggested gain
  change. Starts as a small MLP, so it is cheap and explainable.

### Data (the hard part)
Labelled mixes do not exist, so create them:
- **Synthetic faults** (primary): take clean/balanced multitracks or well-mastered tracks, apply known faults
  with a script: boost 200-500 Hz by 3-9 dB (mud), harsh 2-5 kHz boost, high-pass removal (thin), invert one
  channel below a crossover (phase), heavy compression/clipping (dynamics), wide-stereo collapse. The applied
  fault and amount are the label; the inverse gain is the "advice" target. Include unmodified controls (50 %).
- **Reference curves**: spectra of the author's own finished tracks as the "good" target distribution, plus a
  genre tag; the model learns the deviation from the genre average.
- **Real labels**: the author tags 30-50 own mixes/segments by ear over time (a small JS tagging tool writing
  JSON). Small, but it is the test set and the calibration set.

### Features
Per 1-second segment: 1/3-octave band levels (31 values, dB), their deviation from the genre mean curve,
short-term LUFS, crest factor, correlation per 6 bands, mid/side ratio per 6 bands, transient density. About
80-100 floats. Context: mean of segment features over the whole track appended (global view).

### Model
MLP 100 -> 128 -> 128 -> heads: `fault` softmax (ok, mud, harsh, thin, phase, over-compressed, clipped, narrow),
`band_gain` regression (31 values, dB, only meaningful for tonal faults), trained with CE + Huber. About 40k
parameters. Optional 1D conv over time (kernel 5 segments) to smooth decisions. Calibrated probabilities
(temperature scaling on validation) so low confidence yields no advice rather than wrong advice.

### Advice generation (JS, rules on top of the model)
Merge adjacent flagged segments, map band-gain output to "cut/boost X dB at F Hz, Q", attach the confidence,
and rank by severity. Text templates, not generated text. Output as JSON and a printable report.

### Evaluation
Per-fault precision/recall on synthetic held-out tracks (split by track), then on the author's tagged real set;
baseline = fixed threshold rules on the 3a metrics (e.g. band over genre mean + 4 dB). The model is kept only
if it lowers false alarms at equal recall; otherwise ship the rules.

### Milestones
1. 3a measurement module + `tools/analyze-mix.js` report (no ML), LUFS/phase checked against a known tool.
2. Fault-injection script and dataset (with controls), baseline rule detector metrics.
3. Train MLP, compare to baseline, calibrate.
4. Tagging tool, collect real labels over time, re-evaluate.
5. Advice text/EQ suggestions, optional export of the suggestion as an EQ preset for the author's plugin chain.

### Risks
Taste is subjective (a "muddy" mix may be intentional): always give confidence and the measurement behind it;
synthetic faults too clean (add several random fault combinations and mild noise); advice that harms (never
auto-apply, report only).

---

## Case 4: learn which synth/effect parameters give a target sound

### Goal
Given a target audio clip (a sample, a recording), predict the parameters of a given synth or effect so that
rendering with them sounds like the target. First target: one specific synth patch architecture (the one the
author uses most; a `stk`/`labsound`/`sndobj` patch to begin with, since those render offline from qjsm).

### Why this comes first
Data is free and exactly labelled: render the synth with random parameters, the parameters are the label.
It exercises the whole pipeline (render, features, train, export, JS inference) with no external data.

### Approach
- **4a. Direct regression (inverse model).** audio features -> parameter vector. Fast, one-shot, one forward
  pass, perfect for pure-JS inference.
- **4b. Refinement by search** (later). Use 4a's prediction as the start of a derivative-free search
  (CMA-ES/hill climbing) in JS: render, compare features with the target, adjust. Fixes 4a's errors where
  parameters are ambiguous. The feature distance is the loss, so no differentiable synth is needed.
- Differentiable synth (DDSP-style) is out of scope for the first version; revisit if 4b is too slow.

### Parameter space
Write a **spec** per synth, `synths/<name>.json`: parameter name, range, scale (linear/log), type
(continuous/choice/bool). Continuous ones are normalised to 0..1 (log where appropriate); choices become
softmax heads. The spec travels in the `.nnw` meta, so the JS side denormalises and applies it.
Start with 8-16 parameters (oscillator shape/mix, detune, filter cutoff/resonance, envelope A/D/S/R, LFO rate/
depth); effects: delay time/feedback/mix, reverb size/damp, drive amount.

### Data generation
`tools/render-synth.js` (qjsm, offline): sample parameters from the spec (uniform in normalised space, plus
oversampling of musically useful regions), render a fixed protocol - e.g. a single note C3, 1.5 s, velocity 100 -
and write the audio and the parameters. 100k-500k examples (disk: feature files, not audio). Randomise the
note pitch in the protocol and give the pitch as an extra input so patches transfer between pitches.
Augment the target side with small noise, level changes, short reverb so real recordings are not out of
distribution. Identical parameters rendering identical audio: a test checks determinism and seeds.

### Features
Log-mel spectrogram (64 mels, hop 512, 3 s max -> ~130 frames x 64) plus a global feature vector (MFCC means,
spectral centroid, flatness, attack time, RMS envelope descriptor). The mel front end is **JS code shared
between training dump and inference** (common groundwork item 2).

### Model
Conv1d stack over time (64 -> 96 -> 128 channels, stride 2, kernel 5, ReLU) -> global mean+max pool -> concat
with pitch and global vector -> MLP 256 -> heads: `params` regression 0..1 (sigmoid, loss: MSE or L1) and one
softmax per choice parameter. About 0.2-0.4M parameters; runs in ~10-50 ms in qjsm (checked in the speed budget).
Better loss than raw parameter error: **audio-feature loss** - render the predicted parameters in the training
loop is too slow, so use parameter loss for training and the feature distance only for evaluation and 4b.

### Evaluation
- Parameter error per parameter (some parameters are inaudible or redundant, so it is only a diagnostic).
- **Spectral distance** between target and the rendering of the predicted parameters (multi-resolution STFT
  and mel-MFCC distance) - the real metric. Baselines: random parameters, nearest neighbour in the training
  set (should be beaten by a wide margin).
- Human check: 20 held-out synthetic targets plus 10 real samples, listen A/B.
- Out-of-domain: render targets from a *different* patch family and see graceful failure (confidence).

### Milestones
1. `synths/<first>.json` and `tools/render-synth.js` with the determinism test.
2. Mel front end in JS + a Python side loader; dump 100k examples; feature stability test.
3. `train/synth-params.py` MLP-only baseline (global features only), then the conv model.
4. Export + JS `tools/match-synth.js target.wav -> params.json`, test with stored examples.
5. Spectral-distance evaluator and the nearest-neighbour baseline; target: median distance below baseline.
6. 4b refinement loop and the second synth/effect; then effects chains.

### Risks
Parameter ambiguity (several settings give the same sound): regression averages them into a wrong middle
- mitigate with 4b refinement, or a mixture-density output / predict a few candidates; synthetic-to-real gap
(augment, evaluate on real samples early); synth non-determinism (seeds, no wall-clock in the patch).

---

## Priority and dependencies

1. Common groundwork items 1-3, and the speed measurement (TODO item 6).
2. **Case 4** (free data, full pipeline, immediate useful result).
3. **Case 2** (reuses 4's rendering and feature tooling; produces tokens for Case 1).
4. **Case 1** (needs data from 2 for audio-derived material, or MIDI directly - can start in parallel once the
   event tokeniser exists, it does not depend on audio).
5. **Case 3** (measurement layer is independent and can be built any time; the model needs the fault-injection
   dataset and real labels, which take longest).

Open questions for the author: which synth to target first in Case 4; what MIDI/DAW material exists for Case 1;
which instruments the transcription of Case 2 should cover first; which DAW/plugin chain the Case 3 advice
should be expressed for.
