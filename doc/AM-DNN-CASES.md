# Music/Audio (and two graphics) deep learning use cases

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

## Common groundwork (all cases)

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

## Case 5: vector graphics (icons and glyphs as SVG paths)

Not an audio case: a graphics one, added 2026-11-26. It is the case where `ort.so` earns its place, because the
model is a transformer that pure-JS inference cannot run at a usable speed (see Model).

### Goal
Generate and clean up **flat vector graphics in one style**: icons and glyphs made of filled Bézier paths on a
fixed grid. "Perfect artistic vector graphics" is not a measurable target, so the first target is: at least 80 %
of the outputs parse, have closed paths, render to something recognisable, and have a sensible node count; the
artistic quality is judged by the author through a rating loop (below). Full-colour illustration, gradients and
open-ended text-to-SVG at scale are out of scope (they need a pretrained vision-language model, far beyond a 8 GB GPU).

### Sub-modes
- **5a. Generation.** Class- or text-conditioned: `icon of a key` -> SVG. Autoregressive over path tokens.
- **5b. Vectorizer.** A 64x64 or 128x128 bitmap -> clean SVG with few nodes. A learned replacement for potrace,
  and the better-defined task: the bitmap is the ground truth, so the render loss is exact.
- **5c. Style transfer** between icon sets, only if 5a works: the style is a prefix token or a small embedding.

### Data
- **Fonts first.** OFL-licensed fonts (Google Fonts) give hundreds of thousands of clean glyph outlines with
  consistent winding, few nodes, and a class label (the character). Read with `fontTools` in Python (TrueType
  quadratics are converted to cubics). This is the free, exactly-labelled dataset, the same role synthetic
  rendering plays in Case 4.
- **Icon sets** with permissive licences (Material Icons, Tabler, Lucide, Font Awesome Free; check each licence
  and record it in the dataset manifest) for real iconography, plus the author's own SVGs for the target style.
- Normalise every shape: viewBox 0..1, subpaths flattened to `M L C Z` only (arcs and quadratics to cubics),
  winding made consistent, start point and subpath order canonical (top-left first), transforms baked in, nodes
  merged within a tolerance. Drop shapes with more than 40 segments or self-intersections.
- Pairs for 5b are free: render each normalised SVG to a bitmap (random stroke-free fill, small blur and noise,
  slight affine jitter) and keep the SVG as the label.
- Split by **source file or font family**, never by glyph, or validation leaks style.

### Representation
Tokens, one vocabulary of about 100 ids: commands `M L C Z`, `FILL_EVEN`, `FILL_NONZERO`, `BOS`, `EOS`, `SEP`
(next subpath), and **64 coordinate bins** per axis (a 6-bit grid of the 1x1 box). A cubic is `C` plus six
coordinate tokens, alternating x and y, so a role embedding (x or y, control or end point) is added. A sequence is
capped at 256 tokens (about 35 cubic segments). Colour is a palette index token (16 colours) only if 5c needs it.
Quantisation makes curves slightly jagged: a later refinement head predicts a sub-bin offset per coordinate
(`+-0.5 bin`, regression), so the output is not limited to the grid.

### Model
- **5a:** decoder-only transformer, 6 layers, width 256, 4 heads, learned positions, about 5 M parameters, a
  conditioning prefix of one class or text-embedding token. Trained with next-token cross-entropy, Adam with
  warm-up and cosine decay, dropout 0.1, on the RTX 2070 (a few hours for fonts plus icons).
- **5b:** a small Conv encoder (4 stride-2 blocks) on the bitmap, cross-attention into the same decoder.
- **Inference runs on `ort.so`**, not in `nn.js`: one token step of this model is about 4.7 M MAC, which at the
  measured speed of this `qjsm` (a few M MAC/s) is seconds per token, minutes per icon. ONNX Runtime does it in
  milliseconds. Export with a KV cache (decoder with past-key-values inputs and outputs), which is also what
  `session.run` needs to be called once per token with the cache fed back, the loop of `examples/gru/gru-loop.js`.
  A tiny variant (3 layers, width 128, about 1.2 M MAC per token) is worth trying in pure JS after the measurement of
  TODO item 6, and would need a multi-head attention layer added to `nn/nn.js`.

### Losses and the render feedback
- Stage 1: cross-entropy (plus the offset regression). This teaches syntax and shapes.
- Stage 2: a **render loss**. A differentiable rasteriser turns the (relaxed) predicted paths into a bitmap that
  is compared with the target (5b: L2 and an IoU loss; 5a: a loss against a class-conditional CNN or CLIP
  embedding, later a preference reward). Options, in order: `pydiffvg` if it builds in the Python 3.12 venv, else
  a small soft rasteriser written in PyTorch (flatten each cubic to 16 points, signed-distance coverage with a
  sigmoid edge, winding number by summation). Sampling discrete tokens is not differentiable, so for 5a stage 2 is
  policy-gradient or best-of-n fine-tuning (sample 8, keep the best by reward, train on it), while 5b can use a
  straight-through relaxation or train an offset head by the render loss only.
- Regularisers a designer cares about: node count, curvature smoothness, no self-intersection, closed subpaths.

### Rendering and checks in JS
One exact reference renderer in JS, `tools/svg-render.js` (scanline fill with non-zero and even-odd winding,
anti-aliased with 4x4 supersampling), used for the 5b dataset dump, for previews, and for evaluation; `qjs-nanovg`
or `qjs-opencv` can show and save the result. The soft rasteriser of stage 2 must agree with it: a fixture test
renders 50 shapes with both and requires IoU above 0.98 (the same "one definition, checked" rule as the features
of Case 4).

### Evaluation
- **Validity rate**: parses, closed, within the grid, no NaN.
- **5b**: re-render IoU against the input, chamfer distance of the outlines, node count against potrace on the
  same bitmaps (the model must be at least as accurate with fewer nodes to be worth it).
- **5a**: class accuracy of a small CNN trained on the real rendered icons (does it look like a key?), node count
  and curvature distributions against the training set, novelty (nearest-neighbour distance to the training set,
  to catch memorisation), and diversity at a fixed temperature.
- **Artistic quality**: a small HTML page (`tools/rate-icons.html`) showing a grid of 100 outputs with a rating
  per cell, saved as JSON. The ratings become the evaluation set, and later a preference reward for stage 2.

### Milestones
1. Dataset builder: fonts and icon sets to normalised paths, token files, manifest with licences; a roundtrip
   test (tokens -> SVG -> tokens is lossless on the grid).
2. `tools/svg-render.js` and a viewer page; fixture against a second renderer (Python, `cairosvg` or the soft one).
3. Train 5b on the font pairs with cross-entropy only, measure IoU and node count, compare with potrace.
4. Add the render loss and the offset head, show the gain on IoU.
5. Train 5a on fonts plus icons, validity rate and class accuracy, a first rating round.
6. ONNX export with KV cache, a JS sampler loop on `ort.so` (`tools/generate-icon.js "key"` -> SVG), latency per icon.
7. Stage 2 for 5a: best-of-n or policy-gradient with the rating-trained reward.
8. 5c style prefix, only if the first rating rounds are good.

### Risks
Sequence length (detailed shapes blow the 256-token cap: split into subpath-level generation, or raise the cap
and the cost); quantisation jaggedness (the offset head); self-intersections and winding errors (the regularisers
and a post-process that fixes winding); mode collapse into blobs at low temperature; scraped SVGs with thousands of
nodes polluting the data (the filter above); licences of icon sets (record, and train the shipped model only on
what is allowed); KV-cache ONNX export quirks; 8 GB of GPU memory for the longer sequences (gradient
checkpointing, smaller batch). "Perfect" stays a direction, not a milestone.

---

## Case 6: schematic image to netlist (from book scans and photos)

A second graphics case, added 2026-11-26 and revised the same day: the target images are **figures in books, as
scans and as photos of pages**. That is a harder domain than a clean CAD export, and it changes the plan: the real
images (not synthetic ones) decide the result, a capture and rectification stage comes first, and the author's
own book figures become the labelled test set. The model is not one network but a **pipeline of small models and
exact algorithms**: the networks see (figure, symbols, text, wires) and plain JS code builds the netlist, because
connectivity is a graph problem with a right answer, not something to learn end to end.

### Goal
Take a picture of a book page or figure and produce a **netlist**: the components with their type, reference
designator and value, and the nets with the (component, pin) pairs on each. From the netlist emit what the author
needs: SPICE, a KiCad or EAGLE netlist, and an EAGLE `.scr` through the `eagle-circuit` skill, so a figure from a
book turns into an editable design. A book figure is often **incomplete by design** (values stated in the text, a
part named only by its symbol, an arrow for "to the next stage"), so the output allows unknowns and marks them.
"Perfect" is a direction: the target is a good rate of correct netlists on clean scans and a fast way to correct
the rest (a review page), never silent trust.

### The pipeline
Each stage is trained or written, tested and measured alone; the output of one is the input of the next.
- **6-0. Capture and rectification** (new, mostly classical, in `qjs-opencv`). Find the figure on the page (a
  small classifier says "schematic, other drawing, graph, photo, text") and cut it out; for a photo find the page
  corners and undo the perspective; flatten the lighting (divide by a blurred background), remove halftone dots and
  paper texture, deskew by the dominant line angle, normalise the scale by the estimated symbol size, and keep a
  grayscale image (binarisation destroys thin lines and is only used for the wire stage). A curved page near the
  spine is out of scope at first (it would need a learned dewarper); the capture page asks for a flat page.
- **6a. Symbol detector.** Finds every component and says what it is and how it is turned (box, class, rotation 0,
  90, 180, 270, mirror). The class list follows the **era and field of the books** (open question): resistors (zig-zag
  and box), capacitors (plain, polarised, variable), inductors and transformers (with core lines), diodes, LED, Zener,
  bipolar and field-effect transistors, op-amps, switches, relays, lamps, meters, batteries and cells, loudspeaker
  and microphone, crystals, fuses, ground and supply symbols, ICs as boxes, logic gates, and for older books the
  **vacuum tubes** (diode, triode, pentode).
- **6b. Text reader.** Reads designators (`R1`, `C₂` with a subscript), values (`10 kΩ`, `0,1 µF`, `4u7`, `2K2`),
  supply rails (`+9 V`, `V_CC`), node names (`A`, `OUT`) and, in books, also the **numbers that point to the
  text** (`(a)`, `Fig. 3.4`, a boxed letter). It attaches each text to the nearest symbol or wire.
- **6c. Wire and junction extractor.** A small segmentation net gives a wire mask and a junction heat map with the
  symbols and text masked out, then classical code thins the mask and vectorizes it into straight segments. A scan
  breaks lines (gaps) and smudges them; the stage bridges small collinear gaps. A crossing is a connection only
  when there is a dot, or according to the book's convention (see Risks).
- **6d. Pin locator.** Fixed pin points per symbol class in its canonical frame, moved by rotation and mirror
  (`symbols/<class>.json`); for an IC box the pin stubs and their numbers or names are detected.
- **6e. Connectivity.** Pure JS, no learning: union-find over wire segments that touch at junctions or end on a
  pin, net labels and power symbols merge nets of the same name.
- **6f. Netlist and checks.** Emit the netlist, run electrical rule checks (floating pins, two outputs on one net,
  supply shorted to ground, missing values), attach a confidence to every decision, and mark the doubtful spots for
  the review page.

### Data
The labels of a real book figure do not exist, so the data plan has three layers, in this order of importance:
1. **Real figures from the author's books, labelled with the review page** (below). This is the test set and the
   fine-tuning set. Start with 50 figures to measure the difficulty, aim for 300, stratified by book, by scan
   against photo, and by era. Everything is split **by book**, never by page or figure, or the same drawing style
   is in train and test. The corrected netlist and the symbol/wire positions from the review page are the labels.
2. **Synthetic book-style images** for pre-training, so the networks arrive at the real data already knowing symbols,
   text and wires. Sources: the author's ~295 EAGLE `.sch` files (exact labels for every stage) and open KiCad
   designs, drawn by a JS renderer (`tools/render-eagle.js`, sharing the drawing core of Case 5's `svg-render.js`),
   plus a random circuit generator for rare classes. The renderer has a **book style**: symbol sets of the era
   (zig-zag resistors, tube symbols, open terminals), line weight variation with a slight wobble, **typeset fonts**
   (a serif face, italic designators, subscripts, `Ω` and `µ`), and figure captions and arrows near the drawing.
3. **A degradation pipeline** applied on top, which is what makes synthetic look like a book: halftone and ink
   spread, paper texture and yellowing, show-through of the other page side, scan noise and lines, small rotation,
   perspective and lens blur for photos, shadows and uneven light, JPEG at several qualities, resolution from 100
   to 600 dpi, cropping. It is checked **against real scans by eye with the author** (a contact sheet of 50 real
   and 50 synthetic crops) and by a classifier trained to tell them apart, which should not do better than
   chance before pre-training is trusted.
- A hand-drawn circuit dataset (CGHD is one that exists, check its licence and labels) adds line-style variety.
- **Copyright**: the book scans stay local. `data/schematics/` is git-ignored, the dataset manifest records the
  book and page only, and neither the scans nor weights trained on them are published without checking the books'
  rights. (This is a note to check, not legal advice.)
- **The labelling loop** keeps the effort small: the synthetic-trained model proposes a netlist for a real figure,
  the author corrects it in the review page, the correction becomes a label, the model is fine-tuned, repeat. The
  first 50 figures are slow, later ones get faster as the model learns the books' style.

### Models
- **6a:** a CenterNet- or RetinaNet-style detector with a **pretrained ImageNet backbone** (ResNet-18 sized, about 5
  to 10 M parameters; a pretrained backbone is more robust to scan and photo variation than one trained from scratch)
  and a rotation/mirror head, written with `torchvision` (YOLO-family `ultralytics` code is AGPL, avoid it). The figure
  is resized so a resistor body is at least 24 px; large figures are tiled with 25 % overlap and merged by NMS; the
  scale is estimated first (a quick pass, then the real one at the right scale), and test-time augmentation at two
  scales helps small symbols.
- **6b:** a CRNN with CTC on text crops, about 1 to 2 M parameters, trained on rendered text in many typeset fonts
  (free labels) and on the real crops of the labelled figures, with a character set including `Ω µ k M u n p` and
  subscript handling; Tesseract is the baseline to beat. Decimal commas and the locale of the book matter
  (the language of the books is an open question).
- **6c:** a small U-Net (about 2 M parameters), wire mask and junction map, trained with the degradation pipeline so
  gaps and smudges are in the data; Dice plus BCE on thin lines, focal loss on junctions.
- **Inference runs on `ort.so`** with `qjs-opencv` for the image work (the `u2netp-poc.js` pattern of this
  repository): convolutional nets on images are far too slow for pure JS. Everything from 6d on, the graph and the
  rules, is plain JS and runs in the same engines on Bun, Node and Deno.

### Losses and training
- 6a: focal loss for the heat map, L1 for box size, CE for rotation and class; class balancing (resistors outnumber
  tubes 50 to 1) and copy-paste of rare symbols into other figures.
- 6b: CTC loss. 6c: Dice plus BCE, focal for junctions.
- **Staged training**: pre-train on synthetic with the full degradation, then fine-tune on the real labelled
  figures with a small learning rate and the same augmentation (real data is scarce, so it is oversampled and
  mixed with synthetic in each batch to avoid forgetting).
- **Consistency fine-tune** on the end metric: cases where a wrong crossing or pin decision changes the netlist are
  over-sampled (hard negatives: crossings without dots, T-junctions at pins, parallel close wires).

### Evaluation
- The **test set is real book figures** held out by book, with scans and photos reported separately. Synthetic
  validation is only a sanity check, never the headline number.
- Stage metrics: 6a mAP at IoU 0.5 and rotation accuracy; 6b character and whole-string accuracy by class (designator,
  value, label); 6c wire IoU and junction F1; 6d pin position error in px.
- **Netlist metrics (the ones that matter)**: match predicted to true components by class and position, then compare
  the set of (component, pin, net) triples as net-equivalence classes (a net may be named differently, so compare
  the partitions): pin-connection precision and recall, component F1, value accuracy, and the **exact-netlist rate**.
- Targets written down before training, to be revised after the first 50 real figures: on clean scans pin-connection
  F1 above 0.90 and fewer than five corrections per figure; on photos pin-connection F1 above 0.80. The first
  honest number is the baseline of the synthetic-only model on real figures, and the whole project is measured
  against how far the real fine-tuning moves it.
- An error report that attributes each wrong net to its stage (rectification, missed symbol, wrong rotation, broken
  wire, false junction, misread label), so work goes where the losses are.

### Review page
`tools/review-netlist.html`: the figure with the detections and the nets overlaid in colour, doubtful spots flagged,
click to change a class, rotate a symbol, join or cut a wire, correct a value, mark a part as "value in the text".
Save writes the corrected netlist and the positions, which are the labels for the real set, and the
emit step exports SPICE/KiCad/EAGLE from it. This page is where the labelled real data comes from, so it is built
before the networks (milestone 2).

### Milestones
1. **Corpus survey**: list the books, count figures, scans against photos, eras and languages; pick 20 figures and
   label them by hand with the review page prototype to calibrate how hard the domain is.
2. `tools/review-netlist.html` (view, correct, save) and `tools/render-eagle.js`/`render-kicad.js` (sharing the
   drawing core with Case 5), with the label dumper per stage and a test that a rendered file's netlist matches the
   `.sch`'s own nets.
3. **6e/6f on ground-truth inputs** (true symbols, wires, text) must reproduce the netlist exactly: any failure is a
   code bug found before a network exists. Includes the dot/crossing conventions as a per-figure setting.
4. **The degradation pipeline** and the real-vs-synthetic contact sheet and classifier check; adjust until it
   passes.
5. 6-0 capture and rectification (figure finder, perspective, lighting, deskew, scale) on the real scans and
   photos, judged by eye and by the downstream metric.
6. 6c wire/junction extractor and the vectorization with gap bridging; 6a detector with rotation and scale
   estimation; 6d pin tables; 6b text reader. Pre-train on synthetic, record the baseline on real figures.
7. The whole chain on `ort.so` and `qjs-opencv`: `tools/schematic-to-netlist.js page.jpg out.net`, the netlist
   metrics, the error report, latency per figure.
8. The labelling loop to 300 real figures, fine-tuning rounds, the photo set.
9. Emitters: SPICE, KiCad netlist, EAGLE `.scr` (round trip: figure, netlist, `.scr`, loaded in EAGLE).

### Risks
**Domain gap**: synthetic never fully matches a book (the contact sheet and the classifier check it, and real
fine-tuning is the real fix, so the plan counts on labelling work). Print and capture: halftone and moire,
show-through, thin lines lost in a bad scan, shadows and curvature in photos (curved pages near the spine are out of
scope). Drawing conventions: a dot means connected in one book, a hop-over or no dot means not connected in an old
one (a per-figure setting, ideally detected); buses and labels as the only link between parts; text touching wires;
dashed lines for boxes; **vintage symbol sets and tubes** that no modern library has (add classes as the corpus
demands); non-schematic figures in the same book (the figure classifier must reject graphs and block diagrams, or
accept block diagrams as a different, easier output); multi-unit ICs and pin numbers that exist only in a datasheet;
values in odd notation or only in the surrounding text; text in a language the OCR was not trained on; netlists
that look plausible but are wrong (hence confidences, rule checks and the review page, never silent output);
copyright of the scans. Out of scope at first: multi-sheet hierarchy, simulation of the circuit's behaviour, curved
pages, whole pages with several figures handled automatically (the figure is cut out by hand or the finder).

---

## Priority and dependencies

1. Common groundwork items 1-3, and the speed measurement (TODO item 6).
2. **Case 4** (free data, full pipeline, immediate useful result).
3. **Case 2** (reuses 4's rendering and feature tooling; produces tokens for Case 1).
4. **Case 1** (needs data from 2 for audio-derived material, or MIDI directly - can start in parallel once the
   event tokeniser exists, it does not depend on audio).
5. **Case 3** (measurement layer is independent and can be built any time; the model needs the fault-injection
   dataset and real labels, which take longest).
6. **Case 5** (independent of the audio cases; start with the 5b vectorizer on font pairs once Case 4 has proven
   the train/export/verify loop, and plan the ONNX KV-cache export because the model needs `ort.so`).
7. **Case 6** (schematic to netlist; shares the renderer core with Case 5, and its 6e/6f connectivity code needs no
   model, so it can start with milestones 1 to 3 at any time; the real figures and the review page come first, the
   networks only need the existing `u2netp`-style ORT plus `qjs-opencv` inference pattern, so they do not wait for
   Case 5).

Open questions for the author: which synth to target first in Case 4; what MIDI/DAW material exists for Case 1;
which instruments the transcription of Case 2 should cover first; which DAW/plugin chain the Case 3 advice
should be expressed for; which icon style or sets Case 5 targets and which licences the shipped model may use; for Case 6: which books (era, field: tube
or transistor or digital), the language of their text, how many figures are available as scans against photos, whether
the author wants to label real figures with the review page (the real labels decide the result), whether the
`.sch` files may all be used as training data, and the symbol standard of the books (zig-zag or box resistors).
