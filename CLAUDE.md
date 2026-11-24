# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **QuickJS native C module** that exposes [ONNX Runtime](https://onnxruntime.ai/) to JavaScript through its
C API (`onnxruntime_c_api.h`, no C++ API needed). It builds to a shared library `ort.so` that `qjsm` loads on
`import ... from 'ort'`. Purpose: run models the author trains himself (sound design, synth parameters,
basslines, accompaniments) on the workstation, the Raspberry Pi and the phone (Termux), and, through the
N-API adapter of `~/Sources/js-engines`, on Bun, Node and Deno.

Status: project skeleton only; read `TODO.md` for what comes next.

## Build & run

Not wired yet (first task in `TODO.md`). The intended shape, as in the sibling projects:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Debug     # -DORT_PREFIX=<onnxruntime install> if not found
cmake --build build
QUICKJS_MODULE_PATH=build qjsm tests/ort-test.js     # qjsm, NOT plain qjs
```

Use the `quickjs-native-bindings` skill for any C code here (class registration, ownership, finalizers,
CMake layout). Match the neighbours: `~/Sources/quickjs/qjs-nanovg` is the lean reference project.

## Track tasks, changes and bugs

- **ChangeLog**: every change to the project gets an entry in the same step as the change, newest last.
  Format: a header line `YYYY-MM-DD  Roman Senn  <roman.l.senn@gmail.com>`, a blank line, then the text
  indented by one tab: a summary sentence, a blank line, and `* file (function):` paragraphs saying what changed
  and why. Wrap at about 72 columns. Do not edit old entries.
- **TODO.md**: open work only, grouped under headings. Add an item when you find work you are not doing now;
  delete it (not tick it) when it is done, the ChangeLog keeps the history. Mark the one in progress.
- **BUGS**: when you discover a bug that is not the one you are fixing, append it to the end of `BUGS`
  instead of fixing it inline or ignoring it. Do not fix unrelated bugs without asking. Format:

  ```
  - <canonical-name>: <description, wrapped prose: what is wrong, how it was found, root cause if known>

      <JS repro that triggers it>

  ```

  A blank line between the description and the repro, and one after the repro before the next entry.
  Delete the entry when the bug is fixed (the ChangeLog records the fix).

## Conventions

- File names `quickjs-ort.c` (or `ort-*.c` split), identifiers prefixed `ort_`/`js_ort_`, class ids
  `js_ort_<class>_class_id`; JS names mirror the ONNX Runtime ones (`Session`, `Tensor`, `run`, `inputNames`).
- Every class gets `Symbol.toStringTag`. Native objects hold `OrtValue*`/`OrtSession*`; finalizers call the
  matching `Release*` and tolerate NULL.
- ONNX Runtime is not thread-safe with QuickJS: never call into JS from an ORT thread.
- Docs in `doc/native/ort.md`, tests in `tests/` (tinytest style if the tree has `tinytest.js`).
- Comments follow the `comment-style` skill; C formatting follows `.clang-format` once one is added
  (copy it from qjs-nanovg).

## C style

Apply to every new or changed C line. The quickjs-native-bindings skill has the base rules, these are the
project's own and win where they differ.

- **Declarations come first.** Every variable is declared at the beginning of the block scope it lives in,
  before the first statement of that block. Never declare in the middle of a block, not even right before the
  first use. The one exception is the loop variable of a `for(` header. A value that is known at the top is
  initialised in its declaration. An empty line separates the declarations of a block from its first statement.
  A declaration whose initialiser is only tested by the very next statement is not initialised there: it is
  declared, and the call moves into the test (next rule).
- **Fold an assignment into the test** when the assigned value is only used for that test or right after it:
  the call is the condition, wrapped in its own parentheses, so a status, a pointer or a lookup result is never
  assigned on one line and tested on the next. Keep the unfolded form only where the folded line would be hard
  to read.
- **A closing brace of an inner block is followed by an empty line.** This holds for every block that closes
  inside a function: `if`, `for`, `while`, `switch`, a plain block. It does not hold when the next token
  continues the same statement (`else`, the `while` of a `do`) or closes the enclosing block.

## Git commits

Do not add the `Co-Authored-By:` or `Claude-Session:` trailers (or any other attribution lines) to commit
messages in this project, even when Claude Code's attribution guidance asks for them. Commit and push only
when the user asks.
