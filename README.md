# GPT-2 From Scratch

An independent, educational implementation of GPT-2 in PyTorch, developed while working through [ARENA 1.1: Transformers from Scratch](https://arena-chapter1-transformer-interp.streamlit.app/).

> **Status:** Scaffold only. The model classes are intentionally left unimplemented so I can learn by writing them myself.

## Project layout

```text
gpt2/
├── model.py                # Implement GPT-2 here — no solutions included
├── tests/
│   ├── test_environment.py # Basic environment smoke checks
│   └── README.md
├── notebooks/
│   └── scratch.ipynb       # Optional tensor experiments and notes
├── pyproject.toml          # uv project dependencies
├── uv.lock                 # Created by first `uv sync` (not bundled)
├── .python-version        # Python 3.12
└── .gitignore
```

## Setup: uv + VS Code (macOS, Linux, or Windows via equivalent commands)

Install [uv](https://docs.astral.sh/uv/) first if it is not installed.

From the **`gpt2/`** directory:

```bash
uv python install 3.12
uv sync
uv run python model.py
uv run pytest
```

`uv sync` creates `.venv/`, resolves dependencies, and generates `uv.lock` automatically. Commit the resulting `uv.lock` to Git for reproducibility. The lockfile is **not included** in this starter archive because dependencies could not be resolved in the build environment. Select the `gpt2/.venv/bin/python` interpreter in VS Code (on Windows, `.venv\Scripts\python.exe`).

**Mac / MPS:** `model.py` checks for Apple Metal (MPS), then CUDA, then CPU. This is only device selection; moving tensors and model parameters to the chosen device is part of the exercises.

If you prefer activating your environment manually:

```bash
source .venv/bin/activate
python model.py
pytest
```

## Independent learning workflow

1. Read a section of ARENA 1.1.
2. Implement the corresponding class or function in `model.py` **without copying reference solutions**.
3. Create small experiments in `notebooks/scratch.ipynb`.
4. Add shape, value, and behavior tests to `tests/`.
5. Later compare against GPT-2 weights via TransformerLens.

Suggested component order (following ARENA's own sequence): `Config`, `LayerNorm`, `Embed`, `PosEmbed`, `Attention`, `MLP`, `TransformerBlock`, `DemoTransformer`.

## ARENA reference checkout (separate repository)

If you want their provided exercises and tests locally, place ARENA alongside (not inside) your `gpt2` repository:

```text
workspace/
├── gpt2/                # Your GitHub project
└── ARENA_materials/     # Independent reference checkout
```

```bash
cd ..
git clone https://github.com/ARENA-education/ARENA_materials.git
```

Some ARENA tests import their own `solutions.py` and rely on the ARENA directory layout. This project does **not** bundle or modify those files. Test against upstream independently when useful.

## Attribution

Inspired by the open-source [ARENA materials](https://github.com/ARENA-education/ARENA_materials). This repository is intended to contain my **own** implementation, tests, and notes, not copied solutions or exercise source files. Consult ARENA's license before redistributing their code.
