---
title: Installation
---

# Installation

Atria is a monorepo of local packages managed with [uv](https://docs.astral.sh/uv/). The packages are not published to PyPI — installation is from source.

## From source (recommended)

```bash
git clone https://github.com/atriaml/atria
cd atria
uv sync
```

`uv sync` reads the workspace `pyproject.toml`, resolves all local packages (`atria_registry`, `atria_types`, `atria_datasets`, etc.) from their subdirectories, and installs them as editable packages into the project's virtual environment.

## Activating the environment

```bash
# activate the uv-managed venv
source .venv/bin/activate

# or run commands directly through uv
uv run python -c "import atria_registry"
```

## Individual packages (from their subdirectory)

If you only need one module, you can install it in isolation from its own directory:

```bash
cd atria_registry
uv sync
```

Note that most modules depend on `atria_registry` and `atria_types` at minimum, so their own `pyproject.toml` will pull those in automatically.

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/) (`pip install uv` or `curl -Ls https://astral.sh/uv/install.sh | sh`)
