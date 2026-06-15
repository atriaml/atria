---
title: Installation
---

# Installation

Atria is a monorepo. Each package can be installed independently; install only what you need.

## Core packages

=== "pip"

    ```bash
    pip install atria-registry atria-types atria-datasets atria-models atria-ml atria-insights atria-hub atria-cli
    ```

=== "uv"

    ```bash
    uv add atria-registry atria-types atria-datasets atria-models atria-ml atria-insights atria-hub atria-cli
    ```

## Individual packages

Install a single package when you only need part of the framework:

```bash
# just the registry and types, no ML dependencies
pip install atria-registry atria-types

# datasets only
pip install atria-datasets

# models + transforms
pip install atria-models atria-transforms
```

## Development (from source)

```bash
git clone https://github.com/atriaml/atria
cd atria
uv sync --all-packages
```
