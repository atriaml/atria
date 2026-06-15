---
title: atria_cli
---

# atria_cli

`atria_cli` is the unified command-line entry point for all Atria workflows. It exposes dataset preparation, model management, and hub operations as CLI commands — each backed by the same config and registry system used by the Python API.

## Design philosophy

The CLI maps directly to task configs. Passing a YAML or JSON config file is all that's needed to run any workflow:

```bash
atria train --config train_config.yaml
atria evaluate --config eval_config.yaml
atria explain --config explanation_config.yaml
```

This means:
- **No Python boilerplate** for common operations.
- **Reproducibility**: every run is fully described by its config file.
- **Remote execution**: configs can be sent to a remote machine and executed identically.

## Commands

```
atria
├── sign_in          ← Authenticate with Atria Hub
├── sign_out         ← Remove stored credentials
├── sign_up          ← Create a new hub account
├── datasets
│   ├── prepare_and_upload  ← Download, process, cache, and push to hub
│   └── download            ← Pull a dataset artifact from hub
└── models
    ├── upload       ← Push a model snapshot to hub
    └── download     ← Pull a model snapshot from hub
```

## Implementation

`atria_cli` uses [Python Fire](https://github.com/google/python-fire) to turn the command dict into a CLI. Each command is a plain Python function that receives CLI arguments and constructs the appropriate config or API call.

The entry point is:

```python
if __name__ == "__main__":
    fire.Fire({
        "sign_in": sign_in,
        "datasets": {"prepare_and_upload": ..., "download": ...},
        "models": {"upload": ..., "download": ...},
    })
```

This keeps commands as regular functions testable in isolation, with Fire handling argument parsing.
