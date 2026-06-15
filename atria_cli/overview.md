---
title: atria_cli
---

# atria_cli

`atria_cli` is the command-line entry point for Atria Hub operations. It exposes authentication, dataset publishing, and model snapshot management as CLI commands — each backed by the same Python API used programmatically.

## Commands

```
atria
├── sign_in          ← Authenticate with Atria Hub (email + password)
├── sign_out         ← Remove stored credentials
├── sign_up          ← Create a new hub account
├── datasets
│   ├── prepare_and_upload  ← Prepare a registered dataset and push to hub
│   └── download            ← Pull a dataset artifact from hub
└── models
    ├── upload       ← Upload a model snapshot directory to hub
    └── download     ← Pull a model snapshot from hub
```

## Implementation

`atria_cli` uses [Python Fire](https://github.com/google/python-fire) to turn the command dict into a CLI. Each command is a plain Python function that receives CLI arguments:

```python
fire.Fire({
    "sign_in": sign_in,
    "sign_out": sign_out,
    "sign_up": sign_up,
    "datasets": {"prepare_and_upload": ..., "download": ...},
    "models": {"upload": ..., "download": ...},
})
```

This keeps commands as regular functions testable in isolation, with Fire handling argument parsing.

## Training and evaluation

Training and evaluation are not CLI commands — they are run directly via Python using `Trainer`, `Evaluator`, and `ModelExplainer` with the appropriate `TaskConfig`. The CLI handles only hub operations (authentication, artifact push/pull).
