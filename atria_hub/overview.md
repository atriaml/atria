---
title: atria_hub
---

# atria_hub

`atria_hub` is Atria's versioned artifact store. It provides a unified interface for pushing and pulling datasets and model snapshots, replacing ad-hoc file sharing with a structured, authenticated, versioned repository.

## The problem it solves

In most ML workflows, sharing a trained model means passing around checkpoint files and separately documenting which config, dataset, and preprocessing was used. When the config changes, documentation drifts. When the checkpoint file is moved, references break.

Atria Hub treats datasets and models as **first-class versioned artifacts**:
- A model snapshot bundles weights + full config — pulling it gives back a fully reproducible pipeline.
- A dataset artifact includes metadata, schema, and provenance — the same dataset can be reproduced from the hub with the same split and preprocessing.

## Module structure

```
atria_hub
├── api/
│   ├── auth.py          ← Login / token management
│   ├── datasets.py      ← Dataset push / pull API
│   ├── models.py        ← Model push / pull API
│   └── health_check.py  ← Hub connectivity check
├── client.py            ← AtriaHubClient (HTTP client)
├── hub.py               ← AtriaHub (top-level facade)
├── config.py            ← Hub URL and storage URL from env
├── credentials_storage.py ← keyring-based credential persistence
└── models.py            ← Pydantic models for hub API responses
```

## Key classes

| Class | Role |
|---|---|
| `AtriaHub` | Facade: wraps all API objects behind a single entry point |
| `AtriaHubClient` | HTTP client with authentication headers |
| `DatasetsApi` | Push / pull dataset artifacts |
| `ModelsApi` | Push / pull model snapshot artifacts |
| `AuthApi` | Login, token refresh, credential management |

## Authentication

Credentials are stored via the OS keyring (using `keyring`). After `atria sign_in`, subsequent CLI calls and Python API calls authenticate automatically without re-entering credentials.

## Relation to atria_models and atria_datasets

- `ModelPipeline.upload_to_hub()` and `ModelPipeline.load_from_hub()` call `ModelsApi` internally.
- Dataset push/pull operations call `DatasetsApi` internally.
- Both use `AtriaHub` as the interface, keeping model and dataset code agnostic of the HTTP layer.
