---
title: Hub Client
---

# Hub Client

## AtriaHubClient

`AtriaHubClient` is the low-level HTTP client that all hub API objects use. It handles:

- **Base URL configuration** — from `settings.ATRIAX_URL` (env var or default)
- **Storage URL** — separate URL for binary artifact storage (`settings.ATRIAX_STORAGE_URL`)
- **Authentication** — attaches bearer tokens from the keyring to every request
- **Connection checking** — `AtriaHubConnectionError` when the hub is unreachable

## AtriaHub

`AtriaHub` is the high-level facade. It initializes all API objects from a single client:

```python
hub = AtriaHub()
hub.datasets   # DatasetsApi
hub.models     # ModelsApi
hub.auth       # AuthApi
hub.health_check  # HealthCheckApi
```

Most user-facing code interacts with `AtriaHub` via the CLI or via `ModelPipeline.upload_to_hub()` / `ModelPipeline.load_from_hub()` — direct use of the client is rarely needed.

## API objects

| API | Key operations |
|---|---|
| `DatasetsApi` | `get_or_create()`, `upload_files()`, `download_files()`, `validate()` |
| `ModelsApi` | `get_or_create()`, `upload_snapshot()`, `download_files()`, `validate()` |
| `AuthApi` | Login and token management via Supabase |
| `RepoCredentialsApi` | Repository-level access tokens for LakeFS storage |
| `HealthCheckApi` | `health_check()` — checks hub availability |

## Credential storage

After login, credentials are stored in the OS keyring (via the `keyring` library):
- On Linux: `libsecret` or `kwallet`
- On macOS: `Keychain`
- On Windows: `Windows Credential Manager`

The `use_key_ring=True` default means credentials persist across sessions and processes. Pass `use_key_ring=False` to use in-memory credentials only (useful in CI environments).

## Environment configuration

| Env var | Default | Description |
|---|---|---|
| `ATRIAX_URL` | `http://127.0.0.1:8000` | Hub API base URL |
| `ATRIAX_STORAGE_URL` | `http://127.0.0.1:8090` | LakeFS binary artifact storage URL |
